import json
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from .config import DATA_ROOT, DATABASE_URL

DATA_ROOT.mkdir(parents=True, exist_ok=True)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 30})


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record) -> None:
    # WAL lets the API read while the worker writes progress.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


# Columns added after the first release; create_all() doesn't alter existing tables.
_ADDED_COLUMNS = {
    "contribution": {"detail": "VARCHAR", "timeline": "JSON"},
    "version": {
        "splat_kind": "VARCHAR DEFAULT ''",
        "train_status": "VARCHAR",
        "train_quality": "VARCHAR",
        "train_auto": "BOOLEAN DEFAULT 0",
        "train_progress": "FLOAT DEFAULT 0",
        "train_eta": "FLOAT",
        "train_detail": "VARCHAR",
        "train_timeline": "JSON",
        "train_error": "VARCHAR",
        "preview_url": "VARCHAR",
        "preview_step": "INTEGER",
        "updated_at": "DATETIME",
        "dense_url": "VARCHAR DEFAULT ''",
        "splats": "JSON",
    },
}


def init_db() -> None:
    from . import models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        for table, columns in _ADDED_COLUMNS.items():
            existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})")}
            for column, sql_type in columns.items():
                if column not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
        # Dense clouds used to occupy the splat slot (splat_kind "dense"); move them out.
        conn.exec_driver_sql(
            "UPDATE version SET dense_url = splat_url, splat_url = '', splat_path = '', splat_kind = '' "
            "WHERE splat_kind = 'dense'"
        )
        _backfill_splats(conn)


def _backfill_splats(conn) -> None:
    """Versions from before `splats` existed: collect the kinds trained on disk."""
    rows = conn.exec_driver_sql(
        "SELECT version.id, version.number, scene.slug, version.splat_kind, version.splat_url FROM version "
        "JOIN scene ON scene.id = version.scene_id WHERE version.splats IS NULL"
    ).fetchall()
    for version_id, number, slug, kind, url in rows:
        splats = {}
        for lod in sorted((DATA_ROOT / "scenes" / slug / "versions" / f"v{number}").glob("*/lod/splat-lod.rad")):
            splats[lod.parent.parent.name] = "/files/" + lod.relative_to(DATA_ROOT).as_posix()
        if kind and url:
            splats[kind] = url
        conn.exec_driver_sql("UPDATE version SET splats = ? WHERE id = ?", (json.dumps(splats), version_id))


@contextmanager
def session_scope() -> Iterator[Session]:
    with Session(engine, expire_on_commit=False) as session:
        yield session
