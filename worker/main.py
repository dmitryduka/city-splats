"""Worker loop: processes queued contributions one at a time, in upload order, and trains
queued version splats when no video is waiting.

Registration is sequential on purpose: every video registers against the scene HEAD left by
the previous one. Frame extraction and depth prediction don't depend on HEAD, so they run
ahead for queued videos (worker.prefetch).
usage: python -m worker.main
"""

from __future__ import annotations

import logging
import time
import traceback

from sqlmodel import select

from backend.app.config import DECODE_THREADS, INGEST_VIDEOS, SFM_VIDEOS
from backend.app.db import init_db, session_scope
from backend.app.models import Contribution, Scene, Version, utcnow
from worker import compute
from worker.prefetch import Prefetcher
from worker.runner import process, train_version

log = logging.getLogger("worker")


def recover_interrupted() -> None:
    """A crash mid-job leaves 'processing' rows behind; requeue them (work dirs are rebuilt)."""
    with session_scope() as session:
        for c in session.exec(select(Contribution).where(Contribution.status == "processing")).all():
            log.warning("requeueing interrupted contribution %s", c.id)
            c.status, c.stage, c.progress, c.preview_url = "queued", None, 0.0, None
            session.add(c)
        for v in session.exec(select(Version).where(Version.train_status.in_(["training", "cancelling"]))).all():
            log.warning("requeueing interrupted training of version %s", v.id)
            v.train_status = "queued" if v.train_status == "training" else None
            v.preview_url = v.preview_step = None
            session.add(v)
        session.commit()


def claim_next() -> int | None:
    with session_scope() as session:
        c = session.exec(
            select(Contribution).where(Contribution.status == "queued").order_by(Contribution.id)
        ).first()
        if c is None:
            return None
        c.status, c.started_at, c.updated_at, c.error = "processing", utcnow(), utcnow(), None
        session.add(c)
        scene = session.get(Scene, c.scene_id)
        scene.updated_at = utcnow()
        session.add(scene)
        session.commit()
        return c.id


def claim_training() -> int | None:
    """Oldest queued training; automatic jobs for versions that are no longer HEAD are
    dropped (a newer version will train instead)."""
    with session_scope() as session:
        for v in session.exec(select(Version).where(Version.train_status == "queued").order_by(Version.id)).all():
            scene = session.get(Scene, v.scene_id)
            if v.train_auto and scene.head_version_id != v.id:
                v.train_status, v.train_detail, v.updated_at = None, "superseded by a newer version", utcnow()
                session.add(v)
                continue
            v.train_status, v.train_error, v.train_progress, v.updated_at = "training", None, 0.0, utcnow()
            session.add(v)
            session.commit()
            return v.id
        session.commit()
        return None


def run_training(version_id: int) -> None:
    log.info("training splat for version %s", version_id)
    try:
        train_version(version_id)
    except Exception as exc:  # noqa: BLE001 - record any failure on the version
        log.exception("training of version %s failed", version_id)
        with session_scope() as session:
            v = session.get(Version, version_id)
            v.train_status = "failed"
            v.train_error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=5)}"
            v.train_eta, v.preview_url, v.preview_step, v.updated_at = 0, None, None, utcnow()
            session.add(v)
            session.commit()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    init_db()
    log.info("worker up; device=%s", compute.device())
    recover_interrupted()
    prefetch = Prefetcher(INGEST_VIDEOS, DECODE_THREADS, SFM_VIDEOS)
    prefetch.start()
    while True:
        contribution_id = claim_next()
        if contribution_id is None:
            version_id = claim_training()
            if version_id is None:
                time.sleep(2)
            else:
                run_training(version_id)
            continue
        log.info("processing contribution %s", contribution_id)
        try:
            process(contribution_id, prefetch)
        except Exception as exc:  # noqa: BLE001 - record any failure on the job
            log.exception("contribution %s failed", contribution_id)
            with session_scope() as session:
                c = session.get(Contribution, contribution_id)
                c.status = "failed"
                c.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=5)}"
                c.eta_seconds = 0
                c.finished_at = c.updated_at = utcnow()
                session.add(c)
                session.commit()
        finally:
            with session_scope() as session:
                c = session.get(Contribution, contribution_id)
                prefetch.forget(c, session.get(Scene, c.scene_id).slug)


if __name__ == "__main__":
    main()
