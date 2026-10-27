"""Seed the "room" scene from the M0 spike outputs (two videos already processed).

usage: python -m scripts.seed_room
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pycolmap
from sqlmodel import select

from backend.app.config import BUILD_LOD_BIN, REPO_ROOT, SCENES_ROOT, files_url
from backend.app.db import init_db, session_scope
from backend.app.models import Contribution, Scene, Version, utcnow
from worker.pipeline.export import export_layers
from worker.pipeline.lod import build_lod
from worker.progress import rel

SPIKE = REPO_ROOT / "data/spike"
VIDEOS = [
    REPO_ROOT / "videos/room/20261021_003502.mp4",
    REPO_ROOT / "videos/room/20261021_013509.mp4",
]


def spike_name_to_scene(name: str, c1: int, c2: int) -> str:
    return f"c{c2}/{name[3:]}" if name.startswith("v2/") else f"c{c1}/{name}"


def rename_images(db_src: Path, rec_src: Path, out_dir: Path, rename) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "database.db"
    shutil.copyfile(db_src, db)
    with sqlite3.connect(db) as conn:
        for image_id, name in conn.execute("SELECT image_id, name FROM images").fetchall():
            conn.execute("UPDATE images SET name = ? WHERE image_id = ?", (rename(name), image_id))
    rec = pycolmap.Reconstruction(rec_src)
    for image in rec.images.values():
        image.name = rename(image.name)
    rec_dir = out_dir / "sparse" / "0"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec.write(rec_dir)
    return db, rec_dir


def main() -> None:
    init_db()
    combined_ply = SPIKE / "brush_out2/splat_5000.ply"
    if not combined_ply.exists():
        raise SystemExit(f"{combined_ply} missing: wait for the combined Brush run to finish")

    with session_scope() as session:
        if session.exec(select(Scene).where(Scene.slug == "room")).first():
            raise SystemExit("scene 'room' already exists")
        scene = Scene(slug="room", name="Room")
        session.add(scene)
        session.flush()
        contributions = []
        for seq, video in enumerate(VIDEOS, start=1):
            c = Contribution(
                scene_id=scene.id, seq=seq, filename=video.name, video_path=str(video), status="processing"
            )
            session.add(c)
            contributions.append(c)
        session.commit()
        c1, c2 = (c.id for c in contributions)

    scene_dir = SCENES_ROOT / "room"
    images = scene_dir / "images"
    shutil.rmtree(images / f"c{c1}", ignore_errors=True)
    shutil.copytree(SPIKE / "images", images / f"c{c1}", ignore=shutil.ignore_patterns("v2"))
    shutil.copytree(SPIKE / "images/v2", images / f"c{c2}", dirs_exist_ok=True)

    rename = lambda name: spike_name_to_scene(name, c1, c2)  # noqa: E731
    color = {f"c{c1}": 0, f"c{c2}": 1}
    steps = [
        (contributions[0], SPIKE / "sfm_sift/database.db", SPIKE / "sfm_sift/0",
         SPIKE / "brush_out/splat_7000.ply", SPIKE / "frames.json"),
        (contributions[1], SPIKE / "v2/database.db", SPIKE / "v2/sparse/0",
         combined_ply, SPIKE / "frames_v2.json"),
    ]
    parent: Version | None = None
    for c, db_src, rec_src, ply_src, frames_json in steps:
        work = scene_dir / "contributions" / f"c{c.id}"
        db, rec_dir = rename_images(db_src, rec_src, work / "sfm", rename)
        shutil.copyfile(frames_json, work / "frames.json")
        splat = work / "splat.ply"
        shutil.copyfile(ply_src, splat)
        rad = build_lod(BUILD_LOD_BIN, splat, work / "lod")
        rec = pycolmap.Reconstruction(rec_dir)
        export_layers(rec, work / "layers", lambda name: color[name.split("/")[0]])
        frames = json.loads(frames_json.read_text())
        new = [i for i in rec.images.values() if i.name.startswith(f"c{c.id}/")]
        metrics = {
            "device": "mps",
            "keyframes": len(frames),
            "holdouts": sum(f["holdout"] for f in frames),
            "registered": sum(i.has_pose for i in new),
            "registration_ratio": round(sum(i.has_pose for i in new) / len(frames), 3),
            "points": rec.num_points3D(),
            "train_steps": 7000 if parent is None else 5000,
            "warm_start": parent is not None,
            "seeded_from_spike": True,
        }
        with session_scope() as session:
            version = Version(
                scene_id=c.scene_id,
                number=1 if parent is None else parent.number + 1,
                parent_id=parent.id if parent else None,
                contribution_id=c.id,
                contribution_ids=(parent.contribution_ids if parent else []) + [c.id],
                database_path=rel(db),
                reconstruction_path=rel(rec_dir),
                splat_path=rel(splat),
                splat_url=files_url(rad),
                layers_url=files_url(work / "layers"),
                metrics=metrics,
            )
            session.add(version)
            session.flush()
            row = session.get(Contribution, c.id)
            row.status, row.stage, row.progress, row.eta_seconds = "done", "published", 1.0, 0.0
            row.version_id, row.metrics = version.id, metrics
            row.started_at = row.finished_at = row.updated_at = utcnow()
            scene_row = session.get(Scene, c.scene_id)
            scene_row.head_version_id, scene_row.updated_at = version.id, utcnow()
            session.add_all([row, scene_row])
            session.commit()
            parent = version
        print(f"version {version.number}: {rec.num_reg_images()} images, {rec.num_points3D()} points -> {rad}")


if __name__ == "__main__":
    main()
