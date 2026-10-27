"""Rebuild a scene's dense clouds in version order (after changing densification settings),
reusing each version's reconstruction; registration is not rerun.

  python -m scripts.rebuild_dense <scene-slug>
"""

from __future__ import annotations

import json
import logging
import sys
import time

import pycolmap
from sqlmodel import select

from backend.app.config import DATA_ROOT, SCENES_ROOT, files_url
from backend.app.db import session_scope
from backend.app.models import Contribution, Scene, Version, utcnow
from worker.runner import _densify


class _ConsoleProgress:
    def update(self, fraction: float, detail: str | None = None) -> None:
        print(f"\r  {fraction:5.0%} {detail or ''}\033[K", end="", flush=True)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    slug = sys.argv[1]
    with session_scope() as session:
        scene = session.exec(select(Scene).where(Scene.slug == slug)).one()
        versions = session.exec(select(Version).where(Version.scene_id == scene.id).order_by(Version.number)).all()
    for version in versions:
        with session_scope() as session:
            parent = session.get(Version, version.parent_id) if version.parent_id else None
            contribution = session.get(Contribution, version.contribution_id)
        work = SCENES_ROOT / slug / "contributions" / contribution.image_dir_name
        frames = json.loads((work / "frames.json").read_text())
        names = [f"{contribution.image_dir_name}/{f['name']}" for f in frames]
        rec = pycolmap.Reconstruction(DATA_ROOT / version.reconstruction_path)
        # The parent was re-read after its own rebuild, so its cloud and voxel are current.
        metrics = dict(version.metrics)
        started = time.time()
        print(f"v{version.number} ({contribution.filename}, {len(names)} frames)")
        rad = _densify(parent, rec, SCENES_ROOT / slug / "images", names, work, metrics, _ConsoleProgress())
        with session_scope() as session:
            v = session.get(Version, version.id)
            v.metrics = metrics
            v.dense_url = files_url(rad)
            v.updated_at = utcnow()
            session.add(v)
            session.commit()
        print(f"\n  {metrics['dense']} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
