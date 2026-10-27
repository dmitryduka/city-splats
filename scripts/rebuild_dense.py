"""Rebuild a scene's dense clouds in version order (after changing densification settings, e.g.
the scene's DEPTH_MODEL or DENSE_STRIDE overrides), reusing each version's reconstruction;
registration is not rerun. A version's frames are the ones its reconstruction added to its
parent's, so this also works for a scene cloned with scripts.clone_scene.

  python -m scripts.rebuild_dense <scene-slug>
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pycolmap
from sqlmodel import select

from backend.app.config import DATA_ROOT, SCENES_ROOT, files_url
from backend.app.db import session_scope
from backend.app.models import Contribution, Scene, Version, utcnow
from worker.runner import _densify


class _ConsoleProgress:
    def update(self, fraction: float, detail: str | None = None) -> None:
        print(f"\r  {fraction:5.0%} {detail or ''}\033[K", end="", flush=True)


def _names(path: str) -> set[str]:
    return {image.name for image in pycolmap.Reconstruction(DATA_ROOT / path).images.values() if image.has_pose}


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
        names = sorted(_names(version.reconstruction_path) - (_names(parent.reconstruction_path) if parent else set()))
        rec = pycolmap.Reconstruction(DATA_ROOT / version.reconstruction_path)
        image_root = SCENES_ROOT / Path(version.reconstruction_path).parts[1] / "images"
        # The parent was re-read after its own rebuild, so its tiles and voxel are current.
        metrics = dict(version.metrics)
        started = time.time()
        print(f"v{version.number} ({contribution.filename}, {len(names)} frames)")
        tiles = _densify(parent, rec, image_root, names, work, metrics, _ConsoleProgress(), settings=scene.settings)
        with session_scope() as session:
            for row in (session.get(Version, version.id), session.get(Contribution, version.contribution_id)):
                row.metrics = {k: v for k, v in row.metrics.items() if k != "dense_failed"} | {
                    k: metrics[k] for k in ("dense", "dense_tiles", "dense_voxel")}
                row.updated_at = utcnow()
                session.add(row)
            v = session.get(Version, version.id)
            v.dense_url = files_url(tiles)
            session.commit()
        print(f"\n  {metrics['dense']} in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
