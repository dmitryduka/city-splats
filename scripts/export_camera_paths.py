"""Write camera_paths.json (worker.pipeline.camera_paths) next to the viewer layers of versions
made before the worker wrote it, so their videos can be replayed. Reads the database only.

  python -m scripts.export_camera_paths <scene-slug> [version number]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pycolmap
from sqlmodel import select

from backend.app.config import DATA_ROOT, SCENES_ROOT
from backend.app.db import session_scope
from backend.app.models import Scene, Version
from worker.pipeline.camera_paths import export_camera_paths


def main() -> None:
    slug = sys.argv[1]
    number = int(sys.argv[2]) if len(sys.argv) > 2 else None
    with session_scope() as session:
        scene = session.exec(select(Scene).where(Scene.slug == slug)).one()
        query = select(Version).where(Version.scene_id == scene.id).order_by(Version.number)
        if number is not None:
            query = query.where(Version.number == number)
        versions = session.exec(query).all()
    for v in versions:
        rec = pycolmap.Reconstruction(DATA_ROOT / v.reconstruction_path)
        # The frames live with the scene that made the reconstruction (as in worker.runner).
        image_root = SCENES_ROOT / Path(v.reconstruction_path).parts[1] / "images"
        out = DATA_ROOT / v.layers_url.removeprefix("/files/") / "camera_paths.json"
        export_camera_paths(rec, image_root, out)
        print(f"v{v.number}: {out.relative_to(DATA_ROOT)}")


if __name__ == "__main__":
    main()
