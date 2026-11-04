"""Put terrain under a georeferenced version's map (worker.pipeline.terrain): writes terrain.bin
next to the version's files and geo["terrain"] in its metrics; --remove takes it out again
(the map goes back to the flat ground plane).

  python -m scripts.build_terrain <scene-slug> [version number] [--remove]
"""

from __future__ import annotations

import sys

import numpy as np
import pycolmap
from sqlmodel import select

from backend.app.config import DATA_ROOT, SCENES_ROOT, files_url
from backend.app.db import session_scope
from backend.app.models import Scene, Version
from worker.pipeline.terrain import build_terrain
from worker.pipeline.tiles import sample_cloud

SAMPLE_POINTS = 20_000_000  # of the dense cloud: on Yerevan v12 ~2900 a 30 m square with ground, 50 s


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    remove = "--remove" in sys.argv
    with session_scope() as session:
        scene = session.exec(select(Scene).where(Scene.slug == args[0])).one()
        number = int(args[1]) if len(args) > 1 else session.get(Version, scene.head_version_id).number
        v = session.exec(select(Version).where(Version.scene_id == scene.id, Version.number == number)).one()
        geo = v.metrics.get("geo")
        if not geo:
            sys.exit(f"v{number} has no georeference")
        out = SCENES_ROOT / scene.slug / "versions" / f"v{number}" / "terrain.bin"
        if remove:
            geo = {k: val for k, val in geo.items() if k != "terrain"}
            out.unlink(missing_ok=True)
        else:
            rec = pycolmap.Reconstruction(DATA_ROOT / v.reconstruction_path)
            cameras = np.array([image.projection_center() for image in rec.images.values() if image.has_pose])
            points, _ = sample_cloud(v.metrics["dense_tiles"], DATA_ROOT, SAMPLE_POINTS)
            terrain = build_terrain(points, cameras, geo, out, DATA_ROOT / "terrain" / "terrarium")
            geo = geo | {"terrain": terrain | {"url": files_url(out)}}
        v.metrics = v.metrics | {"geo": geo}
        session.add(v)
        session.commit()
        print(f"v{number}:", geo.get("terrain", "terrain removed"))


if __name__ == "__main__":
    main()
