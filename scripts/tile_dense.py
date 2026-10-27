"""Give a scene's versions tiled dense clouds (worker.pipeline.tiles), in version order, from
each contribution's own cloud (dense/contribution.ply, already in the scene frame); for
versions made before the cloud was tiled. Depth is not predicted again.

  python -m scripts.tile_dense <scene-slug>
"""

from __future__ import annotations

import logging
import sys
import time

from sqlmodel import select

from backend.app.config import BUILD_LOD_BIN, DATA_ROOT, DENSE_TILE_POINTS, LOD_JOBS, SCENES_ROOT, files_url
from backend.app.db import session_scope
from backend.app.models import Contribution, Scene, Version, utcnow
from worker.pipeline.dense import read_points_ply
from worker.pipeline.tiles import Spill, update_tiles
from worker.runner import _tiles_built, write_tileset


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    slug = sys.argv[1]
    with session_scope() as session:
        scene = session.exec(select(Scene).where(Scene.slug == slug)).one()
        versions = session.exec(select(Version).where(Version.scene_id == scene.id).order_by(Version.number)).all()
    manifests: dict[int, dict] = {}
    for version in versions:
        with session_scope() as session:
            contribution = session.get(Contribution, version.contribution_id)
        dense_dir = SCENES_ROOT / slug / "contributions" / contribution.image_dir_name / "dense"
        parent = manifests.get(version.parent_id)
        voxel = parent["voxel"] if parent else version.metrics.get("dense_voxel")
        if voxel is None:  # the first version's own voxel (it has no dense_voxel only if it failed)
            voxel = version.metrics["dense"]["voxel"]
        started = time.time()
        spill = Spill(dense_dir / "spill", voxel)
        xyz, rgb = read_points_ply(dense_dir / "contribution.ply")
        for k in range(0, len(xyz), 5_000_000):
            spill.add(xyz[k:k + 5_000_000], rgb[k:k + 5_000_000])
        del xyz, rgb
        manifest = update_tiles(parent, spill, dense_dir / "tiles", DATA_ROOT, BUILD_LOD_BIN, DENSE_TILE_POINTS,
                                LOD_JOBS, on_progress=lambda d, t: print(f"\r  tile LoDs {d}/{t}\033[K", end="", flush=True))
        for leftover in (dense_dir / "spill").glob("*"):
            leftover.unlink()
        (dense_dir / "spill").rmdir()
        manifests[version.id] = manifest
        total = sum(t["points"] for t in manifest["tiles"].values())
        before = sum(t["points"] for t in parent["tiles"].values()) if parent else 0
        built = _tiles_built(parent, manifest)
        tileset = write_tileset(manifest, dense_dir / "tiles.json")
        dense = dict(version.metrics.get("dense", {})) | {
            "points": total - before, "scene_points": total, "tiles": len(manifest["tiles"]), "tiles_built": built}
        with session_scope() as session:
            for row in (session.get(Version, version.id), session.get(Contribution, version.contribution_id)):
                metrics = {k: v for k, v in row.metrics.items() if k not in ("dense_cloud", "dense_failed")}
                row.metrics = metrics | {"dense": dense, "dense_tiles": manifest, "dense_voxel": voxel}
                row.updated_at = utcnow()
                session.add(row)
            v = session.get(Version, version.id)
            v.dense_url = files_url(tileset)
            session.commit()
        print(f"\rv{version.number}: {total:,} points in {len(manifest['tiles'])} tiles "
              f"({built} built) in {time.time() - started:.0f}s\033[K")


if __name__ == "__main__":
    main()
