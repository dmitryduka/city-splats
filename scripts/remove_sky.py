"""Sky splats out of a version's trained blocks (worker.pipeline.sky), each block's LoD rebuilt
in place, for blocks trained before the worker did this itself. With --watch it keeps going
while the version trains, taking each block as it finishes (masks on the CPU, leaving the GPU
to training). The input splats stay as splat_with_sky.ply.

  python -m scripts.remove_sky <version-id> [quality] [--watch]
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pycolmap

from backend.app.config import BUILD_LOD_BIN, DATA_ROOT, SCENES_ROOT, TRAIN_SKY_EVERY
from backend.app.db import session_scope
from backend.app.models import Scene, Version
from worker.pipeline.lod import build_lod
from worker.pipeline.sky import remove_sky

POLL_SECONDS = 300


def _finished(version_id: int, quality: str) -> tuple[list[str], Path, Path, bool]:
    """Keys of the blocks trained so far, the blocks dir, the training images and whether
    training is still running."""
    with session_scope() as session:
        v = session.get(Version, version_id)
        scene = session.get(Scene, v.scene_id)
    running = v.train_status in ("queued", "training") and v.train_quality == quality
    blocks = (v.metrics.get("training_report") or {}).get("blocks") if running else \
        (v.metrics.get(f"splat_{quality}") or {}).get("blocks")
    train_images = SCENES_ROOT / scene.slug / "train_images"
    image_root = train_images if train_images.exists() else SCENES_ROOT / Path(v.reconstruction_path).parts[1] / "images"
    return list(blocks or {}), SCENES_ROOT / scene.slug / "versions" / f"v{v.number}" / f"{quality}_blocks", image_root, running


def _record(version_id: int, key: str, stats: dict) -> None:
    with session_scope() as session:
        v = session.get(Version, version_id)
        v.metrics = v.metrics | {"sky_removed": (v.metrics.get("sky_removed") or {}) | {key: stats}}
        session.add(v)
        session.commit()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    version_id, quality = int(args[0]), args[1] if len(args) > 1 else "full"
    watch = "--watch" in sys.argv
    every = TRAIN_SKY_EVERY or 4
    while True:
        keys, blocks_dir, image_root, running = _finished(version_id, quality)
        with session_scope() as session:
            v = session.get(Version, version_id)
            slug, geo = session.get(Scene, v.scene_id).slug, v.metrics["geo"]
        for key in keys:
            block_dir = blocks_dir / key
            marker = block_dir / "sky.json"
            if marker.exists() or not (block_dir / "lod").exists():
                continue
            started = time.time()
            rec = pycolmap.Reconstruction(block_dir / "model")
            # CPU while a block trains on the GPU; the masks are cached per scene either way.
            stats = remove_sky(block_dir, rec, geo, image_root, SCENES_ROOT / slug / "sky_masks", every,
                               device="cpu" if running else None, keep_original=True)
            # Build next to the served LoD and swap, so viewers only miss it for a moment.
            build_lod(BUILD_LOD_BIN, block_dir / "splat.ply", block_dir / "lod_new", quality=True)
            old = block_dir / "lod_old"
            shutil.rmtree(old, ignore_errors=True)
            (block_dir / "lod").rename(old)
            (block_dir / "lod_new").rename(block_dir / "lod")
            shutil.rmtree(old)
            stats["seconds"] = round(time.time() - started, 1)
            marker.write_text(json.dumps(stats))
            _record(version_id, key, stats)
            print(f"{key}: {stats}", flush=True)
        if not (watch and running):
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
