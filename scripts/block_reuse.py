"""Dry run of block reuse: which of a version's splat blocks training would take from its
trained ancestor and which it would train, and why (worker.runner._plan_blocks). Reads the
data directory only: the database is copied to a temporary directory first.

  python -m scripts.block_reuse <version-id> [quality]
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> None:
    version_id, quality = int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "full"
    from backend.app.config import DATA_ROOT  # noqa: PLC0415 - only to find the database

    copy = Path(tempfile.mkdtemp())
    for suffix in ("", "-wal", "-shm"):
        if (DATA_ROOT / f"scenes.db{suffix}").exists():
            shutil.copy(DATA_ROOT / f"scenes.db{suffix}", copy / f"scenes.db{suffix}")
    os.environ["SCENES_DATABASE_URL"] = f"sqlite:///{copy / 'scenes.db'}"
    import pycolmap  # noqa: PLC0415

    from backend.app.config import TRAIN_BLOCK_FRAMES, TRAIN_SETTINGS, scene_setting  # noqa: PLC0415
    from backend.app.db import session_scope  # noqa: PLC0415
    from backend.app.models import Scene, Version  # noqa: PLC0415
    from worker.pipeline.cleanup import clean_sparse  # noqa: PLC0415
    from worker.pipeline.splat.blocks import plan_blocks  # noqa: PLC0415
    from worker.runner import _block_settings, _plan_blocks  # noqa: PLC0415

    with session_scope() as session:
        v = session.get(Version, version_id)
        scene = session.get(Scene, v.scene_id)
    steps, max_resolution = TRAIN_SETTINGS[quality]
    steps = scene_setting(scene.settings, f"TRAIN_{quality.upper()}_STEPS", steps)
    max_resolution = scene_setting(scene.settings, f"TRAIN_{quality.upper()}_RES", max_resolution)
    rec = pycolmap.Reconstruction(DATA_ROOT / v.reconstruction_path)
    clean_sparse(rec)
    bp = _plan_blocks(v, quality, steps, max_resolution, _block_settings(scene), rec)
    if bp.ancestor is None:
        print(f"v{v.number}: no earlier version trained in blocks at {quality}; all {len(bp.blocks)} blocks train")
        return
    before = pycolmap.Reconstruction(DATA_ROOT / bp.ancestor.reconstruction_path)
    added = ({im.name for im in rec.images.values() if im.has_pose}
             - {im.name for im in before.images.values() if im.has_pose})
    old = bp.ancestor.metrics[f"splat_{quality}"]["blocks"]
    print(f"v{v.number} against v{bp.ancestor.number}: {len(added)} new frames; "
          f"{len(bp.reused)} of {len(bp.blocks)} blocks reused, {len(bp.stale)} to train")
    print(f"{'block':<12} {'frames':>6} {'was':>5} {'margin':>6} {'new own':>7} {'new margin':>10}  decision")
    for b in bp.blocks:
        own = len(added & set(b.core))
        margin = len(added & (set(b.images) - set(b.core)))
        was = old.get(b.key, {}).get("frames", "-")
        decision = "reuse" if b.key in bp.reused else f"train ({bp.stale[b.key]})"
        print(f"{b.key:<12} {len(b.core):>6} {was:>5} {len(b.images) - len(b.core):>6} {own:>7} {margin:>10}  {decision}")

    # What planning every version from scratch (before keep) gives, in the version's own map frame.
    fresh = plan_blocks(rec, v.metrics["geo"], TRAIN_BLOCK_FRAMES)
    print("planned from scratch:", ", ".join(f"{b.key} ({len(b.core)})" for b in fresh),
          f"- {sum(b.key not in old for b in fresh)} keys not in v{bp.ancestor.number}")
    shutil.rmtree(copy)


if __name__ == "__main__":
    main()
