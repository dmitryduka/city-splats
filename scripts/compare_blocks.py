"""Held-out PSNR per training block: the version's block-trained splats against its globally
trained splat, on the same held-out frames (neither trained on them). The global splat is
measured by a 1-step Brush run warm-started from it on each block's dataset.

  python -m scripts.compare_blocks <version-id> [quality]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from backend.app.config import DATA_ROOT, SCENES_ROOT, TRAIN_SETTINGS
from backend.app.db import session_scope
from backend.app.models import Scene, Version
from worker import compute
from worker.pipeline.splat.brush import eval_results


def main() -> None:
    version_id, quality = int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "full"
    with session_scope() as session:
        v = session.get(Version, version_id)
        scene = session.get(Scene, v.scene_id)
    blocks = v.metrics[f"splat_{quality}"]["blocks"]
    image_root = SCENES_ROOT / Path(v.reconstruction_path).parts[1] / "images"
    global_ply = SCENES_ROOT / scene.slug / "versions" / f"v{v.number}" / quality / "splat.ply"
    out = SCENES_ROOT / scene.slug / "versions" / f"v{v.number}" / f"{quality}_compare"
    _, max_resolution = TRAIN_SETTINGS[quality]
    rows = []
    for key, block in blocks.items():
        block_dir = DATA_ROOT / block["dir"]
        eval_names = (block_dir / "train" / "eval.txt").read_text().split("\n")
        work = out / key
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        compute.trainer().train(image_root, block_dir / "model", work, 1, max_resolution, global_ply, 1,
                                lambda s: None, eval_names=eval_names)
        shutil.rmtree(work / "exports", ignore_errors=True)
        shutil.rmtree(work / "dataset", ignore_errors=True)
        measured = eval_results(work)
        rows.append({"block": key, "frames": block["frames"], "eval_frames": len(eval_names),
                     "global": measured[-1] if measured else None, "blocks": block["eval"]})
        print(json.dumps(rows[-1]), flush=True)
    (out / "comparison.json").write_text(json.dumps(rows, indent=1))
    weight = sum(r["eval_frames"] for r in rows if r["global"] and r["blocks"])
    for name in ("global", "blocks"):
        psnr = sum(r[name]["psnr"] * r["eval_frames"] for r in rows if r["global"] and r["blocks"]) / max(weight, 1)
        print(f"{name}: {psnr:.2f} dB (held-out frames weighted, {weight} frames)")


if __name__ == "__main__":
    main()
