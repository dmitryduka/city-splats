"""One video's own SfM model (its piece of the scene), run as a subprocess so several videos
can be reconstructed side by side with a fixed share of the cores each, and so each has its
own COLMAP log (progress lines of concurrent runs can't be told apart within one process).

  python -m worker.pipeline.piece_job <image_root> <names_file> <out_dir> <features> <threads> [<gps.json> | -]

Prints JSON lines on stdout: {"fraction": f, "detail": "..."} while running, then
{"result": {...InitialResult fields...}}. COLMAP's log goes to stderr.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from worker.pipeline.features import feature_config
from worker.pipeline.sfm import reconstruct_initial
from worker.reporting import FractionReporter


def _emit(message: dict) -> None:
    sys.stdout.write(json.dumps(message, default=str) + "\n")
    sys.stdout.flush()


def _exit_with_parent() -> None:
    """Stop when the worker dies (no orphan keeps writing into a dir the restarted worker redoes)."""
    parent = os.getppid()
    while os.getppid() == parent:
        time.sleep(2)
    os._exit(1)


def main() -> None:
    threading.Thread(target=_exit_with_parent, daemon=True).start()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    image_root, names_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    features, threads = feature_config(sys.argv[4]), int(sys.argv[5])
    names = [n for n in names_file.read_text().split("\n") if n]
    gps = json.loads(Path(sys.argv[6]).read_text()) if len(sys.argv) > 6 and sys.argv[6] != "-" else None
    reporter = FractionReporter(lambda fraction, detail: _emit({"fraction": round(fraction, 4), "detail": detail}))
    result = reconstruct_initial(
        image_root, names, out_dir, features, report=reporter, num_threads=threads,
        positions={name: np.array(xyz) for name, xyz in gps["positions"].items()} if gps else None,
        position_sigma=tuple(gps["sigma"]) if gps else (3.0, 3.0, 0.3),
    )
    _emit({"result": result.__dict__})


if __name__ == "__main__":
    main()
