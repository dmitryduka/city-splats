"""Benchmarks with live progress, using the real pipeline code.

  python -m spikes.bench sfm <aliked|sift> [num_frames]        # first-video SfM on room video 1
  python -m spikes.bench brush <brush_binary> [steps] [every]  # splat training speed on room v1

Results are appended to data/bench/results.jsonl (one JSON line per run) so runs on the Mac
and the DGX can be compared.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOM = ROOT / "data/scenes/room"
RESULTS = ROOT / "data/bench/results.jsonl"


def _power() -> str:
    try:
        import subprocess

        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        return "ac" if "AC Power" in out else "battery"
    except OSError:
        return "unknown"


def _record(row: dict) -> None:
    row |= {"time": time.strftime("%Y-%m-%d %H:%M"), "host": platform.node(), "power": _power()}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def bench_sfm(features_name: str, num_frames: int | None) -> None:
    from worker.pipeline.features import feature_config
    from worker.pipeline.sfm import reconstruct_initial
    from worker.reporting import ConsoleReporter

    frames = json.loads((ROOM / "contributions/c1/frames.json").read_text())
    names = [f"c1/{f['name']}" for f in frames][:num_frames]
    out = ROOT / f"data/bench/sfm_{features_name}_{len(names)}"
    shutil.rmtree(out, ignore_errors=True)
    reporter = ConsoleReporter(f"[sfm {features_name}]")
    started = time.time()
    result = reconstruct_initial(ROOM / "images", names, out, feature_config(features_name), report=reporter)
    _record({
        "bench": "sfm",
        "features": features_name,
        "frames": len(names),
        "total_s": round(time.time() - started, 1),
        **{f"{k}_s": round(v, 1) for k, v in reporter.timings.items()},
        "registered": result.num_registered,
        "points": result.num_points,
        "reproj_px": round(result.mean_reproj_error, 3),
    })


def bench_brush(binary: str, steps: int, every: int) -> None:
    from worker.pipeline.splat import Snapshot
    from worker.pipeline.splat.brush import BrushTrainer
    from worker.reporting import format_duration

    work = ROOT / f"data/bench/brush_{Path(binary).parent.parent.name}_{steps}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    started = time.time()
    marks: list[tuple[int, float]] = []

    def on_snapshot(s: Snapshot) -> None:
        elapsed = time.time() - started
        marks.append((s.step, elapsed))
        rate = s.step / elapsed
        eta = (s.total_steps - s.step) / rate if rate else None
        sys.stdout.write(
            f"\r\033[K  [brush] step {s.step}/{s.total_steps}  {rate:.2f} steps/s  "
            f"elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}"
        )
        sys.stdout.flush()

    ply = BrushTrainer(Path(binary)).train(
        ROOM / "images", ROOM / "contributions/c1/sfm/sparse/0", work, steps, 1600, None, every, on_snapshot
    )
    total = time.time() - started
    print()
    # Steady-state rate excludes startup (dataset load, kernel autotune) before the first snapshot.
    steady = (marks[-1][0] - marks[0][0]) / (marks[-1][1] - marks[0][1]) if len(marks) > 1 else None
    _record({
        "bench": "brush",
        "binary": binary,
        "steps": steps,
        "total_s": round(total, 1),
        "startup_s": round(marks[0][1] - marks[0][0] / steady, 1) if steady else None,
        "steps_per_s": round(steady, 2) if steady else round(steps / total, 2),
        "splats": int(ply.read_bytes()[:400].split(b"element vertex ")[1].split(b"\n")[0]),
    })


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "sfm":
        bench_sfm(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    elif kind == "brush":
        bench_brush(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2000,
                    int(sys.argv[4]) if len(sys.argv) > 4 else 500)
    else:
        raise SystemExit(__doc__)
