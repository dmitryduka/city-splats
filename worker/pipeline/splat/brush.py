"""Brush trainer (headless CLI). Progress is observed through its periodic PLY exports."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path

from . import Snapshot, TrainingStopped

_EXPORT_RE = re.compile(r"splat_(\d+)\.ply$")


@cache
def _steps_flag(binary: Path) -> str:
    """Brush <= 0.3 takes --total-steps; newer brush-cli renamed it --total-train-iters."""
    help_text = subprocess.run([str(binary), "--help"], capture_output=True, text=True).stdout
    return "--total-train-iters" if "--total-train-iters" in help_text else "--total-steps"


class BrushTrainer:
    name = "brush"

    def __init__(self, binary: Path):
        self.binary = binary

    def train(
        self,
        image_root: Path,
        reconstruction: Path,
        work_dir: Path,
        steps: int,
        max_resolution: int,
        init_ply: Path | None,
        snapshot_every: int,
        on_snapshot: Callable[[Snapshot], None],
        should_stop: Callable[[], bool] = lambda: False,
        sh_degree: int = 3,
        freeze_scales: bool = False,
    ) -> Path:
        # Brush wants a COLMAP dataset dir: images/, sparse/0/, and optionally a .ply that
        # it uses as initialization (warm start from the previous version's splat).
        dataset = work_dir / "dataset"
        if dataset.exists():
            shutil.rmtree(dataset)
        (dataset / "sparse").mkdir(parents=True)
        (dataset / "images").symlink_to(image_root.resolve())
        (dataset / "sparse" / "0").symlink_to(reconstruction.resolve())
        if init_ply is not None:
            shutil.copyfile(init_ply, dataset / "init.ply")

        export_dir = work_dir / "exports"
        shutil.rmtree(export_dir, ignore_errors=True)
        export_dir.mkdir(parents=True)
        # Brush writes its GPU autotune cache to the CWD; keep it out of the repo.
        cache_dir = work_dir.parent.parent / ".brush_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.binary.resolve()),
            str(dataset.resolve()),
            _steps_flag(self.binary.resolve()), str(steps),
            "--eval-split-every", "8",
            "--export-every", str(min(snapshot_every, steps)),
            "--export-path", str(export_dir.resolve()),
            "--export-name", "splat_{iter}.ply",
            "--max-resolution", str(max_resolution),
            "--sh-degree", str(sh_degree),
        ]
        if freeze_scales:
            cmd.append("--freeze-scales")
        log = open(work_dir / "brush.log", "w")
        proc = subprocess.Popen(cmd, cwd=cache_dir, stdout=log, stderr=subprocess.STDOUT)

        seen: set[int] = set()

        def poll_exports() -> None:
            for path in sorted(export_dir.glob("splat_*.ply")):
                match = _EXPORT_RE.search(path.name)
                if not match or int(match.group(1)) in seen:
                    continue
                # Wait until the writer is done (size stable) before handing it out.
                size = path.stat().st_size
                time.sleep(0.5)
                if path.stat().st_size != size:
                    continue
                step = int(match.group(1))
                seen.add(step)
                on_snapshot(Snapshot(step=step, total_steps=steps, ply_path=path))

        while proc.poll() is None:
            if should_stop():
                proc.terminate()
                proc.wait()
                log.close()
                raise TrainingStopped()
            poll_exports()
            time.sleep(2)
        log.close()
        poll_exports()
        if proc.returncode != 0:
            raise RuntimeError(f"brush exited with {proc.returncode}; see {work_dir / 'brush.log'}")

        final = export_dir / f"splat_{steps}.ply"
        if not final.exists():
            raise RuntimeError(f"brush finished but {final.name} was not exported")
        return final
