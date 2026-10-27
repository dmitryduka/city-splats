"""Gaussian splat trainers. Brush (wgpu: Metal/Vulkan) today; gsplat (CUDA) for DGX in M5."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class Snapshot:
    step: int
    total_steps: int
    ply_path: Path


class TrainingStopped(Exception):
    """Training was interrupted because should_stop() returned True."""


class Trainer(Protocol):
    name: str

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
        eval_names: list[str] | None = None,
    ) -> Path:
        """Train and return the final PLY. Calls on_snapshot for intermediate exports and
        raises TrainingStopped if should_stop() turns True (polled every few seconds).
        freeze_scales keeps the init PLY's per-splat sizes and never optimizes sizes or
        rotations ("uniform" splats). eval_names: frames held out for evaluation."""
        ...
