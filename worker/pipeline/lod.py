"""Spark build-lod: splat PLY -> chunked .rad LoD tree for paged streaming."""

from __future__ import annotations

import shutil
import subprocess
import threading
import json

from backend.app.config import LOD_BACKEND

# Upstream builds tile LoDs in a thread pool; serialize GPU builds to bound VRAM.
_CUDA_LOD_LOCK = threading.Lock()
from pathlib import Path


def build_lod(binary: Path, ply: Path, out_dir: Path, quality: bool = True) -> Path:
    """Returns the .rad header path; chunks (.radc) sit next to it."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if LOD_BACKEND == "cuda_voxel":
        import cupy as cp
        from .cuda_lod import build_lod as cuda_build_lod
        with _CUDA_LOD_LOCK:
            try:
                result = cuda_build_lod(ply, out_dir)
                metrics_path = out_dir / "cuda-lod.json"
                metrics = json.loads(metrics_path.read_text())
                metrics["baseline_algorithm"] = "spark_bhattacharyya" if quality else "spark_tiny_voxel"
                metrics_path.write_text(json.dumps(metrics, indent=2))
                return result
            finally:
                cp.get_default_memory_pool().free_all_blocks()
    if LOD_BACKEND != "spark":
        raise ValueError(f"Unknown LOD_BACKEND: {LOD_BACKEND}")
    src = out_dir / "splat.ply"
    # build-lod writes next to its input; link instead of copying the (large) PLY.
    src.symlink_to(ply.resolve())
    # build-lod's default is --quality (bhatt-lod: greedy pairwise merges on one core). For a
    # cloud of identical round splats --quick (tiny-lod, a voxel pyramid) loses little and was
    # 6x faster on 22.8M points (55 s vs ~330 s, 3.6 GB peak).
    cmd = [str(binary), str(src), "--rad-chunked", "--quality" if quality else "--quick"]
    subprocess.run(cmd, check=True, capture_output=True)
    src.unlink()
    rad = out_dir / "splat-lod.rad"
    if not rad.exists():
        raise RuntimeError(f"build-lod did not produce {rad}")
    return rad
