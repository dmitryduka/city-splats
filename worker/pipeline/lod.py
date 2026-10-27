"""Spark build-lod: splat PLY -> chunked .rad LoD tree for paged streaming."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def build_lod(binary: Path, ply: Path, out_dir: Path, quality: bool = True) -> Path:
    """Returns the .rad header path; chunks (.radc) sit next to it."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    src = out_dir / "splat.ply"
    # build-lod writes next to its input; link instead of copying the (large) PLY.
    src.symlink_to(ply.resolve())
    cmd = [str(binary), str(src), "--rad-chunked"]
    if quality:
        cmd.append("--quality")
    subprocess.run(cmd, check=True, capture_output=True)
    src.unlink()
    rad = out_dir / "splat-lod.rad"
    if not rad.exists():
        raise RuntimeError(f"build-lod did not produce {rad}")
    return rad
