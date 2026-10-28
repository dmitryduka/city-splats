"""Removing the splats that paint the sky.

Splat training has to explain every pixel, so the sky and the hazy horizon get painted too:
on Achapnyak as a curved backdrop of large splats far beyond the flown area, and as faint haze
between the roofs and flight height. Two rules, measured on Achapnyak Ultra's first block
(5.5M splats, renders of its 142 held-out frames compared by eye):
- sky vote: a splat whose centre lands on sky in most of the views that see it is in front of
  nothing but sky (a real surface would hide the sky behind it). Sky comes from an image
  segmentation network (worker.pipeline.sky_net). Alone it removed 1.6% of the splats (35% of
  their opacity x area) but left dark blobs of cloud: large sky splats whose centres also land
  on buildings in some views.
- roof cap: splats more than CAP_MARGIN_M above the highest SfM points within ~30 m (anywhere
  without SfM points: above the block's highest ones). With the vote: 3.9% of the splats, 37%
  of the opacity x area, and the sky renders empty. It also trims the far skyline beyond the
  reconstructed area (blurry there anyway); capping only where there are SfM points kept
  that, but left sky blobs over it.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image
from scipy.ndimage import maximum_filter

from worker.pipeline.cleanup import clean_sparse, read_ply, write_ply


def predict_sky(image_root: Path, names: list[str], out_dir: Path, device: str | None = None,
                on_progress: Callable[[int, int], None] | None = None) -> None:
    """Sky masks for the frames, in a subprocess (see sky_net); cached in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names_file = out_dir / f"names_{abs(hash(tuple(names)))}.txt"
    names_file.write_text("\n".join(names))
    proc = subprocess.Popen(
        [sys.executable, "-m", "worker.pipeline.sky_net", str(image_root), str(names_file), str(out_dir),
         *([device] if device else [])],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        tail = (tail + [line])[-20:]
        if line.startswith("sky ") and on_progress:
            done, total = line.split()[1].split("/")
            on_progress(int(done), int(total))
    names_file.unlink(missing_ok=True)
    if proc.wait() != 0:
        raise RuntimeError("sky segmentation failed:\n" + "".join(tail))


CAP_MARGIN_M = 10.0


@dataclass
class SkyStats:
    total: int
    views: int
    sky: int  # on sky in at least `threshold` of the views that see them
    above: int  # above the roofs (and not already on sky)

    def as_dict(self) -> dict:
        return {"total": self.total, "views": self.views, "sky": self.sky, "above_roofs": self.above,
                "kept": self.total - self.sky - self.above}


def sky_views(rec: pycolmap.Reconstruction, every: int) -> list[str]:
    """Every `every`th view by name: sky votes need many views per splat, not every view."""
    return sorted(image.name for image in rec.images.values() if image.has_pose)[::every]


def _above_roofs(xyz: np.ndarray, rec: pycolmap.Reconstruction, geo: dict, margin_m: float,
                 cell_m: float = 20.0, min_points: int = 5) -> np.ndarray:
    """Mask of positions more than margin_m above the roof heightmap: per cell_m cell the 99th
    percentile height of the SfM points (cells with at least min_points), the maximum over
    the 3x3 cells around; elsewhere the points' 99.9th percentile height."""
    enu_from_scene = np.linalg.inv(np.array(geo["scene_from_enu"]))
    to_map = lambda p: p @ enu_from_scene[:3, :3].T + enu_from_scene[:3, 3]
    points = to_map(np.array([p.xyz for p in rec.points3D.values()]))
    splats = to_map(xyz)
    lo = np.minimum(points[:, :2].min(0), splats[:, :2].min(0))
    cell = lambda p: np.floor((p[:, :2] - lo) / cell_m).astype(np.int64)
    pc, sc = cell(points), cell(splats)
    shape = tuple(np.maximum(pc.max(0), sc.max(0)) + 1)
    flat = np.ravel_multi_index(tuple(pc.T), shape)
    order = np.argsort(flat, kind="stable")
    flat, heights = flat[order], points[order, 2]
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    top = np.full(shape, -np.inf)
    for s0, s1 in zip(starts, np.r_[starts[1:], len(flat)]):
        if s1 - s0 >= min_points:
            top.flat[flat[s0]] = np.percentile(heights[s0:s1], 99)
    top = maximum_filter(top, size=3, mode="constant", cval=-np.inf)
    roof = top[sc[:, 0], sc[:, 1]]
    roof = np.where(np.isfinite(roof), roof, np.percentile(points[:, 2], 99.9))
    return splats[:, 2] > roof + margin_m


def clean_sky(splat_ply: Path, rec: pycolmap.Reconstruction, geo: dict, names: list[str], mask_dir: Path,
              out_ply: Path, threshold: float = 0.5, min_views: int = 2) -> SkyStats:
    """Write splat_ply minus the splats whose centre lands on sky in at least `threshold` of
    the views (named, with masks in mask_dir) that see it, if at least min_views do, and the
    ones above the roofs (map frame of geo, worker.pipeline.georef)."""
    header, props, rows = read_ply(splat_ply)
    xyz = rows[:, [props.index("x"), props.index("y"), props.index("z")]].astype(np.float64)
    seen = np.zeros(len(rows), np.uint16)
    sky = np.zeros(len(rows), np.uint16)
    by_name = {image.name: image for image in rec.images.values()}
    for name in names:
        mask = np.asarray(Image.open(mask_dir / f"{name}.png")) > 127
        image = by_name[name]
        camera = rec.cameras[image.camera_id]
        pose = image.cam_from_world()
        cam = xyz @ pose.rotation.matrix().T + pose.translation
        front = np.flatnonzero(cam[:, 2] > 0)
        uv = camera.img_from_cam(cam[front])
        # Points the distortion model can't project come back as NaN (or huge, near the
        # image plane); either is outside the image.
        uv = np.clip(np.nan_to_num(uv, nan=-1.0), -1.0, 1e6)
        u = np.floor(uv[:, 0] * mask.shape[1] / camera.width).astype(np.int64)
        v = np.floor(uv[:, 1] * mask.shape[0] / camera.height).astype(np.int64)
        inside = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        idx = front[inside]
        seen[idx] += 1
        sky[idx] += mask[v[inside], u[inside]]
    voted = (seen >= min_views) & (sky >= threshold * seen)
    above = _above_roofs(xyz, rec, geo, CAP_MARGIN_M)
    write_ply(out_ply, header, rows[~(voted | above)])
    return SkyStats(len(rows), len(names), int(voted.sum()), int((above & ~voted).sum()))


def remove_sky(block_dir: Path, rec: pycolmap.Reconstruction, geo: dict, image_root: Path, mask_dir: Path,
               every: int, device: str | None = None, keep_original: bool = False) -> dict:
    """Sky splats out of a trained block's splat.ply (in place; with keep_original the
    input stays as splat_with_sky.ply), voting over every `every`th of its views. rec (the
    block's model) loses its unreliable SfM points (clean_sparse)."""
    clean_sparse(rec)  # stray high points would lift the roofs
    names = sky_views(rec, every)
    predict_sky(image_root, names, mask_dir, device)
    splat = block_dir / "splat.ply"
    out = block_dir / "splat_no_sky.ply"
    stats = clean_sky(splat, rec, geo, names, mask_dir, out)
    if keep_original:
        splat.rename(block_dir / "splat_with_sky.ply")
    out.replace(splat)
    return stats.as_dict()
