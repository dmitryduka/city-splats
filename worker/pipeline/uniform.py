"""Starting cloud for "uniform" splats: round, one color each, with sizes fixed in training.

One world size fails on an outdoor walk: the waterfall's dense points lie 14-150 units from the
nearest camera, so a size that fits near rocks leaves far trees as dots, and nothing covers the
sky. Instead each splat gets a fixed size on screen (`pixels` std-dev at the training
resolution, seen from its nearest camera), and a sky shell is placed along the cameras' rays
far beyond the scene, colored from the frames and sized to tile the sky without gaps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image
from scipy.spatial import cKDTree



def _shell(rec: pycolmap.Reconstruction, image_root: Path, radius: float, train_focal: float,
           every: int, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points `radius` from the scene centre along a grid of rays of every `every`-th frame."""
    images = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)[::every]
    camera = rec.cameras[images[0].camera_id]
    cols, rows = grid
    u, v = np.meshgrid((np.arange(cols) + 0.5) * camera.width / cols, (np.arange(rows) + 0.5) * camera.height / rows)
    pixels = np.stack([u.ravel(), v.ravel()], axis=1)
    rays_cam = np.hstack([camera.cam_from_img(pixels), np.ones((len(pixels), 1))])
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    centre = np.median([im.projection_center() for im in images], axis=0)
    xyz, rgb = [], []
    for image in images:
        world_from_cam = image.cam_from_world().inverse().matrix()
        origin = world_from_cam[:, 3]
        dirs = rays_cam @ world_from_cam[:, :3].T
        # Distance along each ray to the sphere of `radius` around the scene centre.
        oc = origin - centre
        b = dirs @ oc
        t = -b + np.sqrt(np.maximum(b * b - (oc @ oc - radius**2), 0))
        xyz.append(origin + dirs * t[:, None])
        frame = np.asarray(Image.open(image_root / image.name).convert("RGB"))
        px = np.clip(pixels.astype(int), 0, [frame.shape[1] - 1, frame.shape[0] - 1])
        rgb.append(frame[px[:, 1], px[:, 0]])
    # Std-dev ~0.6 of the grid spacing seen from the cameras covers the sky without gaps.
    spacing_px = camera.width / cols * train_focal / camera.mean_focal_length()
    scale = 0.6 * spacing_px * radius / train_focal
    xyz = np.concatenate(xyz)
    return xyz, np.concatenate(rgb), np.full(len(xyz), scale)


def uniform_init(cloud: tuple[np.ndarray, np.ndarray], rec: pycolmap.Reconstruction, image_root: Path,
                 out_ply: Path, max_resolution: int, pixels: float = 2.0, num_points: int = 500_000,
                 shell_every: int = 4, shell_grid: tuple[int, int] = (27, 48)) -> dict:
    """Write the starting PLY (xyz, rgb, log scales) from (a sample of) the dense cloud and
    return stats."""
    xyz, rgb = cloud
    keep = np.random.default_rng(0).permutation(len(xyz))[:num_points]
    xyz, rgb = xyz[keep], rgb[keep]
    camera = next(iter(rec.cameras.values()))
    train_focal = camera.mean_focal_length() * min(1.0, max_resolution / max(camera.width, camera.height))
    centres = np.array([im.projection_center() for im in rec.images.values() if im.has_pose])
    distance, _ = cKDTree(centres).query(xyz)
    scale = pixels * distance / train_focal

    # Shell twice as far out as the farthest 1% of the scene.
    radius = 2 * float(np.percentile(np.linalg.norm(xyz - np.median(centres, axis=0), axis=1), 99))
    shell_xyz, shell_rgb, shell_scale = _shell(rec, image_root, radius, train_focal, shell_every, shell_grid)

    xyz = np.concatenate([xyz, shell_xyz]).astype(np.float32)
    rgb = np.concatenate([rgb, shell_rgb]).astype(np.uint8)
    log_scale = np.log(np.concatenate([scale, shell_scale])).astype(np.float32)
    rows = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                                     ("blue", "u1"), ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4")])
    rows["x"], rows["y"], rows["z"] = xyz.T
    rows["red"], rows["green"], rows["blue"] = rgb.T
    rows["scale_0"] = rows["scale_1"] = rows["scale_2"] = log_scale
    header = "".join([
        "ply\nformat binary_little_endian 1.0\n",
        f"element vertex {len(rows)}\n",
        *(f"property {'float' if rows.dtype[n] == np.float32 else 'uchar'} {n}\n" for n in rows.dtype.names),
        "end_header\n",
    ])
    with open(out_ply, "wb") as f:
        f.write(header.encode())
        f.write(rows.tobytes())
    return {
        "points": len(keep), "shell_points": len(shell_xyz), "shell_radius": round(radius, 2),
        "size_px": pixels, "size_median": round(float(np.median(scale)), 5),
    }
