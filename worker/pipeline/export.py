"""Export reconstruction layers for the viewer (sparse points + camera centers as PLY)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pycolmap

# Per-contribution palette (sRGB), cycled; shared points use SHARED_COLOR.
CONTRIBUTION_COLORS = np.array(
    [[70, 140, 255], [255, 150, 40], [200, 80, 220], [60, 200, 200], [230, 70, 90]], dtype=np.uint8
)
SHARED_COLOR = np.array([80, 220, 100], dtype=np.uint8)


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    data = np.empty(len(xyz), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    data["xyz"] = xyz
    data["rgb"] = rgb
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def export_layers(
    rec: pycolmap.Reconstruction,
    out_dir: Path,
    contribution_of: Callable[[str], int],
) -> dict[str, Path]:
    """Write points_rgb.ply, points_source.ply (colored by contribution) and cameras.ply."""
    out_dir.mkdir(parents=True, exist_ok=True)
    source = {image_id: contribution_of(img.name) for image_id, img in rec.images.items()}

    xyz, rgb, src_rgb = [], [], []
    for point in rec.points3D.values():
        xyz.append(point.xyz)
        rgb.append(point.color)
        sources = {source[el.image_id] for el in point.track.elements}
        if len(sources) > 1:
            src_rgb.append(SHARED_COLOR)
        else:
            src_rgb.append(CONTRIBUTION_COLORS[sources.pop() % len(CONTRIBUTION_COLORS)])
    xyz_a = np.asarray(xyz, dtype=np.float32)

    centers, cam_rgb = [], []
    for image_id, img in rec.images.items():
        if img.has_pose:
            centers.append(img.projection_center())
            cam_rgb.append(CONTRIBUTION_COLORS[source[image_id] % len(CONTRIBUTION_COLORS)])

    paths = {
        "points_rgb": out_dir / "points_rgb.ply",
        "points_source": out_dir / "points_source.ply",
        "cameras": out_dir / "cameras.ply",
    }
    _write_ply(paths["points_rgb"], xyz_a, np.asarray(rgb, dtype=np.uint8))
    _write_ply(paths["points_source"], xyz_a, np.asarray(src_rgb, dtype=np.uint8))
    _write_ply(paths["cameras"], np.asarray(centers, dtype=np.float32), np.asarray(cam_rgb, dtype=np.uint8))
    return paths
