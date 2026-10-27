"""Sparse-point and splat cleanup, plus an instant "points as splats" model.

Floaters are splats in space the cameras demonstrably saw through. Every SfM observation is
a ray from a camera centre to a 3D point, and everything strictly between them is empty
(otherwise the point would have been occluded). Rays are carved into a voxel grid; splats in
voxels crossed by many rays and away from every SfM point are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814


def clean_sparse(rec: pycolmap.Reconstruction, max_error: float = 2.0, min_track: int = 3,
                 outlier_std: float = 2.5) -> int:
    """Drop unreliable SfM points in place: high reprojection error, short tracks, and
    statistical outliers (mean distance to 8 neighbours far above the scene's typical one).
    Returns the number of points removed."""
    bad = [pid for pid, p in rec.points3D.items() if p.error > max_error or p.track.length() < min_track]
    for pid in bad:
        rec.delete_point3D(pid)
    ids = np.array(list(rec.points3D.keys()))
    xyz = np.array([rec.points3D[i].xyz for i in ids])
    if len(xyz) > 16:
        dist, _ = cKDTree(xyz).query(xyz, k=9)
        mean = dist[:, 1:].mean(1)
        outliers = ids[mean > mean.mean() + outlier_std * mean.std()]
        for pid in outliers:
            rec.delete_point3D(int(pid))
        return len(bad) + len(outliers)
    return len(bad)


def read_ply(path: Path) -> tuple[str, list[str], np.ndarray]:
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode()
    count = int(header.split("element vertex ")[1].split()[0])
    props = [line.split()[-1] for line in header.splitlines() if line.startswith("property")]
    if any(line.split()[1] != "float" for line in header.splitlines() if line.startswith("property")):
        raise ValueError(f"{path}: only float vertex properties are supported")
    rows = np.frombuffer(data, dtype="<f4", count=count * len(props), offset=end).reshape(count, len(props))
    return header, props, rows


def write_ply(path: Path, header: str, rows: np.ndarray) -> None:
    lines = header.splitlines(keepends=True)
    lines = [f"element vertex {len(rows)}\n" if line.startswith("element vertex") else line for line in lines]
    with open(path, "wb") as f:
        f.write("".join(lines).encode())
        f.write(np.ascontiguousarray(rows, dtype="<f4").tobytes())


@dataclass
class CleanupStats:
    total: int
    transparent: int  # nearly invisible (opacity below min_opacity)
    carved: int  # in space the cameras saw through

    @property
    def kept(self) -> int:
        return self.total - self.transparent - self.carved

    def as_dict(self) -> dict:
        return {"total": self.total, "transparent": self.transparent, "carved": self.carved, "kept": self.kept}


def clean_splats(splat_ply: Path, rec: pycolmap.Reconstruction, out_ply: Path,
                 min_opacity: float = 0.02) -> CleanupStats:
    """Write splat_ply minus nearly invisible and mid-air splats to out_ply. rec should be
    cleaned first (clean_sparse).

    Measured on the 3-video room (1.5M splats, holdout PSNR 22.05 dB at 1024 px):
    - opacity < 0.02: 3.4% of splats, -0.04 dB (and less to stream and sort);
    - carving (10 rays, 5 voxels clearance): 1.9%, -0.34 dB; looser (5 rays, 3 voxels)
      removes 2.9% but costs 0.8 dB, as it starts eating faint surface detail.
    A neighbour-density filter for the "haze" around the room was rejected: those sparse
    splats are large, faint background splats painting walls and ceiling (-1.8 dB).
    """
    header, props, rows = read_ply(splat_ply)
    xyz = rows[:, [props.index("x"), props.index("y"), props.index("z")]]
    opacity = 1 / (1 + np.exp(-rows[:, props.index("opacity")]))
    transparent = opacity < min_opacity
    carved = ~transparent & _carved(xyz, rec)
    write_ply(out_ply, header, rows[~(transparent | carved)])
    return CleanupStats(len(rows), int(transparent.sum()), int(carved.sum()))


def _observation_rays(rec: pycolmap.Reconstruction) -> tuple[np.ndarray, np.ndarray]:
    centers = {iid: im.projection_center() for iid, im in rec.images.items() if im.has_pose}
    starts, ends = [], []
    for point in rec.points3D.values():
        for el in point.track.elements:
            center = centers.get(el.image_id)
            if center is not None:
                starts.append(center)
                ends.append(point.xyz)
    return np.asarray(starts), np.asarray(ends)


def _carved(xyz: np.ndarray, rec: pycolmap.Reconstruction, grid: int = 256, min_rays: int = 10,
            end_margin: float = 2.5, clearance: float = 5.0) -> np.ndarray:
    """Mask of positions in voxels crossed by >= min_rays observation rays and at least
    clearance voxels from any SfM point (so textureless surfaces between sparse points
    survive). end_margin voxels in front of each point are not carved."""
    starts, ends = _observation_rays(rec)
    points = np.array([p.xyz for p in rec.points3D.values()])
    # Grid over the observed volume (robust bounds: ignore the extreme 0.5%).
    bounds = np.vstack([starts, points])
    lo, hi = np.percentile(bounds, 0.5, axis=0), np.percentile(bounds, 99.5, axis=0)
    voxel = float((hi - lo).max()) / grid
    lo -= 2 * voxel
    dims = np.ceil((hi + 2 * voxel - lo) / voxel).astype(int)

    def index(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ijk = np.floor((p - lo) / voxel).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < dims), axis=1)
        flat = np.ravel_multi_index(tuple(np.clip(ijk, 0, dims - 1).T), dims)
        return flat, inside

    free = np.zeros(int(np.prod(dims)), dtype=np.uint16)
    vec = ends - starts
    length = np.linalg.norm(vec, axis=1)
    # Sample each ray every half voxel, stopping end_margin voxels short of the point.
    for chunk in range(0, len(starts), 20000):
        s, v, n = starts[chunk:chunk + 20000], vec[chunk:chunk + 20000], length[chunk:chunk + 20000]
        usable = np.maximum(n - end_margin * voxel, 0)
        samples = np.ceil(usable / (0.5 * voxel)).astype(int)
        ray = np.repeat(np.arange(len(s)), samples)
        t = (np.arange(samples.sum()) - np.repeat(np.cumsum(samples) - samples, samples)) * 0.5 * voxel
        p = s[ray] + v[ray] * (t / n[ray])[:, None]
        flat, inside = index(p)
        # Count each voxel once per ray.
        key = np.unique(ray[inside].astype(np.int64) * free.size + flat[inside])
        np.add.at(free, key % free.size, 1)

    occupied = np.zeros(free.size, dtype=bool)
    flat, inside = index(points)
    occupied[flat[inside]] = True
    near_surface = distance_transform_edt(~occupied.reshape(dims)).reshape(-1) < clearance

    flat, inside = index(xyz)
    return inside & (free[flat] >= min_rays) & ~near_surface[flat]


_POINT_PROPS = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity",
                "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"]


def _point_splats(rec: pycolmap.Reconstruction, opacity: float = 0.9,
                  size_factor: float = 0.6) -> np.ndarray:
    """One round splat per SfM point (rows in _POINT_PROPS order), sized to its local point
    spacing and coloured by the point colour."""
    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0
    dist, _ = cKDTree(xyz).query(xyz, k=4)
    spacing = dist[:, 1:].mean(1)
    spacing = np.clip(spacing, np.percentile(spacing, 5), np.percentile(spacing, 95))
    scale = np.log(size_factor * spacing).astype(np.float32)
    n = len(xyz)
    return np.concatenate([
        xyz,
        np.repeat(scale[:, None], 3, axis=1),
        np.full((n, 1), np.log(opacity / (1 - opacity)), dtype=np.float32),
        np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (n, 1)),
        (rgb - 0.5) / SH_C0,
    ], axis=1)


def points_as_splats(rec: pycolmap.Reconstruction, out_ply: Path) -> int:
    """An untrained splat model from the sparse points; builds in about a second."""
    rows = _point_splats(rec)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(rows)}\n"
        + "".join(f"property float {p}\n" for p in _POINT_PROPS)
        + "end_header\n"
    )
    write_ply(out_ply, header, rows)
    return len(rows)
