"""Where a scene lies on the map: a similarity from local east/north/up metres (around a GPS
origin) to the scene frame, fitted to the camera centres of every frame with a GPS position
(DJI telemetry, worker.pipeline.devices.dji). The viewer uses it to lay map tiles under the
cloud.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pycolmap

from worker.pipeline.devices.dji import geo_offset, similarity


def _ground_plane(points: np.ndarray, cell: float = 15.0, min_points: int = 50) -> np.ndarray:
    """(a, b, c) with ground height z = a*east + b*north + c: a robust plane through the low
    points (10th percentile of height) of each `cell`-metre square; flat if too few squares."""
    keys = np.floor(points[:, :2] / cell).astype(int)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    lows = np.array([[*points[inverse == k, :2].mean(0), np.percentile(points[inverse == k, 2], 10)]
                     for k in np.flatnonzero(counts >= min_points)]).reshape(-1, 3)
    if len(lows) < 6:
        return np.array([0.0, 0.0, float(np.percentile(points[:, 2], 10))])
    for _ in range(3):  # roofs and trees over a square lift its low point: drop the outliers
        design = np.c_[lows[:, :2], np.ones(len(lows))]
        plane = np.linalg.lstsq(design, lows[:, 2], rcond=None)[0]
        residual = np.abs(lows[:, 2] - design @ plane)
        lows = lows[residual <= 2.5 * np.median(residual) + 0.5]
    return plane


def georeference(rec: pycolmap.Reconstruction, gps_files: Iterable[Path], min_frames: int = 20,
                 min_span_m: float = 20.0, sink_m: float = 0.5) -> dict | None:
    """{"origin": [lat, lon], "scene_from_enu", "scene_from_map" (4x4), "ground", "error_m",
    "frames", "extent"} or None if too few frames have GPS (or they don't span enough ground to
    fix the heading). The map frame is ENU sheared onto the ground plane (`ground`: a, b, c of
    z = a*east + b*north + c), `sink_m` below it so ground points stay visible above the map.

    ENU is metres east/north/up from the first track's origin; up is height above that video's
    takeoff (other videos' takeoffs may differ by a metre or so, within the fit's error).
    """
    tracks = [json.loads(Path(p).read_text()) for p in gps_files]
    if not tracks:
        return None
    origin = tuple(tracks[0]["origin"])
    centres = {image.name: image.projection_center() for image in rec.images.values()}
    scene, enu = [], []
    for track in tracks:
        east, north = geo_offset(tuple(track["origin"]), origin)
        for name, position in track["positions"].items():
            if name in centres:
                scene.append(centres[name])
                enu.append(np.asarray(position) + [east, north, 0.0])
    if len(scene) < min_frames:
        return None
    scene, enu = np.array(scene), np.array(enu)
    if np.ptp(enu[:, :2], axis=0).max() < min_span_m:
        return None

    # One trimmed refit: frames far off the track (a misplaced chunk, a GPS glitch) don't tilt it.
    keep = np.ones(len(scene), bool)
    for _ in range(2):
        scale, rotation, translation = similarity(scene[keep], enu[keep])
        error = np.linalg.norm(scale * scene @ rotation.T + translation - enu, axis=1)
        keep = error <= max(3 * np.median(error[keep]), 1.0)

    heights = np.array([p.xyz for p in rec.points3D.values()]) @ (scale * rotation).T + translation
    ground = _ground_plane(heights) - [0, 0, sink_m]

    scene_from_enu = np.eye(4)
    scene_from_enu[:3, :3] = rotation.T / scale
    scene_from_enu[:3, 3] = -rotation.T @ translation / scale
    # The map's own frame: east/north on the ground plane (a slight shear of ENU, fine for slopes
    # of a few degrees).
    enu_from_map = np.eye(4)
    enu_from_map[2, :] = [ground[0], ground[1], 1.0, ground[2]]
    placed = scale * scene @ rotation.T + translation
    return {
        "origin": [float(origin[0]), float(origin[1])],
        "scene_from_enu": scene_from_enu.round(9).tolist(),
        "scene_from_map": (scene_from_enu @ enu_from_map).round(9).tolist(),
        "ground": [round(float(v), 5) for v in ground],
        "error_m": round(float(np.median(error[keep])), 2),
        "frames": int(keep.sum()),
        "extent": [round(float(v), 1) for v in (*placed[:, :2].min(0), *placed[:, :2].max(0))],
    }
