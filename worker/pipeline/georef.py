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

    return _geo(rec, scale, rotation, translation, origin, float(np.median(error[keep])), int(keep.sum()),
                scene, sink_m)


def georeference_anchors(rec: pycolmap.Reconstruction, anchors: dict[str, dict], sink_m: float = 0.5) -> dict | None:
    """georeference() from a few known positions instead of a GPS track: anchors maps a
    video prefix (c<id>) to {"start": [lat, lon]} and/or {"end": [lat, lon]}, pinning its
    first or last registered moment. Two anchors fix position, scale and heading on the
    ground; up is the cameras' mean up direction (a rider or walker holds the camera
    upright, so tilts average out), heights are relative to the first anchor's camera."""
    by_prefix: dict[str, list[pycolmap.Image]] = {}
    for image in rec.images.values():
        if image.has_pose:
            by_prefix.setdefault(image.name.split("/")[0], []).append(image)
    points: list[tuple[np.ndarray, tuple[float, float]]] = []
    for prefix, ends in anchors.items():
        images = sorted(by_prefix.get(prefix, []), key=lambda im: (im.name.split("/")[-1], im.name))
        for end, latlon in ends.items():
            if images:
                image = images[0] if end == "start" else images[-1]
                points.append((image.projection_center(), (float(latlon[0]), float(latlon[1]))))
    if len(points) < 2:
        return None
    origin = points[0][1]
    # A camera's up is -y; a rig's views tilted upwards (folders "*_up") don't count.
    ups = np.array([-image.cam_from_world().rotation.matrix()[1] for image in rec.images.values()
                    if image.has_pose and not image.name.rsplit("/", 2)[-2:][0].endswith("_up")])
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    # Scene axes: x, y horizontal, z up.
    x = np.cross(up, [0.0, 0.0, 1.0] if abs(up[2]) < 0.9 else [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    level = np.stack([x, np.cross(up, x), up])
    a, b = (level @ points[0][0]), (level @ points[-1][0])
    east, north = geo_offset(points[-1][1], origin)
    if np.hypot(*(b - a)[:2]) < 1e-6 or np.hypot(east, north) < 1.0:
        return None
    scale = np.hypot(east, north) / np.hypot(*(b - a)[:2])
    theta = np.arctan2(north, east) - np.arctan2(*(b - a)[1::-1])
    turn = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    rotation = turn @ level
    translation = -scale * rotation @ points[0][0]
    centres = np.array([image.projection_center() for image in rec.images.values() if image.has_pose])
    return _geo(rec, scale, rotation, translation, origin, 0.0, len(points), centres, sink_m)


def _geo(rec: pycolmap.Reconstruction, scale: float, rotation: np.ndarray, translation: np.ndarray,
         origin: tuple[float, float], error_m: float, frames: int, scene: np.ndarray, sink_m: float) -> dict:
    """The georeference record of enu = scale * rotation @ scene + translation."""
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
        "error_m": round(error_m, 2),
        "frames": frames,
        "extent": [round(float(v), 1) for v in (*placed[:, :2].min(0), *placed[:, :2].max(0))],
    }


CAMERA_HEIGHT_M = 1.5  # a rider's or walker's camera above the ground


def anchor_offsets(anchors: dict, geo: dict) -> list[np.ndarray]:
    """Where a video's anchors ({"start"/"end": [lat, lon]}) lie in a scene's ENU (metres)."""
    return [geo_offset((float(ll[0]), float(ll[1])), tuple(geo["origin"])) for ll in anchors.values()]


def scene_from_anchors(rec: pycolmap.Reconstruction, prefix: str, anchors: dict, geo: dict) -> pycolmap.Sim3d | None:
    """Place a video's own model (prefix c<id>) in a georeferenced scene by its start/end
    anchors alone, for a video that shares no view with the scene (a ride across town):
    georeference_anchors fixes it in ENU around its start, the scene's geo maps ENU into the
    scene; the start camera goes CAMERA_HEIGHT_M above the scene's ground plane, extended to
    there (a guess on hilly ground, metres off a few km out)."""
    own = georeference_anchors(rec, {prefix: anchors})
    if own is None or "start" not in anchors:
        return None
    enu_from_piece = np.linalg.inv(np.array(own["scene_from_enu"]))
    east, north = geo_offset(tuple(own["origin"]), tuple(geo["origin"]))
    a, b, c = geo["ground"]
    height = a * east + b * north + c + 0.5 + CAMERA_HEIGHT_M  # geo's ground sinks 0.5 m
    shift = np.eye(4)
    shift[:3, 3] = [east, north, height]
    scene_from_piece = np.array(geo["scene_from_enu"]) @ shift @ enu_from_piece
    return pycolmap.Sim3d(scene_from_piece[:3])
