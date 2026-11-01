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
            if images and end in ("start", "end"):
                image = images[0] if end == "start" else images[-1]
                points.append((image.projection_center(), (float(latlon[0]), float(latlon[1]))))
    if len(points) < 2:
        return None
    origin = points[0][1]
    level = _level(rec)
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


def _level(rec: pycolmap.Reconstruction) -> np.ndarray:
    """Rotation into axes x, y horizontal and z up, up being the cameras' mean up (-y; a rig's
    views tilted upwards, folders "*_up", don't count)."""
    ups = np.array([-image.cam_from_world().rotation.matrix()[1] for image in rec.images.values()
                    if image.has_pose and not image.name.rsplit("/", 2)[-2:][0].endswith("_up")])
    up = ups.mean(0)
    up /= np.linalg.norm(up)
    x = np.cross(up, [0.0, 0.0, 1.0] if abs(up[2]) < 0.9 else [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


def warp_to_waypoints(rec: pycolmap.Reconstruction, prefix: str, timestamps: dict[str, float], anchors: dict,
                      out_dir: Path) -> pycolmap.Reconstruction | None:
    """A video's model bent to timed positions along its way, for a long ride whose scale
    drifts (fast straight roads lined with repeating fences and pillars, a ride's first
    reconstruction: 2.0 km for a 3.1 km route). anchors: {"start": [lat, lon], "end": [lat, lon],
    "waypoints": [[seconds, lat, lon], ...]} (seconds into the video, e.g. where each of the
    camera's files ends). Each stretch between consecutive known positions (start at the first
    moment, end at the last) gets its own scale and heading on the ground, so the track passes
    through all of them and stays continuous; inside a stretch the geometry is as mapped.
    Heights carry over from stretch to stretch. Returns the model in metres east/north/up of the
    start anchor (written to out_dir), None without waypoints or start/end."""
    from worker.pipeline.pieces import Piece, compose

    if not anchors.get("waypoints") or "start" not in anchors or "end" not in anchors:
        return None
    centres: dict[str, list[np.ndarray]] = {}
    for image in rec.images.values():
        if image.has_pose and image.name.startswith(prefix + "/"):
            centres.setdefault(image.name.rsplit("/", 1)[-1], []).append(image.projection_center())
    frames = sorted(centres, key=lambda name: timestamps[name])
    if len(frames) < 2:
        return None
    level = _level(rec)
    times = np.array([timestamps[name] for name in frames])
    track = np.array([level @ np.mean(centres[name], axis=0) for name in frames])
    start = (float(anchors["start"][0]), float(anchors["start"][1]))
    knots = [(times[0], start)] + [(float(t), (float(lat), float(lon))) for t, lat, lon in sorted(anchors["waypoints"])
                                   if times[0] < t < times[-1]]
    knots.append((times[-1], (float(anchors["end"][0]), float(anchors["end"][1]))))
    at = [np.array([np.interp(t, times, track[:, axis]) for axis in range(3)]) for t, _ in knots]
    ground = [np.array(geo_offset(latlon, start)) for _, latlon in knots]
    pieces, height = [], 0.0
    for j in range(len(knots) - 1):
        a, b = at[j], at[j + 1]
        scale = np.linalg.norm(ground[j + 1] - ground[j]) / max(np.linalg.norm((b - a)[:2]), 1e-9)
        theta = np.arctan2(*(ground[j + 1] - ground[j])[::-1]) - np.arctan2((b - a)[1], (b - a)[0])
        turn = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rotation = turn @ level
        translation = np.array([*ground[j], height]) - scale * turn @ a
        height += scale * (b - a)[2]
        last = j == len(knots) - 2
        keep = {name for name, t in zip(frames, times) if knots[j][0] <= t and (t < knots[j + 1][0] or last)}
        part = pycolmap.Reconstruction(rec)
        for frame_id in {im.frame_id for im in part.images.values() if im.has_pose
                         and im.name.rsplit("/", 1)[-1] not in keep}:
            part.deregister_frame(frame_id)
        (out_dir / f"stretch{j}").mkdir(parents=True, exist_ok=True)
        part.write(out_dir / f"stretch{j}")
        pieces.append(Piece(out_dir / f"stretch{j}", Path(), pycolmap.Sim3d(scale, pycolmap.Rotation3d(rotation),
                                                                            translation)))
    return compose(pieces, out_dir / "warped")


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
    return [geo_offset((float(ll[0]), float(ll[1])), tuple(geo["origin"])) for end, ll in anchors.items()
            if end in ("start", "end")]


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


def scene_from_track(gps_file: Path, geo: dict) -> pycolmap.Sim3d:
    """Place a video with its own GPS track (its model is in the track's metres: east, north
    and height above takeoff around the track's origin, worker.pipeline.sfm) in a georeferenced
    scene it shares no matched view with: the takeoff ground goes on the scene's ground plane,
    extended to there, which snap_to_points then corrects where the two overlap."""
    origin = json.loads(gps_file.read_text())["origin"]
    east, north = geo_offset((float(origin[0]), float(origin[1])), tuple(geo["origin"]))
    a, b, c = geo["ground"]
    shift = np.eye(4)
    shift[:3, 3] = [east, north, a * east + b * north + c + 0.5]  # geo's ground sinks 0.5 m
    return pycolmap.Sim3d((np.array(geo["scene_from_enu"]) @ shift)[:3])


# snap_to_points: shifts tried as starts (metres up), the shrinking match radii (metres), and
# the fewest points that must end within SNAP_FIT_M of the scene's to accept the shift. DJI_0528
# (Mini 3, 150 k SfM points from 12-58 m up) against the street ride in Yerevan: from 0, +5 and
# +10 m the fit reached the same shift (2.1 m east, 2.7 m north, 5.1 m up; 112 points within
# 1 m); starts of +15 m and more stopped at other shifts with 32-69 points. The shift also put
# the ride's cameras 1.5-2 m over the drone's ground, as on a rider's helmet (5-7 m before).
SNAP_STARTS_M = (-10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
SNAP_RADII_M = (6.0, 4.0, 3.0, 2.0, 1.5)
SNAP_FIT_M = 1.0
SNAP_MIN_POINTS = 50


def snap_to_points(scene_from_piece: pycolmap.Sim3d, piece: pycolmap.Reconstruction, scene: pycolmap.Reconstruction,
                   geo: dict) -> tuple[pycolmap.Sim3d, dict | None]:
    """A placement by GPS moved (translation only, in ENU metres) onto the scene's SfM points
    where the two overlap: a trimmed point-to-point fit, started from several heights (aerial
    and street views share few exact points, so a single start can stop at the wrong one).
    Unchanged (stats None) when too few points agree."""
    from scipy.spatial import cKDTree

    enu_from_scene = np.linalg.inv(np.array(geo["scene_from_enu"]))
    to_enu = lambda xyz: xyz @ enu_from_scene[:3, :3].T + enu_from_scene[:3, 3]  # noqa: E731
    placed = np.array(scene_from_piece.matrix())
    ours = to_enu(np.array([p.xyz for p in piece.points3D.values() if p.track.length() >= 3]) @ placed[:3, :3].T
                  + placed[:3, 3])
    theirs = to_enu(np.array([p.xyz for p in scene.points3D.values() if p.track.length() >= 3]))
    lo, hi = ours[:, :2].min(0) - 20, ours[:, :2].max(0) + 20
    theirs = theirs[(theirs[:, :2] > lo).all(1) & (theirs[:, :2] < hi).all(1)]
    if len(theirs) < SNAP_MIN_POINTS:
        return scene_from_piece, None
    tree = cKDTree(theirs)
    best = None
    for start in SNAP_STARTS_M:
        shift = np.array([0.0, 0.0, start])
        for radius in SNAP_RADII_M:
            for _ in range(6):
                dist, idx = tree.query(ours + shift, distance_upper_bound=radius)
                ok = np.isfinite(dist)
                if not ok.any():
                    break
                shift += np.median(theirs[idx[ok]] - (ours[ok] + shift), axis=0)
        dist, _ = tree.query(ours + shift, distance_upper_bound=SNAP_FIT_M)
        fit = int(np.isfinite(dist).sum())
        if best is None or fit > best[1]:
            best = (shift.copy(), fit)
    shift, fit = best
    stats = {"shift_m": [round(float(v), 2) for v in shift], "points_within_m": fit, "scene_points": len(theirs)}
    if fit < SNAP_MIN_POINTS:
        return scene_from_piece, stats | {"applied": False}
    move = np.eye(4)
    move[:3, 3] = shift
    scene_from_enu = np.array(geo["scene_from_enu"])
    snapped = scene_from_enu @ move @ np.linalg.inv(scene_from_enu) @ np.vstack([placed, [0, 0, 0, 1]])
    return pycolmap.Sim3d(snapped[:3]), stats | {"applied": True}
