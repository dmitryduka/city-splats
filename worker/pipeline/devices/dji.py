"""DJI flight telemetry from the video's subtitle track, as camera positions in local metres.

DJI drones (seen on a Mini 4: handler "DJI.Subtitle") write one cue per second:

    F/2.8, SS 390.55, ISO 100, EV 0, DZOOM 1.000, GPS (44.4647, 40.2058, 25), D 49.85m, H 62.40m, ...

GPS is (longitude, latitude, satellites) rounded to 1e-4 degree (~11 m north, ~8 m east here);
H is the height above takeoff (barometric, 0.1 m), D the horizontal distance from home (1 cm).
So the vertical is precise, the horizontal coarse; D tightens it: home is fitted to all
(position, D) pairs, then each smoothed position is moved along its bearing from home to its
exact distance D. That leaves the bearing error (a few metres at tens of metres from home).

Enough to keep a reconstruction's scale consistent along the flight: hovering and pure
gimbal turns leave image-only SfM free to solve parts of a flight at different scales.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_EARTH_RADIUS = 6_378_137.0
_CUE_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")
_GPS = re.compile(r"GPS \(([-\d.]+),\s*([-\d.]+)")
_HEIGHT = re.compile(r"\bH ([-\d.]+)m")
_DISTANCE = re.compile(r"\bD ([-\d.]+)m")


@dataclass
class Track:
    """Positions in metres: x east, y north, z up, from (origin_lat, origin_lon) at takeoff height."""

    times: np.ndarray  # (N,) seconds into the video
    positions: np.ndarray  # (N, 3)
    origin_lat: float
    origin_lon: float
    sigma: tuple[float, float, float]  # expected error per axis, metres

    def at(self, timestamps: np.ndarray) -> np.ndarray:
        """Positions at these video times (linear between cues, clamped at the ends)."""
        return np.stack([np.interp(timestamps, self.times, self.positions[:, k]) for k in range(3)], axis=1)


def _cues(video: Path) -> list[tuple[float, str]]:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:s:0", "-f", "srt", "-"],
        capture_output=True, text=True,
    ).stdout
    cues, start = [], None
    for line in out.splitlines():
        if match := _CUE_TIME.search(line):
            h, m, s, ms = (int(x) for x in match.groups())
            start = h * 3600 + m * 60 + s + ms / 1000
        elif start is not None and "GPS (" in line:
            cues.append((start, line))
            start = None
    return cues


def read_track(video: Path, smooth_seconds: float = 5.0, min_bearing_distance: float = 15.0) -> Track | None:
    """The flight track, or None if the video has no DJI telemetry (or no GPS fix)."""
    rows = []
    for start, text in _cues(video):
        gps, height, distance = _GPS.search(text), _HEIGHT.search(text), _DISTANCE.search(text)
        if gps and height and distance:
            lon, lat = float(gps.group(1)), float(gps.group(2))
            if lat or lon:
                rows.append((start + 0.5, lat, lon, float(height.group(1)), float(distance.group(1))))
    if len(rows) < 10:
        return None
    t, lat, lon, height, distance = (np.array(c) for c in zip(*rows))
    lat0, lon0 = float(np.median(lat)), float(np.median(lon))
    x = np.radians(lon - lon0) * _EARTH_RADIUS * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * _EARTH_RADIUS
    # Quantised GPS: average over a few seconds of flight.
    kernel = np.ones(max(int(smooth_seconds), 1))
    norm = np.convolve(np.ones_like(x), kernel, mode="same")
    xy = np.stack([np.convolve(x, kernel, mode="same") / norm, np.convolve(y, kernel, mode="same") / norm], 1)

    # Home from the distances, then exact distances along each bearing.
    far = distance > min_bearing_distance
    if far.sum() >= 5:
        home = least_squares(lambda h: np.linalg.norm(xy[far] - h, axis=1) - distance[far],
                             xy[far].mean(0) - 1e-3, loss="soft_l1", f_scale=3.0).x
        offset = xy[far] - home
        xy[far] = home + offset * (distance[far] / np.maximum(np.linalg.norm(offset, axis=1), 1e-6))[:, None]
    return Track(t, np.column_stack([xy, height]), lat0, lon0, sigma=(3.0, 3.0, 0.3))


def geo_offset(origin: tuple[float, float], reference: tuple[float, float]) -> np.ndarray:
    """Where `origin` (lat, lon) lies in metres east/north of `reference` (fine within a city)."""
    return np.array([np.radians(origin[1] - reference[1]) * _EARTH_RADIUS * np.cos(np.radians(reference[0])),
                     np.radians(origin[0] - reference[0]) * _EARTH_RADIUS])


def similarity(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama: (scale, rotation, translation) with y ~ scale * rotation @ x + translation."""
    mx, my = x.mean(0), y.mean(0)
    u, s, vt = np.linalg.svd((y - my).T @ (x - mx) / len(x))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = float(np.trace(np.diag(s) @ d) / (x - mx).var(0).sum())
    return scale, rotation, my - scale * rotation @ mx


def camera_track_defect(centres: np.ndarray, track_positions: np.ndarray,
                        max_error: float = 8.0, window: float = 25.0, max_scale_drift: float = 2.0) -> tuple[str | None, dict]:
    """Whether a reconstruction's camera centres follow the flight track (up to a similarity):
    median distance after alignment within max_error metres, and every stretch of ~`window`
    metres of flight at the same scale. Returns (problem or None, stats)."""
    x, y = centres, track_positions
    scale, rotation, translation = similarity(x, y)
    aligned = scale * x @ rotation.T + translation
    error = float(np.median(np.linalg.norm(aligned - y, axis=1)))
    # Stretches of `window` metres of flight (by the track), compared by path length.
    path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(y, axis=0), axis=1))])
    cuts = np.searchsorted(path, np.arange(0, path[-1], window))
    drifts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 3:
            flown = path[b] - path[a]
            solved = np.linalg.norm(np.diff(aligned[a:b + 1], axis=0), axis=1).sum()
            drifts.append(solved / max(flown, 1e-6))
    drift = (float(min(drifts)), float(max(drifts))) if drifts else (1.0, 1.0)
    stats = {"track_error_m": round(error, 2), "scale_drift": [round(drift[0], 2), round(drift[1], 2)],
             "metres_per_unit": round(scale, 4)}
    problems = []
    if error > max_error:
        problems.append(f"cameras are {error:.1f} m from the GPS track (median)")
    if drift[0] < 1 / max_scale_drift or drift[1] > max_scale_drift:
        problems.append(f"parts of the flight are solved at {drift[0]:.2f}-{drift[1]:.2f}x the GPS scale")
    return ("; ".join(problems) or None), stats
