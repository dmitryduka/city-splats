"""Terrain under the map: elevation around a georeferenced scene, so the street map is draped
over the hills instead of lying on the scene's ground plane.

Heights come from AWS Terrain Tiles (Terrarium PNGs, free, no key; SRTM 30 m in Armenia).
Copernicus GLO-30 would be better but leaves out Armenia (only its 90 m GLO-90 covers it).
The tiles are a surface model: SRTM's radar sees roofs and trees. Around Yerevan's open
Achapnyak it is within a few metres of the ground, but in the centre's 5-9 storey blocks it
lies 15-25 m above the streets (c43's dense ground under the Cascade and Antarain), so no
single height fits both. The scene's heights are relative (worker.pipeline.georef: ENU up
from the first track's origin), so:
- one offset puts the tiles on the scene where the first video starts: its ground there (the
  dense cloud within TAKEOFF_RADIUS_M of the origin) minus the tiles' height at the origin
  (Yerevan: the c13 drone took off on open ground, scene ground -1.5 m, tiles 1075.6 m). The scene's low points over all its squares would be roof-biased instead:
  from a drone 100 m up, a block's lowest SfM points are often on its roof.
- where the scene has ground, the terrain follows it: per grid square the dense cloud's
  ground, minus the tiles there, median-filtered over MEDIAN_CELLS
  (a square that is all roof), smoothed over CORRECT_SIGMA_M and fading out away from the
  scene (CORRECT_FADE), is added; and the map lies SINK_M below it, so ground points stay
  visible above it.
Ground is the lowest height bin holding PEAK_SHARE of the fullest one (GROUND_BIN_M bins):
depth noise leaves a tail below it (at c13's takeoff the peak is at -1.5 m, the 5th
percentile at -6.9 m), and the street is the lowest of the surfaces a square holds.

The result is a float32 grid of heights in the map frame (geo["scene_from_map"]: z above the
fitted ground plane), north rows first, served next to the version's files; geo["terrain"]
says where it is and how it fits.
"""

from __future__ import annotations

import io
import math
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image
from scipy.ndimage import gaussian_filter, generic_filter

from worker.pipeline.devices.dji import _EARTH_RADIUS

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 12  # ~29 m a pixel at Yerevan's latitude: the data's own 30 m
SPACING_M = 30.0
MARGIN_M = 6000.0  # around the scene's extent: the map shows hills a few km out
TAKEOFF_RADIUS_M = 15.0
GROUND_BIN_M = 0.5
PEAK_SHARE = 0.5
GROUND_MIN_POINTS = 200
MEDIAN_CELLS = 5
CORRECT_SIGMA_M = 100.0
# Coverage (smoothed share of grid nodes with ground) at which the correction is half applied:
# a street ride's line of squares has ~0.1 under it, so it gets ~0.9 of its residual.
CORRECT_FADE = 0.01
SINK_M = 0.5


def _tile_xy(lat: np.ndarray, lon: np.ndarray, z: int) -> tuple[np.ndarray, np.ndarray]:
    """Web-Mercator tile coordinates (fractional) at zoom z."""
    n = 2 ** z
    phi = np.radians(lat)
    return (lon + 180) / 360 * n, (1 - np.log(np.tan(phi) + 1 / np.cos(phi)) / math.pi) / 2 * n


def _tile(z: int, x: int, y: int, cache: Path) -> np.ndarray:
    """One tile's heights (m above sea level), 256x256, cached."""
    path = cache / f"{z}/{x}/{y}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TILE_URL.format(z=z, x=x, y=y), timeout=30) as response:
            path.write_bytes(response.read())
    rgb = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"), dtype=np.float64)
    return rgb[..., 0] * 256 + rgb[..., 1] + rgb[..., 2] / 256 - 32768


def sample_heights(lat: np.ndarray, lon: np.ndarray, cache: Path, z: int = ZOOM) -> np.ndarray:
    """Heights above sea level at lat/lon (bilinear over a mosaic of the tiles they need)."""
    tx, ty = _tile_xy(lat, lon, z)
    px, py = tx * 256 - 0.5, ty * 256 - 0.5  # pixel centres
    x0, y0 = int(np.floor(px.min() / 256)), int(np.floor(py.min() / 256))
    x1, y1 = int(np.floor(px.max() / 256)) + 1, int(np.floor(py.max() / 256)) + 1
    mosaic = np.vstack([np.hstack([_tile(z, x, y, cache) for x in range(x0, x1 + 1)]) for y in range(y0, y1 + 1)])
    px, py = px - x0 * 256, py - y0 * 256
    ix = np.clip(np.floor(px).astype(int), 0, mosaic.shape[1] - 2)
    iy = np.clip(np.floor(py).astype(int), 0, mosaic.shape[0] - 2)
    fx, fy = px - ix, py - iy
    return ((mosaic[iy, ix] * (1 - fx) + mosaic[iy, ix + 1] * fx) * (1 - fy)
            + (mosaic[iy + 1, ix] * (1 - fx) + mosaic[iy + 1, ix + 1] * fx) * fy)


def _latlon(east: np.ndarray, north: np.ndarray, origin: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of devices.dji.geo_offset."""
    lat = origin[0] + np.degrees(north / _EARTH_RADIUS)
    return lat, origin[1] + np.degrees(east / (_EARTH_RADIUS * np.cos(np.radians(origin[0]))))


def _ground(heights: np.ndarray) -> float:
    """The lowest well-filled height bin (its centre); see the module docstring."""
    lo = np.floor(heights.min() / GROUND_BIN_M) * GROUND_BIN_M
    counts, _ = np.histogram(heights, bins=np.arange(lo, heights.max() + 2 * GROUND_BIN_M, GROUND_BIN_M))
    counts = np.convolve(counts, np.ones(3) / 3, mode="same")
    return float(lo + (np.argmax(counts >= PEAK_SHARE * counts.max()) + 0.5) * GROUND_BIN_M)


def build_terrain(points: np.ndarray, cameras: np.ndarray, geo: dict, out: Path, cache: Path) -> dict:
    """Write the grid to `out`; returns geo["terrain"] (but its "url"). `points`: a sample of the
    dense cloud, `cameras`: camera centres (both in the scene frame; the cameras give the extent)."""
    origin = (float(geo["origin"][0]), float(geo["origin"][1]))
    enu_from_scene = np.linalg.inv(np.array(geo["scene_from_enu"]))
    enu = points @ enu_from_scene[:3, :3].T + enu_from_scene[:3, 3]
    cams = cameras @ enu_from_scene[:3, :3].T + enu_from_scene[:3, 3]

    near = np.hypot(enu[:, 0], enu[:, 1]) < TAKEOFF_RADIUS_M
    if near.sum() < GROUND_MIN_POINTS:
        raise ValueError(f"only {near.sum()} dense points near the scene's origin")
    takeoff = _ground(enu[near, 2])
    offset = takeoff - float(sample_heights(np.array([origin[0]]), np.array([origin[1]]), cache)[0])

    (min_e, min_n), (max_e, max_n) = cams[:, :2].min(0), cams[:, :2].max(0)  # geo["extent"]: GPS frames only
    x0, y0 = math.floor((min_e - MARGIN_M) / SPACING_M) * SPACING_M, math.floor((min_n - MARGIN_M) / SPACING_M) * SPACING_M
    nx = int(math.ceil((max_e + MARGIN_M - x0) / SPACING_M)) + 1
    ny = int(math.ceil((max_n + MARGIN_M - y0) / SPACING_M)) + 1
    east, north = np.meshgrid(x0 + SPACING_M * np.arange(nx), y0 + SPACING_M * np.arange(ny)[::-1])
    up = sample_heights(*_latlon(east.ravel(), north.ravel(), origin), cache).reshape(ny, nx) + offset

    # The scene's ground on the grid's nodes (each point to its nearest node), its residual.
    col = np.clip(np.round((enu[:, 0] - x0) / SPACING_M).astype(int), 0, nx - 1)
    row = np.clip(np.round((ny - 1) - (enu[:, 1] - y0) / SPACING_M).astype(int), 0, ny - 1)
    node = row * nx + col
    order = np.argsort(node, kind="stable")
    nodes, starts, counts = np.unique(node[order], return_index=True, return_counts=True)
    residual = np.full(ny * nx, np.nan)
    for k, start, count in zip(nodes, starts, counts):
        if count >= GROUND_MIN_POINTS:
            residual[k] = _ground(enu[order[start:start + count], 2]) - up.flat[k]
    residual = residual.reshape(ny, nx)
    covered = np.isfinite(residual)
    with warnings.catch_warnings():  # windows with no ground at all: NaN, dropped by `covered`
        warnings.simplefilter("ignore", RuntimeWarning)
        residual = np.where(covered, generic_filter(residual, np.nanmedian, size=MEDIAN_CELLS, mode="nearest"), 0.0)
    sigma = CORRECT_SIGMA_M / SPACING_M
    coverage = gaussian_filter(covered.astype(float), sigma)
    mean = gaussian_filter(residual, sigma) / np.maximum(coverage, 1e-9)
    up += mean * coverage / (coverage + CORRECT_FADE) - SINK_M
    a, b, c = geo["ground"]
    out.parent.mkdir(parents=True, exist_ok=True)
    (up - (a * east + b * north + c)).astype("<f4").tofile(out)  # map-frame z, north row first
    found = residual[covered]
    return {"source": "AWS Terrain Tiles (SRTM)", "zoom": ZOOM, "spacing_m": SPACING_M,
            "size": [nx, ny], "min": [x0, y0], "offset_m": round(offset, 2), "takeoff_ground_m": round(takeoff, 2),
            "ground_nodes": int(covered.sum()), "correction_m": [round(float(v), 1) for v in np.percentile(found, [5, 50, 95])]}
