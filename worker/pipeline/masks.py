"""Masks of movable things (vehicles, people, bikes) for feature extraction and training.

A street ride is full of traffic: a car keeping pace with the camera looks static to SfM
and would be painted into the splats as a smear along the road, and the rider is in view all
the time. Every image is segmented (worker.pipeline.sky_net, which also writes the sky masks
sky removal later reuses) and the movable classes are masked out, parked cars too. Each mask
is closed (CLOSE of the width) and its blobs touching the bottom edge filled to their convex
hull (bottom_hulls): the rider's patterned helmet, low in the front views, came out of the
segmentation in pieces, and car windows leave holes. A view's rider region (rider_masks) is
masked in all its frames, as the helmet isn't always segmented at all. Last, masks are
dilated by DILATE of the width to cover outlines.

The sky is masked too (not dilated). Training has to paint every pixel it is given, and from
the street the sky was painted as large faint splats hanging over the road at roof height:
fine from the rider's views, a fog over everything from above. Sky removal after training
(worker.pipeline.sky) misses them, as from the side views they land on buildings, not sky.

Masks live at <scene>/masks/<image name>.png (255 keep, 0 ignore): COLMAP's mask_path
convention, which Brush also finds (a masks/ dir whose subpath matches the image's).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from worker.pipeline.sky import predict_sky

DILATE = 0.01
CLOSE = 0.04
MASKS_DIR = "masks"
MOVABLE_DIR = "movable_masks"  # the network's raw output, 255 on movable classes
SKY_DIR = "sky_masks"


def movable_masks(scene_root: Path, image_root: Path, names: list[str], device: str | None = None,
                  on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Write the masks of `names` (images under image_root); returns how much was masked."""
    predict_sky(image_root, names, scene_root / SKY_DIR, device, on_progress, movable_dir=scene_root / MOVABLE_DIR)
    riders = rider_masks(scene_root / MOVABLE_DIR, names)

    def write(name: str) -> float:
        with Image.open(image_root / name) as image:  # reads the header only
            size = image.size
        movable = cv2.imread(str(scene_root / MOVABLE_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        closing = max(3, int(CLOSE * movable.shape[1]) | 1)
        movable = cv2.morphologyEx(movable, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing)))
        movable = np.maximum(movable, bottom_hulls(movable))
        if (rider := riders.get(name.rsplit("/", 1)[0])) is not None:
            movable = np.maximum(movable, rider)
        movable = cv2.resize(movable, size, interpolation=cv2.INTER_NEAREST)
        k = max(3, int(DILATE * size[0]) | 1)
        movable = cv2.dilate(movable, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        sky = cv2.resize(cv2.imread(str(scene_root / SKY_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE), size,
                         interpolation=cv2.INTER_NEAREST)
        ignore = (movable > 127) | (sky > 127)
        out = scene_root / MASKS_DIR / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.where(ignore, 0, 255).astype(np.uint8))
        return float(ignore.mean())

    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:  # cv2 releases the GIL
        masked = list(pool.map(write, names))
    return {"images": len(names), "masked_mean": round(float(np.mean(masked)), 4) if masked else 0.0,
            "rider": {folder: round(float((mask > 127).mean()), 4) for folder, mask in riders.items()}}


def between_masks(mask_root: Path, triples: list[tuple[str, str, str]]) -> None:
    """Masks of frames between two masked keyframes (extra training frames, worker.pipeline.
    splat.frames): (name, after, before) -> ignored where either neighbour is, which covers what
    moved in between (a rig's keyframes are at most 0.33 s apart) without segmenting again."""
    def write(triple: tuple[str, str, str]) -> None:
        name, after, before = triple
        a, b = (cv2.imread(str(mask_root / f"{n}.png"), cv2.IMREAD_GRAYSCALE) for n in (after, before))
        if a is None or b is None:
            return  # unmasked keyframes: so is the extra
        cv2.imwrite(str(mask_root / f"{name}.png"), np.minimum(a, b))

    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:  # cv2 releases the GIL
        list(pool.map(write, triples))


def bottom_hulls(mask: np.ndarray) -> np.ndarray:
    """The convex hulls of the blobs touching the bottom edge (the rider, a vehicle alongside)."""
    _, labels = cv2.connectedComponents((mask > 127).astype(np.uint8))
    out = np.zeros_like(mask)
    for label in set(np.unique(labels[-1])) - {0}:
        cv2.fillConvexPoly(out, cv2.convexHull(cv2.findNonZero((labels == label).astype(np.uint8))), 255)
    return out


# A view's rider region: pixels movable in more than RIDER_SHARE of its frames, in blobs
# touching the bottom edge. On the first ride the helmet was segmented in 40-53% of the front
# views' frames (missed in others), traffic along the horizon in up to 33% but never at the
# bottom edge.
RIDER_SHARE = 0.2
RIDER_MIN_FRAMES = 30


def rider_masks(movable_dir: Path, names: list[str]) -> dict[str, np.ndarray]:
    """Per image folder (a rig view, or an ordinary video), its rider region (255) at the
    segmentation's resolution; folders with fewer than RIDER_MIN_FRAMES frames are skipped."""
    by_folder: dict[str, list[str]] = {}
    for name in names:
        by_folder.setdefault(name.rsplit("/", 1)[0], []).append(name)
    out = {}
    for folder, members in by_folder.items():
        if len(members) < RIDER_MIN_FRAMES:
            continue
        count = None
        for name in members:
            mask = cv2.imread(str(movable_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 127
            count = mask.astype(np.uint32) if count is None else count + mask
        rider = bottom_hulls((count > RIDER_SHARE * len(members)).astype(np.uint8) * 255)
        if rider.any():
            out[folder] = rider
    return out
