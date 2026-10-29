"""Masks of movable things (vehicles, people, bikes) for feature extraction and training.

A street ride is full of traffic: a car keeping pace with the camera looks static to SfM
and would be painted into the splats as a smear along the road, and the rider is in view all
the time. Every image is segmented (worker.pipeline.sky_net, which also writes the sky masks
sky removal later reuses) and the movable classes are masked out, dilated by DILATE of the
width to cover their outlines. Parked cars are masked too.

Masks live at <scene>/masks/<image name>.png (255 keep, 0 ignore): COLMAP's mask_path
convention, which Brush also finds (a masks/ dir whose subpath matches the image's).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from worker.pipeline.sky import predict_sky

DILATE = 0.01
MASKS_DIR = "masks"
MOVABLE_DIR = "movable_masks"  # the network's raw output, 255 on movable classes
SKY_DIR = "sky_masks"


def movable_masks(scene_root: Path, image_root: Path, names: list[str], device: str | None = None,
                  on_progress: Callable[[int, int], None] | None = None) -> dict:
    """Write the masks of `names` (images under image_root); returns how much was masked."""
    predict_sky(image_root, names, scene_root / SKY_DIR, device, on_progress, movable_dir=scene_root / MOVABLE_DIR)
    masked = []
    for name in names:
        with Image.open(image_root / name) as image:  # reads the header only
            size = image.size
        movable = cv2.imread(str(scene_root / MOVABLE_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        movable = cv2.resize(movable, size, interpolation=cv2.INTER_NEAREST)
        k = max(3, int(DILATE * size[0]) | 1)
        movable = cv2.dilate(movable, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        out = scene_root / MASKS_DIR / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.where(movable > 127, 0, 255).astype(np.uint8))
        masked.append(float((movable > 127).mean()))
    return {"images": len(names), "masked_mean": round(float(np.mean(masked)), 4) if masked else 0.0}
