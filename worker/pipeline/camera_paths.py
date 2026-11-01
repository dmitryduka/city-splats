"""Each video's camera path (pose over time) for the viewer to replay the footage."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycolmap

from worker.pipeline.devices.insta360 import read_rig

# Gaussian sigma (seconds) for smoothing orientation. An Insta360 on a bike handlebar shakes by
# a few degrees between keyframes 0.13-0.5 s apart; 0.25 s takes out that jitter, while a turn
# (seconds long) keeps its timing because the window is symmetric. Drone gimbals barely move.
ORIENTATION_SIGMA_S = 0.25


def _smooth_quats(t: np.ndarray, q: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-weighted mean of nearby orientations (sign-aligned quaternions, renormalized;
    close to the true rotation mean for the few degrees of spread within a window)."""
    out = np.empty_like(q)
    for i in range(len(q)):
        w = np.exp(-0.5 * ((t - t[i]) / sigma) ** 2)
        near = w > 1e-3
        aligned = q[near] * np.sign(q[near] @ q[i])[:, None]
        mean = (w[near, None] * aligned).sum(0)
        out[i] = mean / np.linalg.norm(mean)
    return out


def export_camera_paths(rec: pycolmap.Reconstruction, image_root: Path, out_path: Path) -> Path:
    """Write {"c<id>": [[t, x, y, z, qx, qy, qz, qw], ...], "fov": {"c<id>": degrees}}, rows
    sorted by t: the camera centre and
    world_from_cam rotation (COLMAP axes: looking down +z, y down) of every posed keyframe, in
    the reconstruction's frame (the one the layers are in), and each video's vertical field of
    view (2 atan(height / 2 fy) of its keyframes' camera; 90 for a rig's views). Keyed by image prefix, not by
    contribution: a cloned scene shares these files with the original under other ids.

    Rig videos use their reference view (l0_front: the riding direction). Timestamps come
    from the video's frames.json, which lives with the scene that extracted the frames
    (images/c<id> of a cloned scene links there)."""
    by_prefix: dict[str, dict[str, pycolmap.Image]] = defaultdict(dict)
    for image in rec.images.values():
        if image.has_pose:
            prefix, rest = image.name.split("/", 1)
            by_prefix[prefix][rest] = image

    paths: dict = {}
    fovs = {}
    for prefix, images in sorted(by_prefix.items()):
        video_dir = (image_root / prefix).resolve()
        frames_json = video_dir.parent.parent / "contributions" / prefix / "frames.json"
        if not frames_json.exists():
            continue
        rig = read_rig(video_dir)
        view = f"{rig['ref']}/" if rig else ""
        samples, fov = [], []
        for frame in json.loads(frames_json.read_text()):
            image = images.get(view + frame["name"])
            if image is None:
                continue
            world_from_cam = image.cam_from_world().inverse()
            samples.append([frame["timestamp"], *world_from_cam.translation, *world_from_cam.rotation.quat])
            camera = rec.cameras[image.camera_id]
            fov.append(math.degrees(2 * math.atan(camera.height / 2 / camera.focal_length_y)))
        if len(samples) < 2:
            continue
        a = np.array(sorted(samples, key=lambda s: s[0]))
        a[:, 4:] = _smooth_quats(a[:, 0], a[:, 4:], ORIENTATION_SIGMA_S)
        paths[prefix] = np.round(a, 4).tolist()
        # Videos refined to one camera have a single value; the median also covers per-frame ones.
        fovs[prefix] = round(float(np.median(fov)), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(paths | {"fov": fovs}, separators=(",", ":")))
    return out_path
