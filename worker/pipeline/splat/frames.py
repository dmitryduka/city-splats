"""More and sharper training frames than the SfM keyframes, without rerunning SfM.

- One camera per video: chunked SfM gives each chunk its own intrinsics, and some drift far
  from the real lens (Achapnyak: fx 374 / fy 1602 and fx 1531 / fy 451 for a DJI camera with
  square pixels, 85 of 2929 frames). Frames of such chunks are put on the video's most used
  plausible camera (fx ~ fy) and re-posed from their own 2D-3D matches; chunks within 2% of
  it are left alone (some of their frames have no observations left to re-pose from).
- Full resolution: the keyframes are decoded again from the video at up to `side` pixels
  (they were stored at 1920; four of Achapnyak's five videos are 4K). Brush derives the
  field of view and principal point from the model's intrinsics relative to the camera
  size, so the 1920 model serves the larger images as long as the aspect ratio matches.
- Extra frames: `extra` frames per gap between consecutive keyframes (the sharpest candidate
  around the gap's middle), posed by tracking the neighbouring keyframes' SfM observations
  into them (optical flow, forward-backward checked) and solving PnP against those points.

Images are cached per video under the scene's train_images/<video>/ (decoding is the slow
part); poses are recomputed for each training run against its reconstruction.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from worker.pipeline.ingest import _analyse, decode_samples, probe, sample_ranges

# Ingest's candidate rate: keyframe timestamps are sample / 15 (worker.pipeline.ingest).
CANDIDATE_FPS = 15.0
# An extra frame's pose is kept with at least this many PnP inliers, and at least this
# share of its tracked points as inliers.
MIN_INLIERS = 100
MIN_INLIER_RATIO = 0.5


@dataclass
class Video:
    prefix: str  # image name prefix in the reconstruction ("c13")
    path: Path  # the uploaded video
    frames: list[dict]  # its frames.json (name, timestamp, ...)
    segments: int  # how many segments ingest decoded it in (frame numbering depends on it)


@dataclass
class FrameStats:
    videos: dict = field(default_factory=dict)  # prefix -> per-video counts
    keyframes: int = 0
    full_resolution: int = 0
    extra_candidates: int = 0
    extra_posed: int = 0
    extra_held_out: int = 0  # skipped: next to a held-out keyframe
    reposed: int = 0
    dropped: int = 0

    def as_dict(self) -> dict:
        return self.__dict__


def _plausible(camera: pycolmap.Camera) -> bool:
    fx, fy = camera.params[:2]
    return abs(fx / fy - 1) < 0.01


def _close(camera: pycolmap.Camera, ref: pycolmap.Camera) -> bool:
    return _plausible(camera) and all(abs(f / r - 1) < 0.02 for f, r in zip(camera.params[:2], ref.params[:2]))


def _pose(points2D: np.ndarray, points3D: np.ndarray, camera: pycolmap.Camera) -> dict | None:
    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = 4.0
    options.ransac.random_seed = 0
    return pycolmap.estimate_and_refine_absolute_pose(points2D, points3D, camera, options)


def shared_cameras(rec: pycolmap.Reconstruction, stats: FrameStats) -> dict[str, int]:
    """Put the frames of a video's implausible cameras on its most used plausible one,
    re-posing them from their 2D-3D matches; frames that don't fit are deregistered. Returns
    the reference camera per video prefix (extra frames use it)."""
    by_video: dict[str, Counter] = defaultdict(Counter)
    for image in rec.images.values():
        if image.has_pose:
            by_video[image.name.split("/")[0]][image.camera_id] += 1
    chosen = {}
    for prefix, counts in by_video.items():
        ranked = sorted(counts, key=lambda c: (_plausible(rec.cameras[c]), counts[c]), reverse=True)
        ref = chosen[prefix] = ranked[0]
        camera = rec.cameras[ref]
        info = stats.videos.setdefault(prefix, {})
        info |= {"camera": ref, "focal": [round(float(f), 1) for f in camera.params[:2]],
                 "cameras_before": {str(c): [round(float(f), 1) for f in rec.cameras[c].params[:2]] for c in counts}}
        fix = {c for c in counts if c != ref and not _close(rec.cameras[c], camera)}
        moved = []
        for image in [im for im in rec.images.values() if im.has_pose and im.name.startswith(prefix + "/")]:
            if image.camera_id not in fix:
                continue
            obs = [(p.xy, rec.points3D[p.point3D_id].xyz) for p in image.points2D if p.has_point3D()]
            result = _pose(np.array([o[0] for o in obs]), np.array([o[1] for o in obs]), camera) if len(obs) >= 6 else None
            if result is None or result["num_inliers"] < max(30, 0.3 * len(obs)):
                rec.deregister_frame(image.frame_id)
                stats.dropped += 1
                info["dropped"] = info.get("dropped", 0) + 1
                continue
            before = image.projection_center()
            rec.frame(image.frame_id).rig_from_world = result["cam_from_world"]
            moved.append(float(np.linalg.norm(rec.image(image.image_id).projection_center() - before)))
            stats.reposed += 1
        for cid in fix:
            rec.camera(cid).params = camera.params
        if moved:
            info |= {"reposed": len(moved), "moved_median": round(float(np.median(moved)), 4)}
    return chosen


def _gap_picks(samples: list[tuple[int, float, float, float]], keys: list[int], extra: int) -> list[tuple[int, int, int]]:
    """(sample, previous keyframe index, j) for `extra` frames per gap: the sharpest
    candidate in the middle half of each of the gap's `extra` equal parts."""
    sharp = {i: s for i, _, s, _ in samples}
    picks = []
    for k in range(len(keys) - 1):
        a, b = keys[k], keys[k + 1]
        for j in range(extra):
            lo, hi = a + (b - a) * j / extra, a + (b - a) * (j + 1) / extra
            q = (hi - lo) / 4
            candidates = [i for i in range(int(np.ceil(lo + q)), int(np.floor(hi - q)) + 1) if a < i < b and i in sharp]
            if candidates:
                picks.append((max(candidates, key=sharp.__getitem__), k, j))
    return picks


def plan_images(video: Video, out_dir: Path, side: int, extra: int, pool: Executor,
                tick: Callable[[int], None]) -> dict:
    """What to decode for a video: its meta.json if an earlier run made these images, else
    its keyframes and `extra` frames per gap (sharpness measured over the whole video, like
    ingest's first pass; `tick` gets candidates analysed, per 10). The meta is
    {"side", "extra", "keyframes": [name], "extras": [{name, after, before, sample}],
    "decoded": bool}."""
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["side"] == side and meta["extra"] == extra:
            return meta
    info = probe(video.path)
    fps = min(CANDIDATE_FPS, info.fps)
    keys = [round(f["timestamp"] * fps) for f in video.frames]
    names = [f["name"] for f in video.frames]
    extras = []
    if extra:
        parts = [pool.submit(_analyse, info, fps, a, b, lambda: tick(10))
                 for a, b in sample_ranges(info, fps, video.segments)]
        samples = [s for part in parts for s in part.result()]
        for sample, k, j in _gap_picks(samples, keys, extra):
            extras.append({"name": f"{Path(names[k]).stem}_{j + 1}.jpg", "after": names[k],
                           "before": names[k + 1], "sample": sample})
    return {"side": side, "extra": extra, "keyframes": names, "keys": keys, "extras": extras, "decoded": False}


def candidates(video: Video) -> int:
    """Samples plan_images analyses for a video (for progress totals)."""
    info = probe(video.path)
    return int(info.duration * min(CANDIDATE_FPS, info.fps))


def decode_images(video: Video, meta: dict, out_dir: Path, image_root: Path, pool: Executor,
                  tick: Callable[[int], None]) -> dict:
    """Decode the planned frames into out_dir: keyframes at up to meta["side"] px (linked to
    the stored ones when side is 0) and the extras. `tick` gets frames written, per 10."""
    if meta["decoded"]:
        return meta
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video.path)
    fps = min(CANDIDATE_FPS, info.fps)
    todo = {e["sample"]: e["name"] for e in meta["extras"]}
    if meta["side"]:
        todo |= dict(zip(meta["keys"], meta["keyframes"]))
    else:
        for name in meta["keyframes"]:
            link = out_dir / name
            link.unlink(missing_ok=True)
            link.symlink_to((image_root / video.prefix / name).resolve())
    decode_samples(info, todo, out_dir, sample_ranges(info, fps, video.segments), fps, meta["side"] or 1920, pool,
                   lambda done: tick(10) if done % 10 == 0 else None)
    meta = meta | {"decoded": True, "width": info.width, "height": info.height}
    (out_dir / "meta.json").write_text(json.dumps(meta))
    return meta


def _gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image.shape[1] != size[0]:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image


def _track(rec: pycolmap.Reconstruction, keyframe: pycolmap.Image, key_gray: np.ndarray,
           gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The keyframe's SfM observations followed into `gray`: (points2D, points3D)."""
    obs = [(p.xy, rec.points3D[p.point3D_id].xyz) for p in keyframe.points2D if p.has_point3D()]
    if not obs:
        return np.empty((0, 2)), np.empty((0, 3))
    start = np.array([o[0] for o in obs], dtype=np.float32).reshape(-1, 1, 2)
    params = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(key_gray, gray, start, None, **params)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(gray, key_gray, moved, None, **params)
    good = (ok[:, 0] == 1) & (ok_back[:, 0] == 1) & (np.linalg.norm((back - start)[:, 0], axis=1) < 0.5)
    return moved[good, 0].astype(np.float64), np.array([o[1] for o in obs])[good]


def add_extra_frames(rec: pycolmap.Reconstruction, videos: list[Video], metas: dict[str, dict], cameras: dict[str, int],
                     image_root: Path, train_root: Path, held_out: set[str], stats: FrameStats,
                     on_progress: Callable[[int, int], None], workers: int = 6) -> dict[str, str]:
    """Pose the extra frames and add them to `rec` (no observations; Brush only needs poses).
    Returns extra frame name -> the keyframe it follows (blocks take it with that keyframe)."""
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    jobs = []
    for video in videos:
        for e in metas[video.prefix]["extras"]:
            after, before = f"{video.prefix}/{e['after']}", f"{video.prefix}/{e['before']}"
            stats.extra_candidates += 1
            if after in held_out or before in held_out:
                stats.extra_held_out += 1  # would be a near-copy of a held-out frame
                continue
            anchors = [by_name[n] for n in (after, before) if n in by_name]
            if anchors:
                jobs.append((video.prefix, e, anchors))
    lock = threading.Lock()
    done = [0]

    def pose(job):
        prefix, e, anchors = job
        camera = rec.cameras[cameras[prefix]]
        size = (camera.width, camera.height)
        gray = _gray(train_root / prefix / e["name"], size)
        tracked = [_track(rec, a, _gray(image_root / a.name, size), gray) for a in anchors]
        points2D = np.concatenate([t[0] for t in tracked])
        points3D = np.concatenate([t[1] for t in tracked])
        result = _pose(points2D, points3D, camera) if len(points2D) >= MIN_INLIERS else None
        with lock:
            done[0] += 1
            if done[0] % 20 == 0:
                on_progress(done[0], len(jobs))
        if result is None or result["num_inliers"] < max(MIN_INLIERS, MIN_INLIER_RATIO * len(points2D)):
            return None
        return prefix, e, result["cam_from_world"], result["num_inliers"]

    with ThreadPoolExecutor(workers) as pool:
        posed = [p for p in pool.map(pose, jobs) if p is not None]
    follows = {}
    # A trivial frame takes the image's id, so the id must be free as a frame id too.
    next_id = max(max(rec.images), max(rec.frames)) + 1
    inliers = defaultdict(list)
    for prefix, e, cam_from_world, n in posed:
        name = f"{prefix}/{e['name']}"
        image = pycolmap.Image(name=name, camera_id=cameras[prefix], image_id=next_id)
        rec.add_image_with_trivial_frame(image, cam_from_world)
        next_id += 1
        follows[name] = f"{prefix}/{e['after']}"
        inliers[prefix].append(n)
    for prefix, values in inliers.items():
        stats.videos[prefix] |= {"extra_posed": len(values), "extra_inliers_median": int(np.median(values))}
    stats.extra_posed = len(posed)
    on_progress(len(jobs), len(jobs))
    return follows


def copy_poses(src: pycolmap.Reconstruction, dst: pycolmap.Reconstruction) -> None:
    """Give `dst` (the same model, e.g. with fewer points) src's intrinsics and poses, and
    deregister its frames that src no longer has."""
    for cid, camera in src.cameras.items():
        dst.camera(cid).params = camera.params
    poses = {im.name: im.frame_id for im in src.images.values() if im.has_pose}
    for image in list(dst.images.values()):
        if not image.has_pose:
            continue
        if image.name in poses:
            dst.frame(image.frame_id).rig_from_world = src.frame(poses[image.name]).rig_from_world
        else:
            dst.deregister_frame(image.frame_id)
