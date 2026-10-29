"""Insta360 X4 .insv: two fisheye lenses (one 3840x3840 HEVC stream each) as a rig of
virtual pinhole views.

Each keyframe moment becomes VIEWS pinhole images (90 degrees, SIDE px; front/left/right/up
per lens; the downward views would see the rider and vehicle), written as
<images>/<view>/<frame>.jpg. A view's pixels come from one lens only, so there is no stitching
seam; side views reach 5 degrees past the lens's 90, into the other lens's field, so the two
lenses' views share features. rig.json next to them holds the views' intrinsics and
cam_from_rig (later stages detect a rig by it).

Lens model: the factory calibration in the file's trailer (metadata field 54: per lens the
unified "Mei" model, xi + pinhole + 3 radial and 2 tangential terms, in pixels of the
8000x6000 sensor; the video is a centred 5632x5632 crop of it). On the first test ride it
stitched both lenses into a panorama with continuous road markings and kerbs at both seams;
lens 1 is lens 0 turned 180 degrees about the vertical (its 3 cm offset is ignored).
"""

from __future__ import annotations

import json
import struct
import threading
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from worker.pipeline.ingest import VideoInfo, iter_frames

MAGIC = b"8db42d694ccc418790edff439fe026bf"
_FOOTER = 72
_METADATA = 0x0101

SIDE = 1920
FOV = 90.0
MAX_DEG = 100.0  # beyond this the lens sees its own housing
# name: (lens, yaw, pitch) relative to the lens axis; +yaw right, +pitch up.
VIEWS = {
    "l0_front": (0, 0, 0), "l0_left": (0, -50, 0), "l0_right": (0, 50, 0), "l0_up": (0, 0, 50),
    "l1_front": (1, 0, 0), "l1_left": (1, -50, 0), "l1_right": (1, 50, 0), "l1_up": (1, 0, 50),
}
REF_VIEW = "l0_front"
RIG_FILE = "rig.json"


def is_insv(path: Path) -> bool:
    """An Insta360 recording: its trailer ends with the magic string."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        if f.tell() < _FOOTER:
            return False
        f.seek(-32, 2)
        return f.read(32) == MAGIC


def _records(path: Path):
    """(id, bytes) of the trailer's records, walked back from the footer."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(size - _FOOTER)
        footer = f.read(_FOOTER)
        extra = struct.unpack_from("<I", footer, 32)[0]
        offset = _FOOTER
        while offset < extra:
            f.seek(size - offset - 6)
            rid, rsize = struct.unpack("<HI", f.read(6))
            if rsize > extra:
                return  # past the records this parser understands (thumbnails and beyond)
            f.seek(size - offset - 6 - rsize)
            yield rid, f.read(rsize)
            offset += 6 + rsize


def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while True:
        c = b[i]
        i += 1
        v |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return v, i


def _fields(b: bytes) -> dict[int, list]:
    """Top-level fields of a protobuf message, without a schema (ints and byte strings)."""
    out: dict[int, list] = {}
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(b, i)
        elif wire == 1:
            v, i = b[i:i + 8], i + 8
        elif wire == 5:
            v, i = b[i:i + 4], i + 4
        elif wire == 2:
            n, i = _varint(b, i)
            v, i = b[i:i + n], i + n
        else:
            raise ValueError(f"unexpected protobuf wire type {wire}")
        out.setdefault(field, []).append(v)
    return out


@dataclass
class Lens:
    xi: float
    fx: float
    fy: float
    cx: float  # video pixels
    cy: float
    k: tuple[float, float, float]
    p: tuple[float, float]


def read_lenses(path: Path, video_side: int) -> list[Lens]:
    """Both lenses' factory calibration, in pixels of the video stream (video_side square)."""
    meta = next((data for rid, data in _records(path) if rid == _METADATA), None)
    if meta is None:
        raise ValueError(f"{path.name}: no metadata record in the Insta360 trailer")
    fields = _fields(meta)
    if 54 not in fields:
        raise ValueError(f"{path.name}: no lens calibration (metadata field 54)")
    values = fields[54][0].decode().split("_")
    count = int(values[0])
    per_lens = 19  # xi fx fy cx cy yaw pitch roll tx ty tz k1 k2 k3 p1 p2 width height model
    crop = 5632
    if 27 in fields:  # sensor width, height, crop width, crop height
        sub = _fields(fields[27][0])
        crop = sub.get(3, [crop])[0]
    lenses = []
    for i in range(count):
        v = [float(x) for x in values[1 + i * per_lens:1 + (i + 1) * per_lens]]
        width, height = v[16], v[17]  # both lenses side by side: (2 * sensor width) x height
        sensor_w = width / count
        scale = video_side / crop
        lenses.append(Lens(
            xi=v[0], fx=v[1] * scale, fy=v[2] * scale,
            cx=(v[3] - i * sensor_w - (sensor_w - crop) / 2) * scale,
            cy=(v[4] - (height - crop) / 2) * scale,
            k=(v[11], v[12], v[13]), p=(v[14], v[15]),
        ))
    return lenses


def project(lens: Lens, rays: np.ndarray) -> np.ndarray:
    """Video pixels (N x 2) of unit rays in the lens frame (x right, y down, z forward)."""
    mx = rays[:, 0] / (rays[:, 2] + lens.xi)
    my = rays[:, 1] / (rays[:, 2] + lens.xi)
    r2 = mx * mx + my * my
    k1, k2, k3 = lens.k
    p1, p2 = lens.p
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x = mx * radial + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
    y = my * radial + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
    return np.stack([lens.fx * x + lens.cx, lens.fy * y + lens.cy], 1)


def lens_from_view(yaw: float, pitch: float) -> np.ndarray:
    y, p = np.radians(yaw), -np.radians(pitch)  # image y points down
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    return ry @ rx


# Lens 1 faces backwards: turned 180 degrees about the vertical.
LENS1_FROM_LENS0 = np.diag([-1.0, 1.0, -1.0])


def pinhole_params() -> list[float]:
    f = SIDE / 2 / np.tan(np.radians(FOV / 2))
    return [f, f, SIDE / 2, SIDE / 2]


def rig_spec() -> dict:
    """rig.json: PINHOLE params shared by every view and each view's cam_from_rig (3x4, the
    rig being REF_VIEW's camera)."""
    views = {}
    for name, (lens, yaw, pitch) in VIEWS.items():
        cam_from_lens = lens_from_view(yaw, pitch).T
        cam_from_rig = cam_from_lens @ (LENS1_FROM_LENS0 if lens == 1 else np.eye(3))
        views[name] = np.c_[cam_from_rig, np.zeros(3)].round(12).tolist()
    return {"device": "insta360_x4", "model": "PINHOLE", "width": SIDE, "height": SIDE,
            "params": pinhole_params(), "ref": REF_VIEW, "cam_from_rig": views}


def read_rig(images: Path) -> dict | None:
    """The rig spec of a video's images dir (images/c<id>), or None for ordinary videos."""
    path = images / RIG_FILE
    return json.loads(path.read_text()) if path.exists() else None


class Renderer:
    """Remap tables from each lens's fisheye into the views, built once per video."""

    def __init__(self, lenses: list[Lens]):
        f = pinhole_params()[0]
        u, v = np.meshgrid(np.arange(SIDE) + 0.5, np.arange(SIDE) + 0.5)
        rays0 = np.stack([(u.ravel() - SIDE / 2) / f, (v.ravel() - SIDE / 2) / f, np.ones(u.size)], 1)
        rays0 /= np.linalg.norm(rays0, axis=1, keepdims=True)
        self.tables = {}
        for name, (lens, yaw, pitch) in VIEWS.items():
            rays = rays0 @ lens_from_view(yaw, pitch).T
            xy = project(lenses[lens], rays).astype(np.float32) - 0.5
            xy[rays[:, 2] < np.cos(np.radians(MAX_DEG))] = -1
            self.tables[name] = (lens, xy[:, 0].reshape(SIDE, SIDE), xy[:, 1].reshape(SIDE, SIDE))

    def render(self, fisheye: list[np.ndarray]) -> dict[str, np.ndarray]:
        return {name: cv2.remap(fisheye[lens], mx, my, cv2.INTER_LINEAR, borderValue=0)
                for name, (lens, mx, my) in self.tables.items()}


def decode_views(info: VideoInfo, names: dict[int, str], out_dir: Path, ranges: list[tuple[int, int | None]],
                 candidate_fps: float, pool: Executor | None = None,
                 on_written: Callable[[int], None] | None = None) -> None:
    """ingest's decode step for an .insv: both lenses of each chosen sample at full
    resolution, rendered into the views (out_dir/<view>/<name>), plus rig.json."""
    renderer = Renderer(read_lenses(info.path, info.width))
    for view in VIEWS:
        (out_dir / view).mkdir(parents=True, exist_ok=True)
    (out_dir / RIG_FILE).write_text(json.dumps(rig_spec(), indent=1))
    lock = threading.Lock()
    written = [0]
    lens1 = VideoInfo(**(info.__dict__ | {"stream": 1}))

    def decode(first: int, last: int | None) -> None:
        select = sorted(i for i in names if i >= first and (last is None or i < last))
        if not select:
            return
        start = max(first - 1, 0)
        count = None if last is None else last - start
        both = zip(iter_frames(info, candidate_fps, None, select=select, start=start, count=count),
                   iter_frames(lens1, candidate_fps, None, select=select, start=start, count=count))
        for (index, _, front), (_, _, back) in both:
            for view, image in renderer.render([front, back]).items():
                cv2.imwrite(str(out_dir / view / names[index]), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with lock:
                written[0] += 1
                if on_written:
                    on_written(written[0])

    jobs = [lambda a=a, b=b: decode(a, b) for a, b in ranges]
    if pool is None or len(jobs) == 1:
        for job in jobs:
            job()
    else:
        for future in [pool.submit(job) for job in jobs]:
            future.result()


def view_names(prefix: str, frame_names: list[str]) -> list[str]:
    """Image names of a rig video's moments: <prefix>/<view>/<frame>, views of a moment together."""
    return [f"{prefix}/{view}/{name}" for name in frame_names for view in VIEWS]
