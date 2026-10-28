"""Splat training in ground-plane blocks, so a block's quality depends on its own frames and
steps, not on how big the scene has grown. One global run spreads a fixed number of steps
over every frame, so each area got less optimisation with every video added.

Blocks are leaves of a quadtree over the map frame (metres east/north from the scene's GPS
origin, worker.pipeline.georef), on a fixed grid, split while more than `budget` frames look
into a block (a frame looks where the median of its observed SfM points lies); small leaves are
merged into neighbours. Every block trains the full steps: held-out PSNR was still rising by
0.6-1.3 dB per 500-750 steps when Achapnyak v5's blocks stopped at 1750-2450 steps scaled to
their frames (Brush grows splats and decays the learning rate over the run's total steps). The grid is
fixed and leaves split only where frames were added, so a block whose frames didn't change can
be reused by the next version.

Each block trains on the frames looking into it and into a margin around it, from the SfM
points there, and keeps only the splats closest to its own cells (empty space goes to the
nearest block). Seams are therefore cut where each side was trained with context beyond it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from worker.pipeline.cleanup import read_ply, write_ply

ROOT_HALF_M = 8192.0


Rect = tuple[float, float, float, float]  # min east, min north, max east, max north (metres)


@dataclass
class Block:
    key: str  # quadtree cell "<level>_<x>_<y>" (level 0 = root), or "g..." for grouped cells
    cells: list[Rect]  # the ground it owns
    grow: float  # margin around the cells (metres) for training frames and points
    core: list[str]  # frames looking into its cells
    images: list[str]  # training frames: core plus the nearest frames looking into the margin

    @property
    def images_hash(self) -> str:
        return hashlib.sha1("\n".join(sorted(self.images)).encode()).hexdigest()[:16]


def map_from_scene(geo: dict) -> np.ndarray:
    return np.linalg.inv(np.array(geo["scene_from_enu"]))


def _to_map(xyz: np.ndarray, enu_from_scene: np.ndarray) -> np.ndarray:
    return xyz @ enu_from_scene[:3, :3].T + enu_from_scene[:3, 3]


def plan_blocks(rec: pycolmap.Reconstruction, geo: dict, budget: int, min_side_m: float = 25.0,
                margin: float = 0.25, min_margin_m: float = 30.0, max_margin_m: float = 100.0,
                max_margin_share: float = 0.5) -> list[Block]:
    """Quadtree leaves holding at most `budget` frames (or `min_side_m` wide). A parent's
    leaves with under budget/4 frames each are grouped into one block (a split leaves some
    quadrants nearly empty). A block trains on its own frames plus the frames looking within
    a margin of its cells (`margin` of its frames' look extent, clamped to min/max_margin_m),
    nearest first and at most max_margin_share of its own: Brush picks frames uniformly, and
    a block whose margin pulled in the dense centre of the scene (2741 frames for 482 of its
    own) would spend most of its steps elsewhere, the dilution blocks are meant to avoid."""
    enu_from_scene = map_from_scene(geo)
    ids = np.array(list(rec.points3D.keys()))
    pos = dict(zip(ids.tolist(), _to_map(np.array([rec.points3D[i].xyz for i in ids]), enu_from_scene)[:, :2]))
    names, looks = [], []
    for image in rec.images.values():
        if not image.has_pose:
            continue
        seen = [pos[p.point3D_id] for p in image.points2D if p.has_point3D() and p.point3D_id in pos]
        if len(seen) >= 20:
            names.append(image.name)
            looks.append(np.median(seen, axis=0))
    names, looks = np.array(names), np.array(looks)

    leaves: list[tuple[int, int, int, np.ndarray]] = []
    stack = [(0, 0, 0, np.arange(len(names)))]
    while stack:
        level, x, y, idx = stack.pop()
        side = 2 * ROOT_HALF_M / 2**level
        if len(idx) <= budget or side / 2 < min_side_m:
            leaves.append((level, x, y, idx))
            continue
        x0, y0 = -ROOT_HALF_M + x * side, -ROOT_HALF_M + y * side
        right = looks[idx, 0] >= x0 + side / 2
        top = looks[idx, 1] >= y0 + side / 2
        for dx in (0, 1):
            for dy in (0, 1):
                sub = idx[(right == bool(dx)) & (top == bool(dy))]
                if len(sub):
                    stack.append((level + 1, 2 * x + dx, 2 * y + dy, sub))

    def cell(level: int, x: int, y: int) -> Rect:
        side = 2 * ROOT_HALF_M / 2**level
        return (-ROOT_HALF_M + x * side, -ROOT_HALF_M + y * side,
                -ROOT_HALF_M + (x + 1) * side, -ROOT_HALF_M + (y + 1) * side)

    groups: dict[str, list[tuple[int, int, int, np.ndarray]]] = {}
    for leaf in sorted(leaves, key=lambda leaf: leaf[:3]):
        level, x, y, idx = leaf
        small = len(idx) < budget // 4 and level > 0
        key = f"g{level - 1}_{x // 2}_{y // 2}" if small else f"{level}_{x}_{y}"
        groups.setdefault(key, []).append(leaf)

    # Every block trains the full steps (a short run stops growing splats and anneals early,
    # whatever its frames), so a 40-frame block costs as much as a full one: blocks under half
    # the budget join their nearest block while it stays under 1.5x the budget.
    def size(key: str) -> int:
        return sum(len(m[3]) for m in groups[key])

    def gap(a: str, b: str) -> float:
        return min(_rect_gap(cell(*m[:3]), cell(*n[:3])) for m in groups[a] for n in groups[b])

    while True:
        small = sorted((size(k), k) for k in groups if size(k) < budget // 2)
        merges = [(gap(k, other), k, other) for n, k in small for other in groups
                  if other != k and size(other) + n <= 1.5 * budget]
        if not merges:
            break
        _, key, into = min(merges)
        groups[into] += groups.pop(key)

    blocks = []
    for key, members in groups.items():
        idx = np.concatenate([m[3] for m in members])
        cells = [cell(*m[:3]) for m in members]
        extent = float((looks[idx].max(0) - looks[idx].min(0)).max())
        grow = min(max(margin * extent, min_margin_m), max_margin_m)
        distance = np.min([_rect_distance(looks, c) for c in cells], axis=0)
        near = np.flatnonzero((distance > 0) & (distance <= grow))
        near = near[np.argsort(distance[near], kind="stable")][:int(max_margin_share * len(idx))]
        blocks.append(Block(key, cells, grow, sorted(names[idx].tolist()),
                            sorted(names[np.concatenate([idx, near])].tolist())))
    return blocks


def block_plan(blocks: list[Block]) -> list[dict]:
    """Blocks as stored in Version.metrics["block_plan"]: cells in the map frame (metres
    east/north, the frame of geo["scene_from_map"]), in training order."""
    return [{"key": b.key, "cells": [[round(c, 2) for c in cell] for cell in b.cells], "grow": round(b.grow, 1),
             "frames": len(b.core)} for b in blocks]


def block_model(rec: pycolmap.Reconstruction, geo: dict, block: Block) -> pycolmap.Reconstruction:
    """The block's part of a model: its training frames, and the SfM points within its
    margin (the initialisation, and what floaters are carved against)."""
    sub = pycolmap.Reconstruction(rec)
    keep = set(block.images)
    for image in list(sub.images.values()):
        if image.has_pose and image.name not in keep:
            sub.deregister_frame(image.frame_id)
    ids = np.array(list(sub.points3D.keys()))
    xy = _to_map(np.array([sub.points3D[i].xyz for i in ids]), map_from_scene(geo))[:, :2]
    distance = np.min([_rect_distance(xy, c) for c in block.cells], axis=0)
    for pid in ids[distance > block.grow]:
        sub.delete_point3D(int(pid))
    return sub


def block_init(cloud: tuple[np.ndarray, np.ndarray], model: pycolmap.Reconstruction, geo: dict, block: Block,
               count: int, out_ply: Path, seed: int = 0) -> dict:
    """Brush's starting points for a block: up to `count` dense-cloud points within its
    margin, plus the block model's SfM points (Brush takes init.ply instead of the SfM
    points, and the dense cloud has nothing beyond depth range, e.g. far hills). Brush sizes
    them from their nearest neighbours."""
    xyz, rgb = cloud
    xy = _to_map(xyz, map_from_scene(geo))[:, :2]
    near = np.flatnonzero(np.min([_rect_distance(xy, c) for c in block.cells], axis=0) <= block.grow)
    near = np.random.default_rng(seed).permutation(near)[:count]
    sfm = list(model.points3D.values())
    points = np.concatenate([xyz[near], np.array([p.xyz for p in sfm]).reshape(-1, 3)]).astype(np.float32)
    colors = np.concatenate([rgb[near], np.array([p.color for p in sfm]).reshape(-1, 3)]).astype(np.uint8)
    rows = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rows["x"], rows["y"], rows["z"] = points.T
    rows["red"], rows["green"], rows["blue"] = colors.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(rows)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(out_ply, "wb") as f:
        f.write(header.encode())
        f.write(rows.tobytes())
    return {"dense": int(len(near)), "sfm": len(sfm)}


def crop_to_block(splat_ply: Path, out_ply: Path, geo: dict, block: Block, blocks: list[Block]) -> int:
    """Keep the splats nearest to this block's cells (inside them, or in empty space closer
    to them than to any other block's). Returns how many."""
    header, props, rows = read_ply(splat_ply)
    xy = _to_map(rows[:, [props.index("x"), props.index("y"), props.index("z")]], map_from_scene(geo))[:, :2]
    distance = np.stack([np.min([_rect_distance(xy, c) for c in b.cells], axis=0) for b in blocks], axis=1)
    mine = np.argmin(distance, axis=1) == next(i for i, b in enumerate(blocks) if b.key == block.key)
    write_ply(out_ply, header, rows[mine])
    return int(mine.sum())


def _rect_gap(a: Rect, b: Rect) -> float:
    return float(np.hypot(max(a[0] - b[2], b[0] - a[2], 0.0), max(a[1] - b[3], b[1] - a[3], 0.0)))


def _rect_distance(xy: np.ndarray, rect: Rect) -> np.ndarray:
    dx = np.maximum(np.maximum(rect[0] - xy[:, 0], xy[:, 0] - rect[2]), 0)
    dy = np.maximum(np.maximum(rect[1] - xy[:, 1], xy[:, 1] - rect[3]), 0)
    return np.hypot(dx, dy)
