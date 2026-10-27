"""The dense cloud as an octree of tiles: a new video rebuilds only the tiles it touches, each
tile's LoD is built on its own (in parallel, memory bounded by the tile, not the scene), and
the viewer streams every tile as a separate paged LoD (Spark traverses them jointly under one
splat budget).

Tiles lie on a global grid of the scene's voxels. A top-level cell is 2**TOP_SHIFT voxels on a
side (~480 m at Achapnyak's 12 cm voxels) and splits into octants while a tile holds more than
`budget` points. A voxel never straddles tiles, so deduplicating per tile equals deduplicating
the whole cloud. Leaves never merge back, so an unchanged tile keeps its key and is shared by
reference between versions (its directory is under the contribution that last changed it).

Top-level cells with fewer than budget/4 points (mostly depth noise and far terrain around
the scene: 300 of the 333 cells of Achapnyak v4; at half the voxel, 93 of v1's 109 tiles were
cells under 1M points) are packed together, nearby cells first, into tiles of up to `budget`:
the viewer holds every tile's root chunk in a 64K-splat page of a pool of 256, so hundreds of
tiny tiles would crowd out detail. A small cell that grows is promoted to tiles of its own.

Manifest (in version.metrics["dense_tiles"]): {"voxel", "budget", "small": {top cell: tile},
"leaves": {"<top cell>/<level>_<x>_<y>_<z>": tile key}, "tiles": {key: {"dir", "points"}}};
a tile is one big leaf, a pack of small leaves of one top cell, or a pack of small cells
("small_<first cell>"). Its dir (relative to the data root) holds cloud.ply and
lod/splat-lod.rad.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from worker.pipeline.dense import read_points_ply, voxel_downsample, write_cloud_splats, write_points_ply
from worker.pipeline.lod import build_lod

TOP_SHIFT = 12
MIN_SHIFT = 6
POINT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def _pack(cells: np.ndarray) -> np.ndarray:
    """One int64 per row of (n, 3) cell indices, |index| < 2**20."""
    c = cells + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def _cells(xyz: np.ndarray, voxel: float) -> np.ndarray:
    return np.floor(xyz / voxel).astype(np.int64)


def _name(cell) -> str:
    return "_".join(str(int(v)) for v in cell)


def _group(keys: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """(key, indices in original order) for each distinct key."""
    unique, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    for key, idx in zip(unique, np.split(order, np.cumsum(np.bincount(inverse))[:-1])):
        yield int(key), idx


class Spill:
    """A video's fused points bucketed by top-level cell on disk, in arrival order."""

    def __init__(self, root: Path, voxel: float, flush_points: int = 8_000_000):
        self.root, self.voxel, self.flush_points = root, voxel, flush_points
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        self.buffers: dict[str, list[np.ndarray]] = defaultdict(list)
        self.buffered = 0

    def add(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        rows = np.empty(len(xyz), POINT)
        rows["x"], rows["y"], rows["z"] = xyz.T
        rows["red"], rows["green"], rows["blue"] = rgb.T
        tops = _cells(xyz, self.voxel) >> TOP_SHIFT
        for _, idx in _group(_pack(tops)):
            self.buffers[_name(tops[idx[0]])].append(rows[idx])
        self.buffered += len(rows)
        if self.buffered >= self.flush_points:
            self.flush()

    def flush(self) -> None:
        for name, parts in self.buffers.items():
            with open(self.root / f"{name}.bin", "ab") as f:
                for part in parts:
                    f.write(part.tobytes())
        self.buffers.clear()
        self.buffered = 0

    def tops(self) -> list[str]:
        self.flush()
        return sorted(p.stem for p in self.root.glob("*.bin"))

    def read(self, top: str) -> tuple[np.ndarray, np.ndarray]:
        rows = np.fromfile(self.root / f"{top}.bin", POINT)
        return (np.stack([rows["x"], rows["y"], rows["z"]], 1),
                np.stack([rows["red"], rows["green"], rows["blue"]], 1))


def _split(local: np.ndarray, level: int, cell: tuple[int, int, int], budget: int
           ) -> Iterator[tuple[int, tuple[int, int, int], np.ndarray]]:
    """Leaves (level, cell, point indices in order) of the octree under one node; `local` are
    the points' voxel indices within their top-level cell."""
    stack = [(level, cell, np.arange(len(local)))]
    while stack:
        level, cell, idx = stack.pop()
        if len(idx) <= budget or level == MIN_SHIFT:
            yield level, cell, idx
            continue
        child = local[idx] >> (level - 1)
        octant = ((child[:, 0] & 1) << 2) | ((child[:, 1] & 1) << 1) | (child[:, 2] & 1)
        for o in range(8):
            sub = idx[octant == o]
            if len(sub):
                stack.append((level - 1, (2 * cell[0] + (o >> 2), 2 * cell[1] + ((o >> 1) & 1),
                                          2 * cell[2] + (o & 1)), sub))


Leaf = tuple[int, int, int, int]  # (level, x, y, z) within a top-level cell


def _locate(local: np.ndarray, leaves: Iterable[Leaf]) -> np.ndarray:
    """(n, 4) leaf of each point (voxel index within its top-level cell): the leaf containing
    it, or a new leaf for the empty octant of the deepest node above it."""
    leaves = list(leaves)
    target = np.full((len(local), 4), -1, dtype=np.int64)
    pending = np.ones(len(local), bool)
    for level in sorted({leaf[0] for leaf in leaves}):
        keys = np.sort(_pack(np.array([leaf[1:] for leaf in leaves if leaf[0] == level], dtype=np.int64)))
        hit = pending & np.isin(_pack(local >> level), keys)
        target[hit] = np.c_[np.full(hit.sum(), level), local[hit] >> level]
        pending &= ~hit
    if pending.any():
        inner = {(lv + k, x >> k, y >> k, z >> k) for lv, x, y, z in leaves for k in range(1, TOP_SHIFT - lv + 1)}
        for level in range(MIN_SHIFT + 1, TOP_SHIFT + 1):
            cells = np.array([q[1:] for q in inner if q[0] == level], dtype=np.int64).reshape(-1, 3)
            if not len(cells):
                continue
            hit = pending & np.isin(_pack(local >> level), np.sort(_pack(cells)))
            target[hit] = np.c_[np.full(hit.sum(), level - 1), local[hit] >> (level - 1)]
            pending &= ~hit
    assert not pending.any(), "points outside every node of their top-level cell"
    return target


def _by_leaf(target: np.ndarray) -> Iterator[tuple[Leaf, np.ndarray]]:
    # Local cells are < 2**TOP_SHIFT = 4096, so (level, x, y, z) packs into 4 x 12 bits.
    for _, idx in _group((target[:, 0] << 36) | (target[:, 1] << 24) | (target[:, 2] << 12) | target[:, 3]):
        yield tuple(int(v) for v in target[idx[0]]), idx


def update_tiles(parent: dict | None, spill: Spill, out_dir: Path, root: Path, lod_bin: Path,
                 budget: int, jobs: int, on_progress: Callable[[int, int], None] | None = None) -> dict:
    """The parent's tiles with the spilled points merged in (existing points win per voxel).
    Touched tiles are written under out_dir and their LoDs built `jobs` at a time while the
    next cells are merged; the rest are the parent's.

    Octree leaves with at least budget/4 points are tiles of their own; smaller ones (the
    octants a split leaves nearly empty) are packed into shared tiles of up to `budget`, since
    the viewer holds every tile's root chunk in a 64K-splat page of a pool of 256."""
    voxel = spill.voxel
    tiles = dict(parent["tiles"]) if parent else {}
    leaf_tile = dict(parent["leaves"]) if parent else {}  # "<top>/<leaf>" -> tile key
    small_tops: dict[str, str] = {}  # small top cell -> its pack
    if parent:  # versions before packs had one "stray" tile for every small cell
        small_tops = dict(parent.get("small") or dict.fromkeys(parent.get("stray", []), "stray"))
    by_top: dict[str, dict[Leaf, str]] = defaultdict(dict)
    for name, tile in leaf_tile.items():
        top, leaf = name.split("/")
        by_top[top][tuple(int(v) for v in leaf.split("_"))] = tile

    pool = ThreadPoolExecutor(jobs)
    futures: list[Future] = []

    def read(key: str) -> tuple[np.ndarray, np.ndarray]:
        return read_points_ply(root / tiles[key]["dir"] / "cloud.ply")

    def local_cells(top: str, xyz: np.ndarray) -> np.ndarray:
        return _cells(xyz, voxel) - (np.array([int(v) for v in top.split("_")]) << TOP_SHIFT)

    def emit(key: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
        tile_dir = out_dir / key
        shutil.rmtree(tile_dir, ignore_errors=True)
        tile_dir.mkdir(parents=True)
        write_points_ply(tile_dir / "cloud.ply", xyz, rgb)
        write_cloud_splats(tile_dir / "splats.ply", xyz, rgb, voxel)
        tiles[key] = {"dir": str(tile_dir.relative_to(root)), "points": len(xyz)}
        futures.append(pool.submit(_build, lod_bin, tile_dir))

    def retile(top: str, xyz: np.ndarray, rgb: np.ndarray, target: np.ndarray) -> None:
        """Tiles for points of a top-level cell grouped by leaf (`target`, old points first)."""
        pieces: list[tuple[Leaf, np.ndarray, np.ndarray]] = []
        for (level, *cell), idx in _by_leaf(target):
            x, c = voxel_downsample(xyz[idx], rgb[idx], voxel)
            for lv, cl, sub in _split(local_cells(top, x), level, tuple(cell), budget):
                pieces.append(((lv, *cl), x[sub], c[sub]))
        small = []
        for leaf, x, c in pieces:
            if len(x) >= budget // 4:
                emit(f"{top}/{_name(leaf)}", x, c)
                by_top[top][leaf] = f"{top}/{_name(leaf)}"
            else:
                small.append((leaf, x, c))
        # Packs of nearby small leaves (sorted by their corner at the finest level).
        small.sort(key=lambda p: tuple(v << (p[0][0] - MIN_SHIFT) for v in p[0][1:]))
        while small:
            count, n = 0, 0
            while n < len(small) and (n == 0 or count + len(small[n][1]) <= budget):
                count += len(small[n][1])
                n += 1
            pack, small = small[:n], small[n:]
            key = f"{top}/pack_{_name(pack[0][0])}"
            emit(key, np.concatenate([p[1] for p in pack]), np.concatenate([p[2] for p in pack]))
            for leaf, _, _ in pack:
                by_top[top][leaf] = key

    def merge(top: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
        """New points of a tiled top-level cell: the tiles they fall in are re-read, merged
        and retiled; the cell's other tiles stay."""
        leaves = by_top[top]
        target = _locate(local_cells(top, xyz), leaves)
        touched = {leaves[leaf] for leaf, _ in _by_leaf(target) if leaf in leaves}
        old_leaves = [leaf for leaf, key in leaves.items() if key in touched]
        old = [read(key) for key in sorted(touched)]
        for key in touched:
            del tiles[key]
        for leaf in old_leaves:
            del leaves[leaf]
        if old:
            old_xyz, old_rgb = np.concatenate([o[0] for o in old]), np.concatenate([o[1] for o in old])
            old_target = _locate(local_cells(top, old_xyz), old_leaves)
            xyz, rgb = np.concatenate([old_xyz, xyz]), np.concatenate([old_rgb, rgb])
            target = np.concatenate([old_target, target])
        retile(top, xyz, rgb, target)

    def whole(top: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
        retile(top, xyz, rgb, np.tile([TOP_SHIFT, 0, 0, 0], (len(xyz), 1)))

    small_new: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        for top in spill.tops():
            xyz, rgb = spill.read(top)
            if by_top.get(top):
                merge(top, xyz, rgb)
            elif top in small_tops:
                small_new.append((xyz, rgb))
            else:
                xyz, rgb = voxel_downsample(xyz, rgb, voxel)
                if len(xyz) < budget // 4:
                    small_new.append((xyz, rgb))
                else:
                    whole(top, xyz, rgb)

        if small_new:
            # The packs holding any touched small cell are re-read and repacked with the new
            # points; cells grown past budget/4 get tiles of their own.
            touched = {small_tops[_name(t)] for xyz, _ in small_new
                       for t in np.unique(_cells(xyz, voxel) >> TOP_SHIFT, axis=0) if _name(t) in small_tops}
            old = [read(key) for key in sorted(touched)]
            for key in touched:
                del tiles[key]
            for top in [t for t, key in small_tops.items() if key in touched]:
                del small_tops[top]
            xyz = np.concatenate([p[0] for p in old + small_new])
            rgb = np.concatenate([p[1] for p in old + small_new])
            tops = _cells(xyz, voxel) >> TOP_SHIFT
            cells = []
            for _, idx in _group(_pack(tops)):
                top = _name(tops[idx[0]])
                x, c = voxel_downsample(xyz[idx], rgb[idx], voxel)
                if len(x) >= budget // 4:
                    whole(top, x, c)
                else:
                    cells.append((tuple(int(v) for v in tops[idx[0]]), top, x, c))
            cells.sort(key=lambda cell: cell[0])
            while cells:
                count, n = 0, 0
                while n < len(cells) and (n == 0 or count + len(cells[n][2]) <= budget):
                    count += len(cells[n][2])
                    n += 1
                pack, cells = cells[:n], cells[n:]
                key = f"small_{pack[0][1]}"
                emit(key, np.concatenate([p[2] for p in pack]), np.concatenate([p[3] for p in pack]))
                for _, top, _, _ in pack:
                    small_tops[top] = key

        for done, future in enumerate(futures, start=1):
            future.result()
            if on_progress:
                on_progress(done, len(futures))
    finally:
        pool.shutdown(cancel_futures=True)
    leaves = {f"{top}/{_name(leaf)}": key for top, cells in by_top.items() for leaf, key in cells.items()}
    return {"voxel": voxel, "budget": budget, "small": small_tops, "leaves": leaves, "tiles": tiles}


def _build(lod_bin: Path, tile_dir: Path) -> None:
    build_lod(lod_bin, tile_dir / "splats.ply", tile_dir / "lod", quality=False)
    (tile_dir / "splats.ply").unlink()


def lod_files(manifest: dict, root: Path) -> list[tuple[Path, int]]:
    """Each tile's LoD header and point count."""
    return [(root / t["dir"] / "lod" / "splat-lod.rad", t["points"]) for t in manifest["tiles"].values()]


def sample_cloud(manifest: dict, root: Path, count: int, seed: int = 0,
                 on_progress: Callable[[int, int], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """About `count` points drawn uniformly from all tiles, one tile in memory at a time."""
    total = sum(t["points"] for t in manifest["tiles"].values())
    rng = np.random.default_rng(seed)
    xyz, rgb = [], []
    for i, tile in enumerate(manifest["tiles"].values(), start=1):
        x, c = read_points_ply(root / tile["dir"] / "cloud.ply")
        keep = rng.random(len(x)) < count / max(total, 1)
        xyz.append(x[keep])
        rgb.append(c[keep])
        if on_progress:
            on_progress(i, len(manifest["tiles"]))
    return np.concatenate(xyz), np.concatenate(rgb)
