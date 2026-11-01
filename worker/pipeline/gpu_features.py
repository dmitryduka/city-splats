"""ALIKED features, brute-force matches and LightGlue matches (linking videos) on the GPU
(Apple MPS), written straight into COLMAP's database; COLMAP still does geometric (and rig)
verification and mapping.

LightGlue on the M3's GPU: ~1.3 s per pair of ~2.9k keypoints (attention takes 95% of it; fp16
is slower), vs ~3.6 s on COLMAP's CPU ONNX model at ~4.4 cores (ride link: 798 pairs, 48 min).

COLMAP runs ALIKED and its brute-force matcher as ONNX models on the CPU on a Mac (CoreML
can't compile the ALIKED graph). The same network in PyTorch (LightGlue's port of ALIKED,
same weights) on the M3's GPU gives the same features: on 6 test views the same keypoints
(within 1e-4 px) and descriptors (cosine 1.0000); on a 768-view Insta360 minute the same
verified pairs (4786 vs 4776), points (53.5k vs 53.3k) and camera path (0.03% RMS of its
length). Speed on the M3: ~115 ms per 1024 px view instead of ~500 ms; matching ~250 pairs/s
instead of ~22 (see match_pairs).

torch and pycolmap each bundle a libomp and crash in one process, so the GPU half runs as
a subprocess that only touches the database through SQLite:

  python -m worker.pipeline.gpu_features extract <database> <image_root> <names_file> <max_side> <max_keypoints> <min_score> [<mask_root>]
  python -m worker.pipeline.gpu_features match <database> <pairs_file> <min_cossim> <max_ratio> <cross_check>
  python -m worker.pipeline.gpu_features lightglue <database> <pairs_file> <min_score>

Both print "progress <done> <total>" lines.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

PAIR_ID_FACTOR = 2147483647  # COLMAP's pair id: id1 * factor + id2, id1 < id2
ALIKED_N16ROT = 1  # FeatureExtractorType in COLMAP's descriptors table


def available() -> bool:
    """On CUDA, COLMAP's own ONNX models already run on the GPU; this path is for the Mac."""
    from worker import compute

    return os.environ.get("SCENES_GPU_FEATURES", "1") != "0" and compute.device() == "mps"


def _run(args: list[str], on_progress: Callable[[int, int], None] | None) -> None:
    proc = subprocess.Popen([sys.executable, "-m", "worker.pipeline.gpu_features", *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    tail: list[str] = []
    for line in proc.stdout:
        if line.startswith("progress "):
            if on_progress:
                done, total = line.split()[1:3]
                on_progress(int(done), int(total))
            continue
        tail = (tail + [line])[-20:]
    if proc.wait() != 0:
        raise RuntimeError(f"gpu_features {args[0]} failed:\n" + "".join(tail))


def extract(database: Path, image_root: Path, names: list[str], max_side: int, max_keypoints: int,
            min_score: float, mask_root: Path | None = None,
            on_progress: Callable[[int, int], None] | None = None) -> None:
    """Keypoints and descriptors of `names`, whose images (and cameras) are already in the
    database (pycolmap.import_images); masks as COLMAP's mask_path (0 = drop the keypoint)."""
    names_file = database.parent / "gpu_names.txt"
    names_file.write_text("\n".join(names) + "\n")
    _run(["extract", str(database), str(image_root), str(names_file), str(max_side), str(max_keypoints),
          str(min_score), *([str(mask_root)] if mask_root else [])], on_progress)
    names_file.unlink()


def match(database: Path, pairs: list[tuple[str, str]], min_cossim: float, max_ratio: float, cross_check: bool,
          on_progress: Callable[[int, int], None] | None = None) -> Path:
    """Raw matches of the image pairs (by name); returns the pairs file, for verification."""
    pairs_file = database.parent / "pairs.txt"
    pairs_file.write_text("".join(f"{a} {b}\n" for a, b in pairs))
    _run(["match", str(database), str(pairs_file), str(min_cossim), str(max_ratio), str(int(cross_check))],
         on_progress)
    return pairs_file


def match_lightglue(database: Path, pairs_file: Path, min_score: float,
                    on_progress: Callable[[int, int], None] | None = None) -> None:
    """COLMAP's ALIKED LightGlue matches of the pairs in `pairs_file` ("<name> <name>" lines,
    COLMAP's imported pairing), on the GPU; COLMAP then only verifies them."""
    _run(["lightglue", str(database), str(pairs_file), str(min_score)], on_progress)


# --- the subprocess (torch) ---------------------------------------------------------------


def _progress(done: int, total: int) -> None:
    print(f"progress {done} {total}", flush=True)


def _extract_main(database: str, image_root: str, names_file: str, max_side: str, max_keypoints: str,
                  min_score: str, mask_root: str | None = None) -> None:
    import sqlite3
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import numpy as np
    import torch

    from worker.pipeline.aliked_gpu import AlikedGPU

    names = [n for n in Path(names_file).read_text().split("\n") if n]
    con = sqlite3.connect(database)
    ids = dict(con.execute("select name, image_id from images"))
    side, net = int(max_side), AlikedGPU("mps", int(max_keypoints), float(min_score))

    def load(name: str):
        """Decoding and resizing on CPU threads (cv2 releases the GIL) while the GPU works."""
        bgr = cv2.imread(str(Path(image_root) / name))
        height, width = bgr.shape[:2]
        scale = side / max(width, height)
        if scale < 1:  # COLMAP's Bitmap::Thumbnail
            bgr = cv2.resize(bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        mask = cv2.imread(str(Path(mask_root) / f"{name}.png"), cv2.IMREAD_GRAYSCALE) if mask_root else None
        return name, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (width, height), mask

    done = 0
    with torch.no_grad(), ThreadPoolExecutor(6) as pool:
        ahead = deque(pool.submit(load, name) for name in names[:16])  # bounded read-ahead
        for k in range(len(names)):
            name, rgb, (width, height), mask = ahead.popleft().result()
            if k + 16 < len(names):
                ahead.append(pool.submit(load, names[k + 16]))
            kp, desc = net(rgb)
            sx, sy = width / rgb.shape[1], height / rgb.shape[0]
            kp *= np.array([sx, sy], np.float32)  # FeatureKeypoint::Rescale
            if mask is not None:  # MaskFeatures: drop keypoints on 0
                keep = mask[kp[:, 1].astype(int).clip(0, mask.shape[0] - 1),
                            kp[:, 0].astype(int).clip(0, mask.shape[1] - 1)] > 0
                kp, desc = kp[keep], desc[keep]
            rows = np.zeros((len(kp), 6), np.float32)  # x, y and the (rescaled) identity shape
            rows[:, :2], rows[:, 2], rows[:, 5] = kp, sx, sy
            con.execute("insert or replace into keypoints values (?, ?, 6, ?)", (ids[name], len(kp), rows.tobytes()))
            con.execute("insert or replace into descriptors values (?, ?, ?, ?, ?)",
                        (ids[name], ALIKED_N16ROT, len(desc), desc.shape[1] * 4, np.ascontiguousarray(desc).tobytes()))
            done += 1
            if done % 25 == 0 or done == len(names):
                con.commit()
                _progress(done, len(names))
    con.commit()
    con.close()


def _match_main(database: str, pairs_file: str, min_cossim: str, max_ratio: str, cross_check: str) -> None:
    import sqlite3

    import numpy as np
    import torch

    con = sqlite3.connect(database)
    ids = dict(con.execute("select name, image_id from images"))
    pairs = [line.split(" ") for line in Path(pairs_file).read_text().split("\n") if line]
    stats = match_pairs(con, ids, pairs, float(min_cossim), float(max_ratio), cross_check == "1", torch, np)
    con.close()
    print(f"matched {len(pairs)} pairs: {stats}", flush=True)


def match_pairs(con, ids: dict[str, int], pairs: list[list[str]], min_cossim: float, max_ratio: float,
                cross_check: bool, torch, np, group: int = 16) -> dict:
    """COLMAP's bruteforce-matcher.onnx on the GPU: cosine similarity >= min_cossim, Lowe's
    ratio on the distance sqrt(1 - cos) below max_ratio, mutual nearest neighbours.

    The similarity matrices are computed in fp16 (on the M3: 260 vs 150 pairs/s in fp32,
    memory bound), and each match's best similarity is then recomputed in fp32 for the
    thresholds, so fp16 only matters for near-ties. Pairs are grouped by their first image
    (up to `group` partners per matrix product); descriptors are kept on the GPU until the
    last pair that uses them."""
    dev = "mps"
    # Images with fewer than 2 features (a rig's up view of open sky, the sky masked: 21 of a
    # ride minute's 1448 views) match nothing; their pairs get empty match lists.
    total = len(pairs)
    counts = dict(con.execute("select image_id, rows from descriptors"))
    empty = [(a, b) for a, b in pairs if min(counts.get(ids[a], 0), counts.get(ids[b], 0)) < 2]
    for a, b in empty:
        id1, id2 = sorted((ids[a], ids[b]))
        con.execute("insert or replace into matches values (?, 0, 2, ?)", (id1 * PAIR_ID_FACTOR + id2, b""))
    pairs = [(a, b) for a, b in pairs if min(counts.get(ids[a], 0), counts.get(ids[b], 0)) >= 2]
    by_first: dict[str, list[str]] = {}
    for a, b in pairs:
        by_first.setdefault(a, []).append(b)
    order = list(by_first)
    last_use: dict[str, int] = {}
    for k, a in enumerate(order):
        for name in [a, *by_first[a]]:
            last_use[name] = k
    cache: dict[str, torch.Tensor] = {}

    def desc(name: str) -> torch.Tensor:
        if name not in cache:
            rows, cols, data = con.execute("select rows, cols, data from descriptors where image_id = ?",
                                           (ids[name],)).fetchone()
            cache[name] = torch.from_numpy(np.frombuffer(data, np.float32).reshape(rows, cols // 4).copy()).to(dev)
        return cache[name]

    done, total_matches = len(empty), 0
    pending: list[tuple[str, list[str], torch.Tensor, torch.Tensor]] = []

    def flush() -> None:
        nonlocal total_matches
        for a, chunk, ok, j in pending:
            ok, j = ok.cpu().numpy(), j.cpu().numpy()
            for k, b in enumerate(chunk):
                rows = np.nonzero(ok[k])[0]
                m = np.stack([rows, j[k][rows]], 1).astype(np.uint32)
                id1, id2 = ids[a], ids[b]
                if id1 > id2:
                    id1, id2, m = id2, id1, m[:, ::-1]
                con.execute("insert or replace into matches values (?, ?, 2, ?)",
                            (id1 * PAIR_ID_FACTOR + id2, len(m), np.ascontiguousarray(m).tobytes()))
                total_matches += len(m)
        pending.clear()
        con.commit()

    with torch.no_grad():
        for k, a in enumerate(order):
            A = desc(a)
            A16 = A.half()
            rows_a = torch.arange(len(A), device=dev)[None]
            partners = by_first[a]
            for g in range(0, len(partners), group):
                chunk = partners[g:g + group]
                Bs = [desc(b) for b in chunk]
                B = torch.zeros(len(chunk), max(len(x) for x in Bs), A.shape[1], device=dev, dtype=torch.float16)
                for i, Bi in enumerate(Bs):
                    B[i, :len(Bi)] = Bi  # padding scores 0, below any min_cossim
                S = torch.matmul(A16[None], B.transpose(1, 2))  # k x n1 x n2
                best_row = S.argmax(dim=1)  # best row of each column
                j = S.max(dim=2).indices
                second = S.scatter_(2, j[..., None], -2.0).max(dim=2).values.float()
                Bj = torch.stack([torch.cat([Bi, Bi.new_zeros(B.shape[1] - len(Bi), Bi.shape[1])]) for Bi in Bs])
                best = (A[None] * Bj.gather(1, j[..., None].expand(-1, -1, A.shape[1]))).sum(-1)  # fp32
                ok = best >= min_cossim
                if max_ratio > 0:
                    ok &= torch.sqrt((1 - best).clamp(min=0) / (1 - second + 1e-8)) < max_ratio
                if cross_check:
                    ok &= best_row.gather(1, j) == rows_a
                pending.append((a, chunk, ok, j))
                done += len(chunk)
            if len(pending) >= 32:
                flush()
                _progress(done, total)
            for name in [a, *by_first[a]]:
                if last_use[name] <= k:
                    cache.pop(name, None)
    flush()
    _progress(done, total)
    return {"pairs": total, "empty": len(empty), "matches_mean": round(total_matches / max(len(pairs), 1), 1)}


def _cross_sdpa(self, x0, x1, mask=None):
    """lightglue's CrossBlock.forward with its bidirectional attention as two fused attention
    calls (the same numbers within 3e-6): off CUDA it builds both N0 x N1 score matrices and
    two softmaxes, 77-99 ms per block for ~2.9k keypoints on the M3, vs 45-50 ms this way."""
    import torch
    import torch.nn.functional as F

    assert mask is None
    qk0, qk1 = self.to_qk(x0), self.to_qk(x1)
    v0, v1 = self.to_v(x0), self.to_v(x1)
    qk0, qk1, v0, v1 = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2).contiguous() for t in (qk0, qk1, v0, v1))
    m0 = F.scaled_dot_product_attention(qk0, qk1, v1)  # scale: (qk0 * s^0.5) . (qk1 * s^0.5)
    m1 = F.scaled_dot_product_attention(qk1, qk0, v0)
    m0, m1 = (self.to_out(t.transpose(1, 2).flatten(start_dim=-2)) for t in (m0, m1))
    return x0 + self.ffn(torch.cat([x0, m0], -1)), x1 + self.ffn(torch.cat([x1, m1], -1))


def _lightglue_main(database: str, pairs_file: str, min_score: str) -> None:
    """aliked-lightglue.onnx as COLMAP runs it (onnx_matchers.cc): keypoints shifted to
    LightGlue's convention (top-left pixel centre at 0), the image size given, every layer run
    (the ONNX export has no early stopping or point pruning), matches scoring >= min_score."""
    import sqlite3

    import numpy as np
    import torch
    from lightglue import LightGlue
    from lightglue.lightglue import CrossBlock

    if os.environ.get("SCENES_LIGHTGLUE_CROSS_SDPA", "1") != "0":
        CrossBlock.forward = _cross_sdpa
    con = sqlite3.connect(database)
    ids = dict(con.execute("select name, image_id from images"))
    sizes = {image_id: (w, h) for image_id, w, h in con.execute(
        "select image_id, width, height from images join cameras using (camera_id)")}
    pairs = [line.split(" ") for line in Path(pairs_file).read_text().split("\n") if line]
    net = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1,
                    filter_threshold=float(min_score)).eval().to("mps")
    cache: dict[str, dict] = {}

    def features(name: str) -> dict:
        if name not in cache:
            image_id = ids[name]
            rows, data = con.execute("select rows, data from keypoints where image_id = ?", (image_id,)).fetchone()
            kp = np.frombuffer(data, np.float32).reshape(rows, -1)[:, :2] - 0.5
            rows, cols, data = con.execute("select rows, cols, data from descriptors where image_id = ?",
                                           (image_id,)).fetchone()
            desc = np.frombuffer(data, np.float32).reshape(rows, cols // 4)
            cache[name] = {"keypoints": torch.from_numpy(kp.copy())[None].to("mps"),
                           "descriptors": torch.from_numpy(desc.copy())[None].to("mps"),
                           "image_size": torch.tensor([sizes[image_id]], dtype=torch.float32, device="mps")}
            if len(cache) > 64:  # pairs come grouped by query; keep the recent ones
                cache.pop(next(iter(cache)))
        return cache[name]

    total = 0
    with torch.no_grad():
        for done, (a, b) in enumerate(pairs, start=1):
            fa, fb = features(a), features(b)
            m = np.zeros((0, 2), np.uint32)
            if fa["keypoints"].shape[1] and fb["keypoints"].shape[1]:
                out = net({"image0": fa, "image1": fb})
                m = out["matches"][0].cpu().numpy().astype(np.uint32)
            id1, id2 = ids[a], ids[b]
            if id1 > id2:
                id1, id2, m = id2, id1, m[:, ::-1]
            con.execute("insert or replace into matches values (?, ?, 2, ?)",
                        (id1 * PAIR_ID_FACTOR + id2, len(m), np.ascontiguousarray(m).tobytes()))
            total += len(m)
            if done % 50 == 0 or done == len(pairs):
                con.commit()
                _progress(done, len(pairs))
    con.close()
    print(f"matched {len(pairs)} pairs: {round(total / max(len(pairs), 1), 1)} matches each", flush=True)


if __name__ == "__main__":
    sys.modules["pycolmap"] = None  # lightglue imports it optionally; its libomp clashes with torch's
    {"extract": _extract_main, "match": _match_main, "lightglue": _lightglue_main}[sys.argv[1]](*sys.argv[2:])
