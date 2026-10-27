"""Feed-forward reconstruction on the Mac: time, memory and pose accuracy vs COLMAP.

Runs in tools/ff3d/.venv (torch with MPS; VGGT, VGG-T3, MapAnything installed there):
  tools/ff3d/.venv/bin/python spikes/ff3d_bench.py <vggt|vggttt|mapanything|mapanything-apache> <num_frames> [fp32|fp16|bf16]

Frames are an evenly spaced subset of room video 1 (room-aliked c5). Poses are compared with
COLMAP's after a similarity alignment of camera centres (rotation error and centre error as a
fraction of the trajectory extent).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/scenes/room-aliked"
RESULTS = ROOT / "data/bench/ff3d.jsonl"


def colmap_poses(names: list[str]) -> dict[str, np.ndarray]:
    """cam_from_world 3x4 per image name, read from COLMAP's text export (no pycolmap here)."""
    text = ROOT / "data/bench/ff3d_c5_images.txt"
    poses = {}
    for line in text.read_text().splitlines():
        parts = line.split()
        if len(parts) != 10 or line.startswith("#"):
            continue  # comments and POINTS2D lines
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        name = parts[9]
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        poses[name] = np.hstack([R, [[tx], [ty], [tz]]])
    return {n: poses[n] for n in names if n in poses}


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d.T @ s / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / s.var(0).sum()
    return scale, R, mu_d - scale * R @ mu_s


def pose_errors(pred_c2w: np.ndarray, names: list[str]) -> dict:
    gt = colmap_poses(names)
    keep = [i for i, n in enumerate(names) if n in gt]
    gt_c2w = []
    for i in keep:
        P = gt[names[i]]
        R, t = P[:, :3], P[:, 3]
        gt_c2w.append(np.hstack([R.T, (-R.T @ t)[:, None]]))
    gt_c2w = np.array(gt_c2w)
    pred = pred_c2w[keep]
    s, R, t = umeyama(pred[:, :3, 3], gt_c2w[:, :3, 3])
    centres = (s * (R @ pred[:, :3, 3].T)).T + t
    extent = np.ptp(gt_c2w[:, :3, 3], axis=0).max()
    centre_err = np.linalg.norm(centres - gt_c2w[:, :3, 3], axis=1) / extent
    rot_err = []
    for a, b in zip(pred[:, :3, :3], gt_c2w[:, :3, :3]):
        rel = (R @ a).T @ b
        rot_err.append(np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))))
    return {
        "compared": len(keep),
        "centre_err_median_pct": round(float(np.median(centre_err)) * 100, 2),
        "centre_err_max_pct": round(float(centre_err.max()) * 100, 2),
        "rot_err_median_deg": round(float(np.median(rot_err)), 2),
        "rot_err_max_deg": round(float(np.max(rot_err)), 2),
    }


def main() -> None:
    model_name, n = sys.argv[1], int(sys.argv[2])
    precision = sys.argv[3] if len(sys.argv) > 3 else "fp16"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    device = os.environ.get("FF3D_DEVICE", "mps")  # "cpu" to cross-check MPS numerics
    mps = device == "mps"
    frames = json.loads((SCENE / "contributions/c5/frames.json").read_text())
    all_names = [f"c5/{f['name']}" for f in frames]
    idx = np.linspace(0, len(all_names) - 1, n).round().astype(int)
    names = [all_names[i] for i in idx]
    paths = [str(SCENE / "images" / name) for name in names]

    if model_name == "vggt":
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        model = VGGT.from_pretrained("facebook/VGGT-1B")
    elif model_name == "vggttt":
        # The TTT layers call .cuda() directly; send those tensors to MPS instead.
        torch.Tensor.cuda = lambda self, *a, **k: self.to(device)
        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
        torch.cuda.empty_cache = lambda: None
        torch.cuda.mem_get_info = lambda *a, **k: (8 * 2**30, 24 * 2**30)  # -> 6-frame head chunks
        from vggttt.nets.vggt.img import load_and_preprocess_images
        from vggttt.nets.vggt.models.vggt import VGGT

        import vggttt.nets.vggt.heads.utils as head_utils

        def sincos(embed_dim, pos, omega_0=100, dtype=torch.float32):  # float64 is unsupported on MPS
            omega = 1.0 / omega_0 ** (torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
                                      / (embed_dim / 2.0))
            out = torch.einsum("m,d->md", pos.reshape(-1).float(), omega)
            return torch.cat([torch.sin(out).to(dtype), torch.cos(out).to(dtype)], dim=1)

        head_utils.make_sincos_pos_embed = sincos
        model = VGGT.from_pretrained("nvidia/vgg-ttt")
    elif model_name in ("mapanything", "mapanything-apache"):
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images

        # linalg.solve on bf16/fp16 MPS tensors fails (intrinsics recovery); solve tiny systems on CPU.
        solve = torch.linalg.solve
        torch.linalg.solve = lambda A, B, **k: solve(A.float().cpu(), B.float().cpu(), **k).to(A.device, A.dtype)
        repo = "facebook/map-anything-apache" if model_name.endswith("apache") else "facebook/map-anything"
        model = MapAnything.from_pretrained(repo)
    else:
        raise SystemExit(__doc__)
    model = model.eval().to(device)
    if model_name.startswith("mapanything"):
        views = load_images(paths)
        images = views[0]["img"]
    else:
        images = load_and_preprocess_images(paths).to(device)
    if mps:
        torch.mps.empty_cache()

    peak = [0]
    sampling = [True]

    def sample() -> None:
        while sampling[0] and mps:
            peak[0] = max(peak[0], torch.mps.driver_allocated_memory())
            time.sleep(0.1)

    import threading
    threading.Thread(target=sample, daemon=True).start()
    started = time.time()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=precision != "fp32"):
        if model_name == "vggt":
            preds = model(images)
            extri, _ = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
            w2c = extri[0].float().cpu().numpy()
            c2w = np.array([np.vstack([np.hstack([R.T, (-R.T @ t)[:, None]]), [0, 0, 0, 1]])
                            for R, t in ((m[:, :3], m[:, 3]) for m in w2c)])
            points = preds["world_points"][0]
            conf = preds["world_points_conf"][0]
        elif model_name == "vggttt":
            # Settings of the authors' eval config (vggttt/evaluation/config/model/vggttt.yaml).
            preds = model.infer(images, dtype=dtype, num_ttt_steps=int(os.environ.get("FF3D_TTT_STEPS", 2)),
                                memory_efficient_inference=True, use_global_pred=True)
            c2w = preds["pose"].float().cpu().numpy()
            points, conf = preds["pts3d"], preds["conf"]
        else:
            amp = {"fp32": "fp32", "fp16": "fp16", "bf16": "bf16"}[precision]
            preds = model.infer(views, memory_efficient_inference=True, use_amp=precision != "fp32",
                                amp_dtype=amp, apply_mask=True, mask_edges=True)
            c2w = torch.cat([p["camera_poses"] for p in preds]).float().cpu().numpy()
            points = torch.cat([p["pts3d"] for p in preds])
            conf = torch.cat([p["conf"] * p["mask"][..., 0] for p in preds])
        if mps:
            torch.mps.synchronize()
    seconds = time.time() - started
    sampling[0] = False

    conf = conf.float().cpu().numpy()
    row = {
        "model": model_name,
        "device": device,
        "frames": n,
        "precision": precision,
        "input": list(images.shape[-2:]),
        "seconds": round(seconds, 1),
        "peak_gb": round(max(peak[0], torch.mps.driver_allocated_memory() if mps else 0) / 2**30, 1),
        "points_conf_gt_median": int((conf > np.median(conf)).sum()),
        **pose_errors(c2w, names),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))
    np.savez_compressed(ROOT / f"data/bench/ff3d_{model_name}_{n}.npz", c2w=c2w,
                        points=points.float().cpu().numpy().astype(np.float16), conf=conf.astype(np.float16))


if __name__ == "__main__":
    main()
