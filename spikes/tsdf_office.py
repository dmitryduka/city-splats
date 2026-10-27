"""TSDF fusion of the aligned monocular depth maps (vs per-pixel point fusion in
worker.pipeline.dense). Averaging many frames in a signed-distance volume should cancel the
~5% per-frame depth noise into thinner surfaces, and gives a mesh too.

  python -m spikes.tsdf_office [voxel] [trunc_factor]

Uses the office scene's HEAD reconstruction and all its keyframes; writes
data/bench/tsdf_office/{cloud.ply, mesh.ply}.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pycolmap
from PIL import Image

from worker.pipeline.dense import _align, predict_depth, write_points_ply

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/scenes/office"
OUT = ROOT / "data/bench/tsdf_office"


def main() -> None:
    voxel = float(sys.argv[1]) if len(sys.argv) > 1 else 0.015
    trunc = voxel * (float(sys.argv[2]) if len(sys.argv) > 2 else 4.0)
    contributions = ["c8", "c9", "c10", "c11"]
    names = []
    for c in contributions:
        frames = json.loads((SCENE / "contributions" / c / "frames.json").read_text())
        names += [f"{c}/{f['name']}" for f in frames]
    rec = pycolmap.Reconstruction(SCENE / "contributions/c11/sfm/sparse/0")
    depth_dir = OUT / "depth"
    if not all((depth_dir / f"{n}.npy").exists() for n in names):
        t = time.time()
        predict_depth(SCENE / "images", names, depth_dir)
        print(f"depth for {len(names)} frames: {time.time() - t:.0f}s")

    # Tensor VoxelBlockGrid: the legacy ScalableTSDFVolume extracts nothing in the Open3D
    # 0.20 macOS arm64 wheel (even from 268 frames).
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=16, block_count=200_000,
        device=o3c.Device("CPU:0"))
    by_name = {im.name: im for im in rec.images.values() if im.has_pose}
    used, started = 0, time.time()
    for name in names:
        image = by_name.get(name)
        if image is None:
            continue
        camera = rec.cameras[image.camera_id]
        frame = _align(image, camera, rec, np.load(depth_dir / f"{name}.npy"))
        if frame is None or frame.fit_error > 0.2:
            continue
        h, w = frame.depth.shape
        sx, sy = frame.scale
        depth = np.nan_to_num(frame.depth, nan=0.0).astype(np.float32)
        color = np.asarray(Image.open(SCENE / "images" / name).convert("RGB").resize((w, h)))
        fx, fy, cx, cy = camera.params[:4]
        K = o3c.Tensor(np.array([[fx / sx, 0, cx / sx], [0, fy / sy, cy / sy], [0, 0, 1]]), o3c.float64)
        E = o3c.Tensor(np.vstack([image.cam_from_world().matrix(), [0, 0, 0, 1]]), o3c.float64)
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth))
        color_t = o3d.t.geometry.Image(o3c.Tensor(color.astype(np.float32) / 255.0))
        blocks = grid.compute_unique_block_coordinates(depth_t, K, E, 1.0, 8.0, trunc_voxel_multiplier=trunc / voxel)
        grid.integrate(blocks, depth_t, color_t, K, E, 1.0, 8.0, trunc_voxel_multiplier=trunc / voxel)
        used += 1
    print(f"integrated {used}/{len(names)} frames in {time.time() - started:.0f}s (voxel {voxel}, trunc {trunc:.3f})")

    t = time.time()
    # Surfaces seen by at least 3 frames.
    cloud = grid.extract_point_cloud(weight_threshold=3.0)
    xyz = cloud.point.positions.numpy().astype(np.float32)
    rgb = (np.clip(cloud.point.colors.numpy(), 0, 1) * 255).round().astype(np.uint8)
    write_points_ply(OUT / "cloud.ply", xyz, rgb)
    mesh = grid.extract_triangle_mesh(weight_threshold=3.0).to_legacy()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(OUT / "mesh.ply"), mesh)
    print(f"cloud {len(xyz):,} points, mesh {len(mesh.triangles):,} triangles ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
