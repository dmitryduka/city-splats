"""Monocular depth for densification, run as a subprocess (torch and pycolmap each bundle an
OpenMP runtime and can't share a process).

  python -m worker.pipeline.depth_net <image_root> <names_file> <out_dir> <model> [device]

Writes <out_dir>/<name>.npy: relative inverse depth (affine-invariant disparity, float16) at
the network's output resolution; worker.pipeline.dense aligns it to the SfM points.
Model: Depth Anything V2 (Hugging Face id; Small and Base are Apache-2.0).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def main() -> None:
    image_root, names_file, out_dir, model_id = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    device = sys.argv[5] if len(sys.argv) > 5 else ("cuda" if torch.cuda.is_available() else
                                                      "mps" if torch.backends.mps.is_available() else "cpu")
    names = names_file.read_text().split("\n")
    names = [n for n in names if n]
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    started = time.time()
    for k, name in enumerate(names):
        image = Image.open(image_root / name).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            disparity = model(**inputs).predicted_depth[0]
        out = out_dir / f"{name}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, disparity.float().cpu().numpy().astype(np.float16))
        # Progress line parsed by the caller.
        print(f"depth {k + 1}/{len(names)}", flush=True)
    print(f"done {len(names)} frames in {time.time() - started:.1f}s on {device}", flush=True)


if __name__ == "__main__":
    main()
