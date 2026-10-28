"""Sky masks for splat cleanup, run as a subprocess (torch and pycolmap each bundle an OpenMP
runtime and can't share a process).

  python -m worker.pipeline.sky_net <image_root> <names_file> <out_dir> [device]

Writes <out_dir>/<name>.png: 255 where the pixel is sky, at MASK_WIDTH pixels wide; existing
masks are kept, so frames shared by blocks and versions are segmented once.
Model: SegFormer-B2 trained on ADE20K (class 2 is sky). On Achapnyak frames it marked sky
cleanly, leaving buildings and the city on the horizon (1.6 s/frame on the M3's CPU at 640 px).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
SKY_CLASS = 2
MASK_WIDTH = 640
# ImageNet normalisation, as SegFormer's image processor does (which needs torchvision).
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def main() -> None:
    image_root, names_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    device = sys.argv[4] if len(sys.argv) > 4 else ("cuda" if torch.cuda.is_available() else
                                                      "mps" if torch.backends.mps.is_available() else "cpu")
    names = [n for n in names_file.read_text().split("\n") if n]
    todo = [n for n in names if not (out_dir / f"{n}.png").exists()]
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(device).eval()
    started = time.time()
    done = len(names) - len(todo)
    for name in todo:
        image = Image.open(image_root / name).convert("RGB")
        width, height = MASK_WIDTH, round(image.height * MASK_WIDTH / image.width)
        pixels = (np.asarray(image.resize((width, height), Image.BILINEAR)) / 255 - MEAN) / STD
        x = torch.from_numpy(pixels.transpose(2, 0, 1)[None]).float().to(device)
        with torch.no_grad():
            logits = torch.nn.functional.interpolate(model(pixel_values=x).logits, size=(height, width),
                                                     mode="bilinear")
        sky = (logits.argmax(1)[0] == SKY_CLASS).cpu().numpy()
        out = out_dir / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((sky * 255).astype(np.uint8)).save(out)
        done += 1
        # Progress line parsed by the caller.
        print(f"sky {done}/{len(names)}", flush=True)
    print(f"done {len(todo)} frames in {time.time() - started:.1f}s on {device}", flush=True)


if __name__ == "__main__":
    main()
