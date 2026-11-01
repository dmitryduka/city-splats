"""Sky masks for splat cleanup, and optionally masks of movable things (vehicles, people) from
the same pass, run as a subprocess (torch and pycolmap each bundle an OpenMP runtime and
can't share a process).

  python -m worker.pipeline.sky_net <image_root> <names_file> <out_dir> [device] [--movable <dir>]

Writes <out_dir>/<name>.png: 255 where the pixel is sky, at MASK_WIDTH pixels wide (and
<movable dir>/<name>.png: 255 on MOVABLE classes); existing masks are kept, so frames shared
by blocks and versions are segmented once.
Model: SegFormer-B2 trained on ADE20K (class 2 is sky). On Achapnyak frames it marked sky
cleanly, leaving buildings and the city on the horizon (1.6 s/frame on the M3's CPU at 640 px).
"""

from __future__ import annotations

import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"
SKY_CLASS = 2
# ADE20K: person, car, bus, truck, van, minibike, animal, bicycle.
MOVABLE_CLASSES = [12, 20, 80, 83, 102, 116, 126, 127]
MASK_WIDTH = 640
# ImageNet normalisation, as SegFormer's image processor does (which needs torchvision).
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def _ahead(pool: ThreadPoolExecutor, fn, items: list, depth: int):
    """pool.map with at most `depth` results in flight (map submits everything at once)."""
    futures = deque(pool.submit(fn, item) for item in items[:depth])
    for item in items[depth:]:
        yield futures.popleft().result()
        futures.append(pool.submit(fn, item))
    while futures:
        yield futures.popleft().result()


def main() -> None:
    args = sys.argv[1:]
    movable_dir = None
    if "--movable" in args:
        i = args.index("--movable")
        movable_dir = Path(args[i + 1])
        args = args[:i] + args[i + 2:]
    image_root, names_file, out_dir = Path(args[0]), Path(args[1]), Path(args[2])
    device = args[3] if len(args) > 3 else ("cuda" if torch.cuda.is_available() else
                                            "mps" if torch.backends.mps.is_available() else "cpu")
    names = [n for n in names_file.read_text().split("\n") if n]
    todo = [n for n in names if not (out_dir / f"{n}.png").exists()
            or (movable_dir is not None and not (movable_dir / f"{n}.png").exists())]
    # fp16 on the GPU: 183 vs 275 ms per 640 px frame on the M3, with the same movable masks
    # (IoU 0.9995 vs fp32 on 48 ride views).
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(device, dtype).eval()
    mean, std = MEAN.astype(np.float32), STD.astype(np.float32)

    def load(name: str) -> tuple[str, np.ndarray]:
        image = Image.open(image_root / name).convert("RGB")
        width, height = MASK_WIDTH, round(image.height * MASK_WIDTH / image.width)
        return name, (np.asarray(image.resize((width, height), Image.BILINEAR), np.float32) / 255 - mean) / std

    def save(name: str, masks: dict[Path, np.ndarray]) -> None:
        for root, mask in masks.items():
            out = root / f"{name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(out)

    started = time.time()
    done = len(names) - len(todo)
    movable = torch.tensor(MOVABLE_CLASSES, device=device)
    # Images are decoded and masks written on threads while the GPU segments.
    with ThreadPoolExecutor(4) as loader, ThreadPoolExecutor(2) as writer:
        for name, pixels in _ahead(loader, load, todo, 16):
            height, width = pixels.shape[:2]
            x = torch.from_numpy(pixels.transpose(2, 0, 1)[None].copy()).to(device, dtype)
            with torch.no_grad():
                logits = torch.nn.functional.interpolate(model(pixel_values=x).logits, size=(height, width),
                                                         mode="bilinear")
            labels = logits.argmax(1)[0]
            masks = {out_dir: labels == SKY_CLASS}
            if movable_dir is not None:
                masks[movable_dir] = torch.isin(labels, movable)
            writer.submit(save, name, {root: mask.cpu().numpy() for root, mask in masks.items()})
            done += 1
            # Progress line parsed by the caller.
            print(f"sky {done}/{len(names)}", flush=True)
    print(f"done {len(todo)} frames in {time.time() - started:.1f}s on {device}", flush=True)


if __name__ == "__main__":
    main()
