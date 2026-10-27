"""Benchmark SfM feature configs on the room video-1 keyframes (121 frames).

usage: python -m spikes.bench_sfm <config> [<config> ...]
configs: sift_default, sift_fast, aliked_coreml, aliked_cpu
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap

IMAGES = Path("data/spike/images")
NAMES = [f["name"] for f in json.load(open("data/spike/frames.json"))]


def configure(name: str):
    extraction = pycolmap.FeatureExtractionOptions()
    matching = pycolmap.FeatureMatchingOptions()
    device = pycolmap.Device.auto
    if name == "sift_fast":
        extraction.sift.first_octave = 0
        extraction.sift.max_num_features = 4096
    elif name.startswith("aliked"):
        extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
        matching.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
        if name == "aliked_coreml":
            # pycolmap's Device has no "coreml": Device.cuda just sets use_gpu=True, which
            # this macOS build routes to ONNX Runtime's CoreML execution provider.
            device = pycolmap.Device.cuda
    return extraction, matching, device


def run(name: str) -> dict:
    out = Path(f"data/bench/{name}")
    out.mkdir(parents=True, exist_ok=True)
    db = out / "database.db"
    db.unlink(missing_ok=True)
    extraction, matching, device = configure(name)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    timings = {}

    t = time.time()
    pycolmap.extract_features(db, IMAGES, image_names=NAMES, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction, device=device)
    timings["extract"] = time.time() - t

    t = time.time()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 10
    pairing.quadratic_overlap = True
    pycolmap.match_sequential(db, matching_options=matching, pairing_options=pairing, device=device)
    timings["match"] = time.time() - t

    t = time.time()
    recs = pycolmap.global_mapping(db, IMAGES, out)
    timings["map"] = time.time() - t
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    errors = [p.error for p in rec.points3D.values()]
    return {
        "config": name,
        **{k: round(v, 1) for k, v in timings.items()},
        "total": round(sum(timings.values()), 1),
        "registered": f"{rec.num_reg_images()}/{len(NAMES)}",
        "points": rec.num_points3D(),
        "track_len": round(rec.compute_mean_track_length(), 2),
        "reproj": round(float(np.mean(errors)), 3),
    }


if __name__ == "__main__":
    for config in sys.argv[1:]:
        print(json.dumps(run(config)), flush=True)
