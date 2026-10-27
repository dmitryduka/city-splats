"""Micro-benchmark ALIKED extraction/matching knobs on CPU (20 frames, 40 sequential pairs).

usage: python -m spikes.bench_aliked
"""

import json
import sqlite3
import time
from pathlib import Path

import pycolmap

IMAGES = Path("data/scenes/room/images")
NAMES = [f"c3/{i:05d}.jpg" for i in range(20)]
OUT = Path("data/bench/aliked_knobs")


def run(max_size: int, max_features: int, matcher: pycolmap.FeatureMatcherType) -> dict:
    tag = f"{max_size}_{max_features}_{matcher.name}"
    db = OUT / f"{tag}.db"
    db.unlink(missing_ok=True)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
    extraction.max_image_size = max_size
    extraction.aliked.max_num_features = max_features
    matching = pycolmap.FeatureMatchingOptions()
    matching.type = matcher
    t = time.time()
    pycolmap.extract_features(db, IMAGES, image_names=NAMES, camera_mode=pycolmap.CameraMode.SINGLE,
                              extraction_options=extraction)
    extract = time.time() - t
    pairs = OUT / f"{tag}_pairs.txt"
    pairs.write_text("".join(f"{NAMES[i]} {NAMES[j]}\n" for i in range(20) for j in (i + 1, i + 3) if j < 20))
    options = pycolmap.ImportedPairingOptions()
    options.match_list_path = pairs
    t = time.time()
    pycolmap.match_image_pairs(db, matching_options=matching, pairing_options=options)
    match = time.time() - t
    with sqlite3.connect(db) as conn:
        inliers = [r for (r,) in conn.execute("SELECT rows FROM two_view_geometries")]
    n_pairs = len(pairs.read_text().splitlines())
    return {
        "config": tag,
        "extract_s_per_img": round(extract / len(NAMES), 2),
        "match_s_per_pair": round(match / n_pairs, 3),
        "median_inliers": sorted(inliers)[len(inliers) // 2] if inliers else 0,
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    LG, BF = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE, pycolmap.FeatureMatcherType.ALIKED_BRUTEFORCE
    for config in [(1600, 2048, LG), (1600, 2048, BF), (1024, 2048, LG), (1024, 1024, LG), (1024, 1024, BF)]:
        print(json.dumps(run(*config)), flush=True)
