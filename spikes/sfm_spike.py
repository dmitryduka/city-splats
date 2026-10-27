"""M0 spike: features + sequential matching + global mapping on extracted keyframes.

usage: python -m spikes.sfm_spike <work_dir> <sift|aliked>
"""

import sys
import time
from pathlib import Path

import pycolmap


def run(work: Path, feature: str) -> None:
    images = work / "images"
    out = work / f"sfm_{feature}"
    out.mkdir(parents=True, exist_ok=True)
    db = out / "database.db"
    db.unlink(missing_ok=True)

    extraction = pycolmap.FeatureExtractionOptions()
    matching = pycolmap.FeatureMatchingOptions()
    if feature == "aliked":
        extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
        matching.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"

    t = time.time()
    pycolmap.extract_features(db, images, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)
    print(f"extract {time.time() - t:.1f}s", flush=True)

    t = time.time()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 10
    pairing.quadratic_overlap = True
    pycolmap.match_sequential(db, matching_options=matching, pairing_options=pairing)
    print(f"match {time.time() - t:.1f}s", flush=True)

    t = time.time()
    recs = pycolmap.global_mapping(db, images, out)
    print(f"map {time.time() - t:.1f}s", flush=True)
    for idx, rec in recs.items():
        print(idx, rec.summary())


if __name__ == "__main__":
    run(Path(sys.argv[1]), sys.argv[2])
