"""Does ALIKED+LightGlue link room video 3 to videos 1/2 better than SIFT did?

SIFT baseline (same frames): 12 verified cross pairs, 283 inliers, 7/85 c3 frames linked.
usage: python -m spikes.aliked_cross
"""

import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import pycolmap

ROOT = Path("data/scenes/room")
IMAGES = ROOT / "images"
OUT = Path("data/bench/aliked_cross")
ALIKED_VOCAB_TREE_URI = (
    "https://github.com/colmap/colmap/releases/download/3.13.0/"
    "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin;"
    "vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin;"
    "8b2f9bdc44ca7204d8543bb3adab4c03ba9336c84ef41220b5007991036f075e"
)


def names_of(c: str) -> list[str]:
    return [f"{c}/{f['name']}" for f in json.load(open(ROOT / f"contributions/{c}/frames.json"))]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "database.db"
    db.unlink(missing_ok=True)
    c1, c2, c3 = names_of("c1"), names_of("c2"), names_of("c3")

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
    matching = pycolmap.FeatureMatchingOptions()
    matching.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"

    t = time.time()
    for names in (c1, c2, c3):  # one camera per video
        pycolmap.extract_features(db, IMAGES, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extraction)
    print(f"extract {time.time() - t:.0f}s", flush=True)

    t = time.time()
    queries = OUT / "queries.txt"
    queries.write_text("\n".join(c3) + "\n")
    retrieval = pycolmap.VocabTreePairingOptions()
    retrieval.vocab_tree_path = ALIKED_VOCAB_TREE_URI
    retrieval.match_list_path = str(queries)
    retrieval.num_images = 30
    pycolmap.match_vocabtree(db, matching_options=matching, pairing_options=retrieval)
    print(f"retrieval+lightglue {time.time() - t:.0f}s", flush=True)

    with sqlite3.connect(db) as conn:
        names = dict(conn.execute("SELECT image_id, name FROM images"))
        rows = conn.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows >= 15").fetchall()
    cross, linked, to_segment = [], defaultdict(int), 0
    for pair_id, n in rows:
        a, b = names[pair_id // 2147483647], names[pair_id % 2147483647]
        if a.startswith("c3/") == b.startswith("c3/"):
            continue
        new, old = (a, b) if a.startswith("c3/") else (b, a)
        cross.append(n)
        linked[new] = max(linked[new], n)
        to_segment += old.startswith("c2/") and int(old[3:8]) <= 55
    print(json.dumps({
        "cross_pairs": len(cross),
        "cross_inliers": sum(cross),
        "c3_frames_linked": f"{len(linked)}/{len(c3)}",
        "pairs_into_c2_0_55": to_segment,
        "median_inliers": sorted(cross)[len(cross) // 2] if cross else 0,
    }))


if __name__ == "__main__":
    main()
