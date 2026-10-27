"""Quality gates for publishing a new version.

cross_video_consistency is the geometric gate: a model can be perfectly self-consistent
(low reprojection error) and still place a video in the wrong spot. Verified feature
matches *between* videos are independent evidence: if the alignment is right, a 3D point
seen in one video must reproject onto its matched feature in the other.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

_PAIR_ID_FACTOR = 2147483647


@dataclass
class PairConsistency:
    checked: int = 0
    consistent: int = 0
    errors_px: list[float] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.consistent / self.checked if self.checked else 0.0

    def summary(self) -> dict:
        finite = [e for e in self.errors_px if np.isfinite(e)]
        return {
            "checked": self.checked,
            "consistent_ratio": round(self.ratio, 3),
            "median_error_px": round(float(np.median(finite)), 2) if finite else None,
        }


def cross_video_consistency(
    rec: pycolmap.Reconstruction,
    database: Path,
    video_of: Callable[[str], str],
    only_video: str | None = None,
    min_inliers: int = 15,
    max_error_px: float = 4.0,
) -> dict[tuple[str, str], PairConsistency]:
    """Per (video, video) pair: how many cross-video inlier matches agree with the model."""
    by_name = {img.name: img for img in rec.images.values() if img.has_pose}
    with sqlite3.connect(database) as conn:
        names = dict(conn.execute("SELECT image_id, name FROM images"))
        pairs = conn.execute(
            "SELECT pair_id FROM two_view_geometries WHERE rows >= ?", (min_inliers,)
        ).fetchall()
    db = pycolmap.Database.open(database)
    out: dict[tuple[str, str], PairConsistency] = {}
    try:
        for (pair_id,) in pairs:
            id1, id2 = pair_id // _PAIR_ID_FACTOR, pair_id % _PAIR_ID_FACTOR
            n1, n2 = names[id1], names[id2]
            v1, v2 = video_of(n1), video_of(n2)
            if v1 == v2 or (only_video and only_video not in (v1, v2)):
                continue
            if n1 not in by_name or n2 not in by_name:
                continue
            img1, img2 = by_name[n1], by_name[n2]
            stats = out.setdefault(tuple(sorted((v1, v2))), PairConsistency())
            for f1, f2 in db.read_two_view_geometry(id1, id2).inlier_matches:
                p1, p2 = img1.points2D[f1], img2.points2D[f2]
                if p1.has_point3D() and p2.has_point3D() and p1.point3D_id == p2.point3D_id:
                    stats.checked += 1
                    stats.consistent += 1
                    stats.errors_px.append(0.0)
                    continue
                # Reproject whichever side has a 3D point into the other image.
                for src, dst, dst_img in ((p1, p2, img2), (p2, p1, img1)):
                    if not src.has_point3D():
                        continue
                    cam = rec.cameras[dst_img.camera_id]
                    xyz = dst_img.cam_from_world() * rec.points3D[src.point3D_id].xyz
                    error = (
                        float(np.linalg.norm(cam.img_from_cam(xyz) - dst.xy)) if xyz[2] > 0 else float("inf")
                    )
                    stats.checked += 1
                    stats.consistent += error <= max_error_px
                    stats.errors_px.append(error)
                    break
    finally:
        db.close()
    return out


def passes_consistency(
    results: dict[tuple[str, str], PairConsistency],
    new_video: str,
    baseline: dict[tuple[str, str], PairConsistency] | None = None,
    min_checked: int = 30,
    min_ratio: float = 0.6,
    max_regression: float = 0.1,
) -> tuple[bool, str]:
    """Judge the new video on its own links (pooling would let old, dense pairs mask it),
    and make sure links between already-accepted videos didn't get worse than in HEAD."""
    new_pairs = {k: r for k, r in results.items() if new_video in k}
    checked = sum(r.checked for r in new_pairs.values())
    consistent = sum(r.consistent for r in new_pairs.values())
    if checked < min_checked:
        return False, f"too little cross-video evidence for {new_video} ({checked} matches < {min_checked})"
    ratio = consistent / checked
    if ratio < min_ratio:
        worst = min(new_pairs.items(), key=lambda kv: kv[1].ratio)
        return False, (
            f"{new_video}'s matches to other videos disagree with the model ({ratio:.0%} consistent < "
            f"{min_ratio:.0%}; worst {'-'.join(worst[0])}: {worst[1].ratio:.0%})"
        )
    for key, before in (baseline or {}).items():
        after = results.get(key)
        if after is not None and before.checked >= min_checked and after.ratio < before.ratio - max_regression:
            return False, (
                f"existing videos {'-'.join(key)} fit worse than before "
                f"({before.ratio:.0%} -> {after.ratio:.0%})"
            )
    return True, f"{ratio:.0%} of {checked} cross-video matches for {new_video} consistent"
