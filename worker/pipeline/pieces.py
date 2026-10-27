"""A scene as pieces: one SfM model per video, placed in the scene by a similarity transform.

Nothing is optimised over the whole scene. A new video is reconstructed on its own (its piece;
worker.prefetch runs that ahead, so it doesn't depend on HEAD), then linked:

  1. a subset of its frames (queries) is matched against a subset of the existing pieces'
     frames (candidates), copied into a small link database: vocab-tree retrieval keeps the
     best new-to-old pairs, which are matched with the strong (LightGlue) matcher;
  2. a matched keypoint that is a 3D point in both pieces gives a 3D-3D correspondence;
  3. a robust Sim3 over those places the new piece in the scene.

So linking costs depend on the new video and the candidate count, not on the scene's size
(the whole-scene bundle adjustments of incremental registration took 69 min once a scene
passed 1000 frames, when Ceres switches to an iterative solver). At city scale the candidates
come from nearby pieces only (GPS / spatial index).

For consumers that want one COLMAP model (splat training, viewer layers) the pieces are
composed: each transformed and concatenated with renumbered ids, without optimisation.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

from worker.pipeline.devices.dji import geo_offset
from worker.pipeline.features import FeatureConfig
from worker.reporting import StepReporter, step

log = logging.getLogger(__name__)


def sim3_to_json(sim: pycolmap.Sim3d) -> dict:
    return {"scale": float(sim.scale), "rotation_xyzw": [float(x) for x in sim.rotation.quat],
            "translation": [float(x) for x in sim.translation]}


def sim3_from_json(data: dict) -> pycolmap.Sim3d:
    return pycolmap.Sim3d(data["scale"], pycolmap.Rotation3d(np.array(data["rotation_xyzw"])),
                          np.array(data["translation"]))


def apply(sim: pycolmap.Sim3d, xyz: np.ndarray) -> np.ndarray:
    m = sim.matrix()
    return xyz @ m[:, :3].T + m[:, 3]


@dataclass
class Piece:
    """One video's model (or, for scenes built before pieces, the whole old model)."""

    reconstruction: Path
    database: Path
    scene_from_piece: pycolmap.Sim3d = field(default_factory=pycolmap.Sim3d)
    contribution_id: int | None = None
    gps: Path | None = None  # per-frame GPS positions (worker.prefetch writes gps.json), if any

    def to_json(self, root: Path) -> dict:
        relative = lambda path: path.resolve().relative_to(root.resolve()).as_posix()  # noqa: E731
        return {
            "reconstruction": relative(self.reconstruction),
            "database": relative(self.database),
            "scene_from_piece": sim3_to_json(self.scene_from_piece),
            "contribution_id": self.contribution_id,
            "gps": relative(self.gps) if self.gps else None,
        }

    @classmethod
    def from_json(cls, data: dict, root: Path) -> Piece:
        return cls(root / data["reconstruction"], root / data["database"],
                   sim3_from_json(data["scene_from_piece"]), data.get("contribution_id"),
                   root / data["gps"] if data.get("gps") else None)

    def positions(self) -> tuple[tuple[float, float], dict[str, np.ndarray]] | None:
        """(origin lat/lon, frame name -> metres east/north of it), or None without GPS."""
        if self.gps is None or not self.gps.exists():
            return None
        data = json.loads(self.gps.read_text())
        return tuple(data["origin"]), {name: np.array(xyz[:2]) for name, xyz in data["positions"].items()}


@dataclass
class LinkResult:
    scene_from_piece: pycolmap.Sim3d | None
    ok: bool
    reason: str
    stats: dict
    timings: dict[str, float]


def _copy_image(src: pycolmap.Database, name: str, dst: pycolmap.Database, cameras: dict) -> int:
    """Copy an image with its keypoints and descriptors (keypoint order kept, so point2D
    indices of the source model stay valid); returns its id in `dst`."""
    image = src.read_image_with_name(name)
    key = (id(src), image.camera_id)
    if key not in cameras:
        cameras[key] = dst.write_camera(src.read_camera(image.camera_id))
    new_id = dst.write_image(pycolmap.Image(name=name, camera_id=cameras[key]))
    dst.write_keypoints(new_id, src.read_keypoints(image.image_id))
    dst.write_descriptors(new_id, src.read_descriptors(image.image_id))
    return new_id


def _registered(rec: pycolmap.Reconstruction) -> list[pycolmap.Image]:
    return sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)


def link_piece(
    new: Piece,
    pieces: list[Piece],
    out_dir: Path,
    features: FeatureConfig,
    report: StepReporter | None = None,
    query_stride: int = 4,
    candidate_stride: int = 2,
    neighbors: int = 8,
    tolerance: float = 0.03,
    min_inliers: int = 100,
    min_frames: int = 10,
    min_inlier_ratio: float = 0.2,
    radius: float | None = None,
) -> LinkResult:
    """Place `new` in the scene of `pieces` (see the module docstring). With `radius` and GPS
    on both sides, a query is only paired with candidates taken within `radius` metres of it
    (horizontally), and candidates near no query aren't considered at all."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    new_rec = pycolmap.Reconstruction(new.reconstruction)
    recs = [pycolmap.Reconstruction(p.reconstruction) for p in pieces]
    by_name = [{im.name: im for im in _registered(rec)} for rec in recs]
    new_by_name = {im.name: im for im in _registered(new_rec)}

    # 1. Link database: queries from the new piece, candidates from the scene's pieces.
    t = time.time()
    link_db = out_dir / "link.db"
    link_db.unlink(missing_ok=True)
    db = pycolmap.Database.open(str(link_db))
    cameras: dict = {}
    source = pycolmap.Database.open(str(new.database))
    queries = {_copy_image(source, im.name, db, cameras): im.name for im in _registered(new_rec)[::query_stride]}
    source.close()
    # Ground positions (metres, in the new video's GPS frame) where both sides have GPS.
    where: dict[tuple[int, str], np.ndarray] = {}  # (piece index or -1 for new, name) -> xy
    new_gps = new.positions() if radius else None
    if new_gps:
        origin = new_gps[0]
        where |= {(-1, name): xy for name, xy in new_gps[1].items()}
        for k, piece in enumerate(pieces):
            if gps := piece.positions():
                offset = geo_offset(gps[0], origin)
                where |= {(k, name): xy + offset for name, xy in gps[1].items()}
    query_xy = np.array([where[(-1, name)] for name in queries.values() if (-1, name) in where])

    def near(k: int, name: str, xy: np.ndarray | None = None) -> bool:
        """Within radius of the given query position, or of any query (True without GPS)."""
        if (k, name) not in where or not len(query_xy):
            return True
        points = query_xy if xy is None else xy[None]
        return bool((np.linalg.norm(points - where[(k, name)], axis=1) <= radius).any())

    candidates: dict[int, tuple[int, str]] = {}  # link image id -> (piece index, name)
    skipped = 0
    for k, (piece, rec) in enumerate(zip(pieces, recs)):
        names = [im.name for im in _registered(rec)[::candidate_stride]]
        kept = [name for name in names if near(k, name)]
        skipped += len(names) - len(kept)
        if not kept:
            continue
        source = pycolmap.Database.open(str(piece.database))
        for name in kept:
            candidates[_copy_image(source, name, db, cameras)] = (k, name)
        source.close()
    if skipped:
        log.info("link: %d candidate frames farther than %.0f m from the new video skipped", skipped, radius)

    # 2. Retrieval: per query, the most similar candidates (other queries are skipped).
    retrieval = pycolmap.VocabTreePairingOptions()
    retrieval.vocab_tree_path = features.vocab_tree_uri
    retrieval.num_images = neighbors + 4 * neighbors
    pairs: dict[int, list[int]] = {}
    for a, b in pycolmap.VocabTreePairGenerator(retrieval, db, list(queries)).all_pairs():
        query, other = (a, b) if a in queries else (b, a)
        if query not in queries or other not in candidates:
            continue
        k, name = candidates[other]
        if (-1, queries[query]) in where and not near(k, name, where[(-1, queries[query])]):
            continue
        if len(pairs.setdefault(query, [])) < neighbors:
            pairs[query].append(other)
    names = {**queries, **{i: name for i, (_, name) in candidates.items()}}
    db.close()
    pair_file = out_dir / "pairs.txt"
    pair_file.write_text("".join(f"{names[q]} {names[c]}\n" for q, cs in pairs.items() for c in cs))
    timings["link_prepare"] = time.time() - t

    t = time.time()
    imported = pycolmap.ImportedPairingOptions()
    imported.match_list_path = str(pair_file)
    imported.block_size = 50  # COLMAP logs progress per block (default 1225 pairs)
    num_pairs = sum(len(cs) for cs in pairs.values())
    with step(report, "match_retrieval", num_pairs):
        pycolmap.match_image_pairs(link_db, matching_options=features.cross_matching, pairing_options=imported,
                                   device=features.device)
    timings["match_retrieval"] = time.time() - t

    # 3. 3D-3D correspondences through verified matches.
    t = time.time()
    db = pycolmap.Database.open(str(link_db))
    seen: set[tuple[int, int, int]] = set()
    src, tgt, depth, frame_of, observed = [], [], [], [], []
    for query, others in pairs.items():
        new_image = new_by_name[queries[query]]
        for other in others:
            if not db.exists_two_view_geometry(query, other):
                continue
            matches = db.read_two_view_geometry(query, other).inlier_matches
            k, name = candidates[other]
            old_image = by_name[k][name]
            center = apply(pieces[k].scene_from_piece, old_image.projection_center()[None])[0]
            for i, j in matches:
                p_new, p_old = new_image.points2D[i], old_image.points2D[j]
                if not (p_new.has_point3D() and p_old.has_point3D()):
                    continue
                key = (p_new.point3D_id, k, p_old.point3D_id)
                if key in seen:
                    continue
                seen.add(key)
                src.append(new_rec.points3D[p_new.point3D_id].xyz)
                old_xyz = apply(pieces[k].scene_from_piece, recs[k].points3D[p_old.point3D_id].xyz[None])[0]
                tgt.append(old_xyz)
                depth.append(np.linalg.norm(old_xyz - center))
                frame_of.append(new_image.name)
                observed.append(p_new.xy)
    db.close()
    stats: dict = {"queries": len(queries), "candidates": len(candidates), "pairs": num_pairs,
                   "correspondences": len(src), "candidates_skipped_far": skipped}
    if len(src) < 3:
        timings["align"] = time.time() - t
        return LinkResult(None, False, f"no shared 3D points with the scene ({len(src)} correspondences)",
                          stats, timings)

    # 4. Robust Sim3; the inlier threshold scales with how far the points are seen from.
    src, tgt = np.asarray(src), np.asarray(tgt)
    threshold = tolerance * float(np.median(depth))
    ransac = pycolmap.RANSACOptions()
    ransac.max_error = threshold
    estimate = pycolmap.estimate_sim3d_robust(src, tgt, ransac)
    sim = estimate["tgt_from_src"] if estimate else None
    timings["align"] = time.time() - t
    if sim is None:
        return LinkResult(None, False, f"no consistent placement from {len(src)} correspondences", stats, timings)
    inliers = np.linalg.norm(apply(sim, src) - tgt, axis=1) < threshold
    frames = len({f for f, ok in zip(frame_of, inliers) if ok})
    # Accuracy in pixels: the scene's points seen through the placed new cameras, against the
    # keypoints they were matched to.
    piece_from_scene = sim.inverse()
    errors = []
    for xyz, name, xy in zip(tgt[inliers], np.asarray(frame_of)[inliers], np.asarray(observed)[inliers]):
        image = new_by_name[name]
        in_cam = image.cam_from_world() * apply(piece_from_scene, xyz[None])[0]
        if in_cam[2] > 0:
            errors.append(np.linalg.norm(new_rec.cameras[image.camera_id].img_from_cam(in_cam[None])[0] - xy))
    reproj = float(np.median(errors)) if errors else float("nan")
    stats |= {"inliers": int(inliers.sum()), "inlier_ratio": round(float(inliers.mean()), 3),
              "linked_frames": frames, "scale": round(float(sim.scale), 4), "threshold": round(threshold, 5),
              "reproj_px_median": round(reproj, 2)}
    problems = []
    if inliers.sum() < min_inliers:
        problems.append(f"{inliers.sum()} < {min_inliers} consistent 3D matches")
    if frames < min_frames:
        problems.append(f"only {frames} frames link to the scene (< {min_frames})")
    if inliers.mean() < min_inlier_ratio:
        problems.append(f"{inliers.mean():.0%} of 3D matches agree (< {min_inlier_ratio:.0%})")
    reason = "; ".join(problems) or (f"{inliers.sum()} consistent 3D matches over {frames} frames "
                                     f"({inliers.mean():.0%} of {len(src)}), {reproj:.1f} px")
    return LinkResult(sim, not problems, reason, stats, timings)


# --- composition ----------------------------------------------------------------------------

def drop_duplicates(rec: pycolmap.Reconstruction, seen: set[str] | None = None) -> int:
    """Deregister images whose name is already in `seen` (or earlier in `rec`), adding the
    rest to `seen`: incremental mapping lets its models share up to 20 images, so models
    combined into one would hold those frames twice. Writing drops deregistered images."""
    seen = set() if seen is None else seen
    dropped = 0
    for image_id in sorted(rec.reg_image_ids()):
        image = rec.images[image_id]
        if image.name in seen:
            rec.deregister_frame(image.frame_id)
            dropped += 1
        else:
            seen.add(image.name)
    return dropped


def compose(pieces: list[Piece], out_dir: Path) -> pycolmap.Reconstruction:
    """One COLMAP model of all pieces in scene coordinates (ids renumbered), written to
    out_dir. No optimisation: a merged view for training and the viewer. A frame in several
    pieces is kept from the first."""
    lines: dict[str, list[str]] = {n: [] for n in ("cameras", "rigs", "frames", "images", "points3D")}
    offsets = {"camera": 0, "rig": 0, "frame": 0, "image": 0, "point": 0}
    seen: set[str] = set()
    with tempfile.TemporaryDirectory() as tmp:
        for k, piece in enumerate(pieces):
            rec = pycolmap.Reconstruction(piece.reconstruction)
            drop_duplicates(rec, seen)
            rec.transform(piece.scene_from_piece)
            piece_dir = Path(tmp) / str(k)
            piece_dir.mkdir()
            rec.write_text(piece_dir)
            _append_piece(piece_dir, lines, offsets)
        out_dir.mkdir(parents=True, exist_ok=True)
        text_dir = Path(tmp) / "composed"
        text_dir.mkdir()
        for name, rows in lines.items():
            (text_dir / f"{name}.txt").write_text("\n".join(rows) + "\n")
        composed = pycolmap.Reconstruction(text_dir)
    composed.write(out_dir)
    return composed


def _rows(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines() if line and not line.startswith("#")]


def _append_piece(piece_dir: Path, lines: dict[str, list[str]], offsets: dict[str, int]) -> None:
    co, ro, fo, io, po = (offsets[k] for k in ("camera", "rig", "frame", "image", "point"))
    sensor = lambda kind, sid: str(int(sid) + (co if kind == "CAMERA" else 0))  # noqa: E731
    ids = {k: 0 for k in offsets}

    for row in _rows(piece_dir / "cameras.txt"):
        ids["camera"] = max(ids["camera"], int(row[0]))
        lines["cameras"].append(" ".join([str(int(row[0]) + co), *row[1:]]))
    for row in _rows(piece_dir / "rigs.txt"):
        ids["rig"] = max(ids["rig"], int(row[0]))
        out = [str(int(row[0]) + ro), row[1], row[2], sensor(row[2], row[3])]
        rest = row[4:]
        while rest:
            kind, sid, has_pose = rest[:3]
            n = 3 + (7 if has_pose == "1" else 0)
            out += [kind, sensor(kind, sid), *rest[2:n]]
            rest = rest[n:]
        lines["rigs"].append(" ".join(out))
    for row in _rows(piece_dir / "frames.txt"):
        ids["frame"] = max(ids["frame"], int(row[0]))
        out = [str(int(row[0]) + fo), str(int(row[1]) + ro), *row[2:9], row[9]]
        data = row[10:]
        for d in range(0, len(data), 3):
            kind, sid, did = data[d:d + 3]
            out += [kind, sensor(kind, sid), str(int(did) + io)]
        lines["frames"].append(" ".join(out))
    image_lines = (piece_dir / "images.txt").read_text().splitlines()
    image_lines = [line for line in image_lines if not line.startswith("#")]
    for header, points in zip(image_lines[0::2], image_lines[1::2]):
        row = header.split()
        ids["image"] = max(ids["image"], int(row[0]))
        lines["images"].append(" ".join([str(int(row[0]) + io), *row[1:8], str(int(row[8]) + co), *row[9:]]))
        p = points.split()
        for t in range(2, len(p), 3):
            if p[t] != "-1":
                p[t] = str(int(p[t]) + po)
        lines["images"].append(" ".join(p))
    for row in _rows(piece_dir / "points3D.txt"):
        ids["point"] = max(ids["point"], int(row[0]))
        track = row[8:]
        for t in range(0, len(track), 2):
            track[t] = str(int(track[t]) + io)
        lines["points3D"].append(" ".join([str(int(row[0]) + po), *row[1:8], *track]))
    for k in offsets:
        offsets[k] += ids[k]

