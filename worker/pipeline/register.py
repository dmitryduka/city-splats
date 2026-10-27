"""Register a new contribution (video) into an existing reconstruction.

Reuses the base database (features + matches of all accepted images), so only the new
images are extracted and matched:
  * sequential matching within the new video,
  * vocab-tree retrieval with the new images as queries against everything (cross-video).

Then fits the new video into the base model with existing frames held fixed:
  1. "incremental": resume the incremental mapper from the base reconstruction. Fast and
     right when the new video overlaps densely, but it stalls when the overlap is a few
     frames (it can register one frame, yet has no 3D points to register the next).
  2. "submodel" (fallback): reconstruct the new video on its own plus the few base frames
     it matched (anchors), Sim3-align that submodel onto the base through the shared
     anchors, insert the new frames, re-triangulate and bundle-adjust only the new frames.

Every candidate must pass the cross-video consistency gate (worker.pipeline.qc): a model can
be self-consistent and still put the video in the wrong place.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from worker.pipeline.features import FeatureConfig
from worker.pipeline.qc import cross_video_consistency, passes_consistency
from worker.pipeline.sfm import strengthen_weak_links
from worker.reporting import StepReporter, step

log = logging.getLogger(__name__)

_PAIR_ID_FACTOR = 2147483647


@dataclass
class RegistrationResult:
    reconstruction_path: Path
    database_path: Path
    num_new_images: int
    num_new_registered: int
    num_points_before: int
    num_points_after: int
    mean_reproj_error: float
    timings: dict[str, float]
    strategy: str = "incremental"
    gate_ok: bool = False
    gate_reason: str = ""
    consistency: dict | None = None

    @property
    def registration_ratio(self) -> float:
        return self.num_new_registered / max(self.num_new_images, 1)


def _count_new_registered(rec: pycolmap.Reconstruction, new_names: set[str]) -> int:
    return sum(1 for img in rec.images.values() if img.name in new_names and img.has_pose)


def _mean_error(rec: pycolmap.Reconstruction) -> float:
    errors = [p.error for p in rec.points3D.values()]
    return float(np.mean(errors)) if errors else float("nan")


def _anchor_images(
    database: Path, base: pycolmap.Reconstruction, new_names: set[str], min_inliers: int, neighbors: int
) -> list[str]:
    """Base images verified-matched to the new video, plus sequential neighbors for support."""
    with sqlite3.connect(database) as conn:
        names = dict(conn.execute("SELECT image_id, name FROM images"))
        pairs = conn.execute("SELECT pair_id FROM two_view_geometries WHERE rows >= ?", (min_inliers,)).fetchall()
    base_names = {img.name for img in base.images.values() if img.has_pose}
    direct: set[str] = set()
    for (pair_id,) in pairs:
        a, b = names[pair_id // _PAIR_ID_FACTOR], names[pair_id % _PAIR_ID_FACTOR]
        if a in new_names and b in base_names:
            direct.add(b)
        elif b in new_names and a in base_names:
            direct.add(a)
    # Neighbors in the same source video (names are "<contribution>/<index>.jpg").
    by_dir: dict[str, list[str]] = {}
    for name in sorted(base_names):
        by_dir.setdefault(name.split("/")[0], []).append(name)
    anchors = set(direct)
    for name in direct:
        seq = by_dir[name.split("/")[0]]
        i = seq.index(name)
        anchors.update(seq[max(0, i - neighbors) : i + neighbors + 1])
    return sorted(anchors)


def _merge_submodel(
    database: Path,
    base: pycolmap.Reconstruction,
    image_root: Path,
    new_image_names: list[str],
    out_dir: Path,
    min_anchor_inliers: int = 15,
    anchor_neighbors: int = 3,
) -> pycolmap.Reconstruction | None:
    new_names = set(new_image_names)
    anchors = _anchor_images(database, base, new_names, min_anchor_inliers, anchor_neighbors)
    log.info("submodel fallback: %d anchor frames from the base model", len(anchors))
    if len(anchors) < 2:
        return None

    options = pycolmap.GlobalPipelineOptions()
    options.image_names = new_image_names + anchors
    sub_dir = out_dir / "submodel"
    sub_dir.mkdir(parents=True, exist_ok=True)
    submodels = pycolmap.global_mapping(database, image_root, sub_dir, options)
    if not submodels:
        return None
    sub = max(submodels.values(), key=lambda r: _count_new_registered(r, new_names))

    # Same database => consistent image/frame/camera ids; anchors are the shared images.
    tgt_from_sub = pycolmap.align_reconstructions_via_reprojections(sub, base)
    if tgt_from_sub is None:
        log.warning("submodel fallback: could not align submodel to base via anchors")
        return None
    sub.transform(tgt_from_sub)

    merged = pycolmap.Reconstruction(base)
    for image in sub.images.values():
        if image.name not in new_names or not image.has_pose:
            continue
        frame = sub.frame(image.frame_id)
        if not merged.exists_camera(image.camera_id):
            merged.add_camera(sub.cameras[image.camera_id])
        if not merged.exists_rig(frame.rig_id):
            merged.add_rig(sub.rig(frame.rig_id))
        if not merged.exists_frame(frame.frame_id):
            # A fresh Frame: add_frame binds it to the target's own rig object.
            new_frame = pycolmap.Frame()
            new_frame.frame_id = frame.frame_id
            new_frame.rig_id = frame.rig_id
            new_frame.rig_from_world = frame.rig_from_world
            for data_id in frame.data_ids:
                new_frame.add_data_id(data_id)
            merged.add_frame(new_frame)
        new_image = pycolmap.Image(
            name=image.name,
            points2D=pycolmap.Point2DList([pycolmap.Point2D(p.xy) for p in image.points2D]),
            camera_id=image.camera_id,
            image_id=image.image_id,
        )
        new_image.frame_id = image.frame_id
        merged.add_image(new_image)
    # Frames were copied with their (aligned) submodel poses; mark them registered.
    for frame_id in {img.frame_id for img in sub.images.values() if img.name in new_names and img.has_pose}:
        merged.register_frame(frame_id)

    # Triangulate with all poses fixed (adds new points and extends base tracks into the
    # new video), then refine only the new frames: base frames stay constant.
    tri_dir = out_dir / "triangulated"
    tri_dir.mkdir(parents=True, exist_ok=True)
    merged = pycolmap.triangulate_points(merged, database, image_root, tri_dir, clear_points=False)

    config = pycolmap.BundleAdjustmentConfig()
    base_frames = {img.frame_id for img in base.images.values() if img.has_pose}
    for image in merged.images.values():
        if image.has_pose:
            config.add_image(image.image_id)
    for frame_id in base_frames:
        config.set_constant_rig_from_world_pose(frame_id)
    for camera_id in {img.camera_id for img in base.images.values()}:
        config.set_constant_cam_intrinsics(camera_id)
    pycolmap.create_default_bundle_adjuster(pycolmap.BundleAdjustmentOptions(), config, merged).solve()
    return merged


def video_of(name: str) -> str:
    return name.split("/")[0]


def register_contribution(
    base_database: Path,
    base_reconstruction: Path,
    image_root: Path,
    new_image_names: list[str],
    out_dir: Path,
    features: FeatureConfig,
    sequential_overlap: int = 10,
    fallback_below_ratio: float = 0.6,
    report: StepReporter | None = None,
) -> RegistrationResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    database = out_dir / "database.db"
    shutil.copyfile(base_database, database)
    timings: dict[str, float] = {}

    t = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    with step(report, "extract", len(new_image_names)):
        pycolmap.extract_features(
            database,
            image_root,
            image_names=new_image_names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options=features.extraction,
            device=features.device,
        )
    timings["extract"] = time.time() - t

    # Pairs already in the database (base images) are skipped by the matchers.
    t = time.time()
    sequential = pycolmap.SequentialPairingOptions()
    sequential.overlap = sequential_overlap
    sequential.quadratic_overlap = True
    with step(report, "match_sequential"):
        pycolmap.match_sequential(
            database, matching_options=features.sequential_matching, pairing_options=sequential,
            device=features.device,
        )
    timings["match_sequential"] = time.time() - t

    # Fast turns within the new video: re-match weak links (see sfm.strengthen_weak_links).
    t = time.time()
    if strengthen_weak_links(database, new_image_names, features, report):
        timings["match_weak_links"] = time.time() - t

    # Cross-video pairs: retrieval on a subset of new frames (sequential matches carry the
    # links to the rest), matched with the strong (LightGlue) matcher.
    t = time.time()
    query_list = out_dir / "retrieval_queries.txt"
    queries = new_image_names[:: features.retrieval_query_stride]
    query_list.write_text("\n".join(queries) + "\n")
    retrieval = pycolmap.VocabTreePairingOptions()
    retrieval.vocab_tree_path = features.vocab_tree_uri
    retrieval.match_list_path = str(query_list)
    retrieval.num_images = features.retrieval_neighbors
    with step(report, "match_retrieval", len(queries)):
        pycolmap.match_vocabtree(
            database, matching_options=features.cross_matching, pairing_options=retrieval,
            device=features.device,
        )
    timings["match_retrieval"] = time.time() - t

    base = pycolmap.Reconstruction(base_reconstruction)
    new_names = set(new_image_names)
    new_video = video_of(new_image_names[0])
    baseline = cross_video_consistency(base, base_database, video_of)

    def gate(rec: pycolmap.Reconstruction) -> tuple[bool, str, dict]:
        with step(report, "gate"):
            results = cross_video_consistency(rec, database, video_of)
        ok, reason = passes_consistency(results, new_video, baseline=baseline)
        return ok, reason, {"-".join(k): r.summary() for k, r in results.items()}

    candidates: list[tuple[str, pycolmap.Reconstruction, bool, str, dict]] = []

    t = time.time()
    options = pycolmap.IncrementalPipelineOptions()
    options.fix_existing_frames = True
    options.multiple_models = False
    sparse_dir = out_dir / "sparse"
    (sparse_dir / "incremental").mkdir(parents=True, exist_ok=True)
    with step(report, "map", base.num_reg_images() + len(new_image_names)):
        reconstructions = pycolmap.incremental_mapping(
            database, image_root, sparse_dir / "incremental", options=options, input_path=base_reconstruction
        )
    timings["map"] = time.time() - t
    if reconstructions:
        rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        if _count_new_registered(rec, new_names) >= fallback_below_ratio * len(new_image_names):
            candidates.append(("incremental", rec, *gate(rec)))
            log.info("incremental: %s", candidates[-1][3])

    if not candidates or not candidates[-1][2]:
        t = time.time()
        with step(report, "submodel_merge", len(new_image_names)):
            merged = _merge_submodel(database, base, image_root, new_image_names, out_dir / "fallback")
        timings["submodel_merge"] = time.time() - t
        if merged is not None:
            candidates.append(("submodel", merged, *gate(merged)))
            log.info("submodel: %s", candidates[-1][3])

    passing = [c for c in candidates if c[2]]
    if passing:
        strategy, rec, gate_ok, gate_reason, consistency = passing[0]
    elif candidates:
        strategy, rec, gate_ok, gate_reason, consistency = max(
            candidates, key=lambda c: _count_new_registered(c[1], new_names)
        )
    else:
        strategy, rec, gate_ok, consistency = "none", base, False, {}
        gate_reason = f"could not place {new_video} in the scene (no strategy registered enough frames)"

    rec_path = sparse_dir / "0"
    rec_path.mkdir(exist_ok=True)
    rec.write(rec_path)
    return RegistrationResult(
        reconstruction_path=rec_path,
        database_path=database,
        num_new_images=len(new_image_names),
        num_new_registered=_count_new_registered(rec, new_names),
        num_points_before=base.num_points3D(),
        num_points_after=rec.num_points3D(),
        mean_reproj_error=_mean_error(rec),
        timings=timings,
        strategy=strategy,
        gate_ok=gate_ok,
        gate_reason=gate_reason,
        consistency=consistency,
    )
