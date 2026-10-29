"""Jobs: process one contribution (ingest -> SfM/register -> layers -> dense cloud ->
publish) and train one version's splat (train -> remove floaters -> LoD)."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pycolmap
from sqlmodel import func, select

from backend.app.config import (
    AUTO_TRAIN,
    BUILD_LOD_BIN,
    DATA_ROOT,
    DECODE_THREADS,
    DEPTH_MODEL,
    DENSE_STRIDE,
    DENSE_TILE_POINTS,
    LINK_CANDIDATE_STRIDE,
    LINK_NEIGHBORS,
    LINK_QUERY_STRIDE,
    LINK_RADIUS_M,
    LOD_JOBS,
    MIN_REGISTRATION_RATIO,
    SCENES_ROOT,
    TRAIN_BLOCK_FRAMES,
    TRAIN_DENSE_INIT,
    TRAIN_EXTRA_FRAMES,
    TRAIN_FRAME_SIDE,
    TRAIN_PREVIEW_EVERY,
    TRAIN_SHARED_CAMERAS,
    TRAIN_SKY_EVERY,
    TRAIN_SETTINGS,
    TRAIN_STALE_FRAMES,
    TRAIN_STALE_MARGIN,
    UNIFORM_INIT_POINTS,
    UNIFORM_PIXELS,
    files_url,
    scene_setting,
)
from backend.app.db import session_scope
from backend.app.models import Contribution, Scene, Version, utcnow
from worker import compute
from worker.pipeline.cleanup import clean_sparse, clean_splats, points_as_splats
from worker.pipeline.dense import (
    DenseStats, align_frames, fuse_frames, predict_depth,
)
from worker.pipeline.export import export_layers
from worker.pipeline.georef import georeference
from worker.pipeline.features import DEFAULT_FEATURES, feature_config
from worker.pipeline.ingest import probe
from worker.pipeline.lod import build_lod
from worker.pipeline.sky import remove_sky
from worker.pipeline.pieces import Piece, apply, compose, link_piece
from worker.pipeline.splat import Snapshot, TrainingStopped
from worker.pipeline.splat.blocks import Block, block_init, block_model, block_plan, crop_to_block, plan_blocks
from worker.pipeline.splat.frames import (
    FrameStats,
    Video,
    add_extra_frames,
    candidates,
    copy_poses,
    decode_images,
    plan_images,
    shared_cameras,
)
from worker.pipeline.splat.brush import eval_results
from worker.pipeline.tiles import Spill, lod_files, sample_cloud, update_tiles
from worker.pipeline.uniform import uniform_init
from worker.prefetch import Prefetcher
from worker.progress import Progress, TimingModel, contribution_sink, rel, training_sink

log = logging.getLogger(__name__)

# Typical keyframes per second of video from ingest's motion-based spacing; only used to
# pre-estimate SfM time before the frames exist.
EXPECTED_KEYFRAME_RATE = 3.0


def _update(contribution_id: int, **fields) -> None:
    with session_scope() as session:
        c = session.get(Contribution, contribution_id)
        for key, value in fields.items():
            setattr(c, key, value)
        c.updated_at = utcnow()
        session.add(c)
        session.commit()


def publish(
    contribution_id: int,
    scene_id: int,
    head: Version | None,
    database: Path,
    reconstruction: Path,
    work: Path,
    dense_rad: Path,
    metrics: dict,
) -> int:
    """Create a Version with its dense cloud, move HEAD, mark the contribution done and
    queue automatic splat training if AUTO_TRAIN is set. Its splat starts as the trained
    ancestor's blocks (inherit_blocks)."""
    with session_scope() as session:
        scene = session.get(Scene, scene_id)
        number = (
            session.exec(select(func.max(Version.number)).where(Version.scene_id == scene_id)).one() or 0
        ) + 1
        version = Version(
            scene_id=scene_id,
            number=number,
            parent_id=head.id if head else None,
            contribution_id=contribution_id,
            contribution_ids=(head.contribution_ids if head else []) + [contribution_id],
            database_path=rel(database),
            reconstruction_path=rel(reconstruction),
            splat_path="",
            splat_url="",
            dense_url=files_url(dense_rad),
            layers_url=files_url(work / "layers"),
            metrics=metrics,
        )
        if AUTO_TRAIN in TRAIN_SETTINGS:
            version.train_status, version.train_quality, version.train_auto = "queued", AUTO_TRAIN, True
        session.add(version)
        session.flush()
        scene.head_version_id = version.id
        scene.updated_at = utcnow()
        contribution = session.get(Contribution, contribution_id)
        contribution.status = "done"
        contribution.version_id = version.id
        contribution.progress, contribution.stage_progress, contribution.eta_seconds = 1.0, 1.0, 0.0
        contribution.stage = "published"
        contribution.metrics = metrics
        contribution.preview_url = contribution.preview_step = None
        contribution.finished_at = contribution.updated_at = utcnow()
        session.add_all([scene, contribution])
        session.commit()
        version_id = version.id
    try:
        inherit_blocks(version_id)
    except Exception:  # noqa: BLE001 - the version is complete without it; training plans again
        log.exception("version %s: could not take over the earlier version's blocks", version_id)
    return number


def process(contribution_id: int, prefetch: Prefetcher) -> None:
    with session_scope() as session:
        c = session.get(Contribution, contribution_id)
        scene = session.get(Scene, c.scene_id)
        head = session.get(Version, scene.head_version_id) if scene.head_version_id else None
        seq_by_dir = {
            f"c{other.id}": other.seq
            for other in session.exec(select(Contribution).where(Contribution.scene_id == scene.id)).all()
        }

    scene_dir = SCENES_ROOT / scene.slug
    image_root = scene_dir / "images"
    work = scene_dir / "contributions" / c.image_dir_name
    # Frames and depth maps may already be there (worker.prefetch); anything else is left from
    # an interrupted run.
    for stale in (work / name for name in ("sfm", "link", "scene", "layers", "dense", "preview")):
        shutil.rmtree(stale, ignore_errors=True)

    device = compute.device()
    timing = TimingModel(device)
    video = Path(c.video_path)
    info = probe(video)
    est_frames = info.duration * EXPECTED_KEYFRAME_RATE
    stages = [("ingest", timing.estimate("ingest", info.duration)), ("sfm", timing.estimate("sfm", est_frames))]
    if head is not None:
        stages.append(("register", timing.estimate("register", est_frames)))
    stages += [("layers", timing.estimate("layers", est_frames)), ("dense", timing.estimate("dense", est_frames))]
    progress = Progress(contribution_sink(c.id), stages, timing)
    metrics: dict = {"device": device, "video_seconds": round(info.duration, 1)}
    try:
        # 1. Ingest (usually run ahead by the prefetcher)
        frames, seconds = _await(prefetch.ingest(c, scene.slug, retry=True), progress, "ingest")
        timing.record("ingest", info.duration, seconds)
        names = [f"{c.image_dir_name}/{f['name']}" for f in frames]
        metrics |= {"keyframes": len(names), "holdouts": sum(f["holdout"] for f in frames),
                    "ingest_seconds": seconds}
        for stage in ("sfm", "register", "layers"):
            if any(name == stage for name, _ in stages):
                progress.set_estimate(stage, timing.estimate(stage, len(names)))
        _update(c.id, metrics=metrics)

        # 2. The video's own model, its piece of the scene (usually built ahead). A scene keeps
        # the feature type it was built with; scenes predating the "features" metric used SIFT.
        features = feature_config(head.metrics.get("features", "sift") if head else DEFAULT_FEATURES)
        metrics["features"] = features.name
        sfm, seconds = _await(prefetch.piece(c, scene.slug, names, retry=True), progress, "sfm")
        timing.record("sfm", len(names), seconds)
        ratio = sfm.registration_ratio
        metrics |= {
            "registered": sfm.num_registered, "camera_model": sfm.camera_model,
            "registration_ratio": round(ratio, 3), "reproj_error_px": round(sfm.mean_reproj_error, 3),
            "sfm_seconds": seconds, "sfm_timings": {k: round(v, 1) for k, v in sfm.timings.items()},
        }
        if sfm.gps:
            metrics["gps"] = sfm.gps
        if sfm.chunking:
            metrics["chunking"] = sfm.chunking
        rejection = None
        if sfm.defect:
            rejection = f"reconstruction failed after retries: {sfm.defect}"
        elif ratio < MIN_REGISTRATION_RATIO:
            rejection = (f"only {ratio:.0%} of frames reconstructed (< {MIN_REGISTRATION_RATIO:.0%}); "
                         "the video is likely too blurry or textureless")

        # 3. Place the piece in the scene: matched against a subset of the existing pieces'
        # frames, aligned by a similarity transform; nothing else moves (worker.pipeline.pieces).
        gps = sfm.database_path.parent / "gps.json"
        piece = Piece(sfm.reconstruction_path, sfm.database_path, contribution_id=c.id,
                      gps=gps if gps.exists() else None)
        pieces = scene_pieces(head) if head is not None else []
        if head is not None and rejection is None:
            pairs = -(-sfm.num_registered // LINK_QUERY_STRIDE) * LINK_NEIGHBORS
            progress.start("register", substeps=[("match_retrieval", pairs)], rate_suffix=features.name)
            with prefetch.paused():
                link = link_piece(piece, pieces, work / "link", features, report=progress,
                                  query_stride=LINK_QUERY_STRIDE, candidate_stride=LINK_CANDIDATE_STRIDE,
                                  neighbors=LINK_NEIGHBORS, radius=LINK_RADIUS_M)
            timing.record("register", len(names), progress.finish())
            metrics |= {"strategy": "pieces", "gate": link.reason, "link": link.stats,
                        "sfm_timings": metrics["sfm_timings"] | {k: round(v, 1) for k, v in link.timings.items()}}
            if link.ok:
                piece.scene_from_piece = link.scene_from_piece
            else:
                rejection = f"could not place the video in the scene: {link.reason}"
        if rejection:
            _update(c.id, status="rejected", metrics=metrics, error=rejection, eta_seconds=0, finished_at=utcnow())
            return
        pieces.append(piece)
        _update(c.id, metrics=metrics)

        # 4. The pieces as one model (training and viewer layers); viewer layers (sparse points
        # by source video + camera centers)
        progress.start("layers")
        rec = compose(pieces, work / "scene")
        metrics |= {"points": rec.num_points3D(), "points_before": head.metrics.get("points") if head else None,
                    "pieces": [p.to_json(DATA_ROOT) for p in pieces]}
        export_layers(rec, work / "layers", lambda name: seq_by_dir.get(name.split("/")[0], 1) - 1)
        if geo := georeference(rec, [p.gps for p in pieces if p.gps]):
            metrics["geo"] = geo
        timing.record("layers", len(names), progress.finish())

        # 5. Dense point cloud (the scene's main view): this video's frames densified and
        # merged into the parent version's cloud. If densification fails, the SfM points
        # (as round splats) stand in so the version is still viewable.
        progress.start("dense")
        try:
            depth_job, relay = prefetch.depth(c, scene.slug, names, retry=True)
            if not depth_job.done():
                relay.attach(lambda f, detail: progress.update(0.6 * f, detail=detail))
            depth_dir = depth_job.result()
            with prefetch.paused():
                dense_rad = _densify(head, pycolmap.Reconstruction(piece.reconstruction), image_root, names,
                                     work, metrics, progress, depth_dir, scene_from_piece=piece.scene_from_piece,
                                     settings=scene.settings)
            shutil.rmtree(depth_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001 - the version is still useful without a dense cloud
            log.exception("densification failed for contribution %s; using the sparse points", c.id)
            clean_sparse(rec)
            preview_dir = work / "preview"
            preview_dir.mkdir(exist_ok=True)
            points_as_splats(rec, preview_dir / "splat.ply")
            dense_rad = build_lod(BUILD_LOD_BIN, preview_dir / "splat.ply", preview_dir / "lod", quality=False)
            metrics["dense_failed"] = True
            # The next video still merges into the cloud the scene had before this one.
            if head and "dense_tiles" in head.metrics:
                metrics |= {k: head.metrics[k] for k in ("dense_tiles", "dense_voxel")}
        timing.record("dense", len(names), progress.finish())

        # 6. Publish a new version and move HEAD
        number = publish(c.id, scene.id, head, piece.database, work / "scene", work, dense_rad, metrics)
        log.info("contribution %s published as version %s", c.id, number)
    finally:
        progress.close()


def scene_pieces(version: Version) -> list[Piece]:
    """The version's pieces; a version from before pieces is one piece (its whole model)."""
    if "pieces" in version.metrics:
        return [Piece.from_json(p, DATA_ROOT) for p in version.metrics["pieces"]]
    database = DATA_ROOT / version.database_path
    gps = database.parent / "gps.json"
    return [Piece(DATA_ROOT / version.reconstruction_path, database, gps=gps if gps.exists() else None)]


def _await(job: tuple, progress: Progress, stage: str):
    """Result of a prefetch job: a stage done ahead, or one whose progress we report while
    waiting for it."""
    future, relay = job
    if future.done() and future.exception() is None:
        result, seconds = future.result()
        progress.done_ahead(stage, seconds)
        return result, seconds
    progress.start(stage)
    relay.attach(lambda f, detail: progress.update(f, detail=detail))
    result, seconds = future.result()
    progress.finish()
    return result, seconds


def _densify(head: Version | None, rec: pycolmap.Reconstruction, image_root: Path, names: list[str],
             work: Path, metrics: dict, progress: Progress, depth_dir: Path | None = None,
             scene_from_piece: pycolmap.Sim3d | None = None, settings: dict | None = None) -> Path:
    """Dense cloud for this contribution, merged into the parent's tiles; returns the tile
    list the viewer loads. The scene keeps the first video's voxel size. Depth maps are
    predicted unless `depth_dir` already holds them. `rec` is the video's piece, placed by
    scene_from_piece. `settings`: the scene's overrides (scene_setting)."""
    stride = scene_setting(settings, "DENSE_STRIDE", DENSE_STRIDE)
    dense_dir = work / "dense"
    dense_dir.mkdir(exist_ok=True)
    if depth_dir is None:
        depth_dir = dense_dir / "depth"
        predict_depth(image_root, names, depth_dir, scene_setting(settings, "DEPTH_MODEL", DEPTH_MODEL),
                      on_progress=lambda done, total: progress.update(
                          0.6 * done / total, detail=f"Predicting depth {done}/{total}"))
    frames, footprint = align_frames(rec, names, depth_dir, stride=stride)
    parent = head.metrics.get("dense_tiles") if head else None
    voxel = parent["voxel"] if parent else footprint * (scene_from_piece.scale if scene_from_piece else 1.0)
    spill = Spill(dense_dir / "spill", voxel)

    def sink(xyz: np.ndarray, rgb: np.ndarray) -> None:
        spill.add(apply(scene_from_piece, xyz).astype(np.float32) if scene_from_piece else xyz, rgb)

    fused = fuse_frames(frames, image_root, sink, stride=stride,
                        on_progress=lambda done, total: progress.update(
                            0.6 + 0.2 * done / total, detail=f"Fusing depth maps {done}/{total}"))
    shutil.rmtree(dense_dir / "depth", ignore_errors=True)
    manifest = update_tiles(parent, spill, dense_dir / "tiles", DATA_ROOT, BUILD_LOD_BIN,
                            scene_setting(settings, "DENSE_TILE_POINTS", DENSE_TILE_POINTS),
                            scene_setting(settings, "LOD_JOBS", LOD_JOBS),
                            on_progress=lambda done, total: progress.update(
                                0.8 + 0.2 * done / total, detail=f"Building tile LoDs {done}/{total}"))
    shutil.rmtree(dense_dir / "spill", ignore_errors=True)
    before = sum(t["points"] for t in parent["tiles"].values()) if parent else 0
    total = sum(t["points"] for t in manifest["tiles"].values())
    stats = DenseStats(len(names), len(frames), total - before, voxel,
                       float(np.median([f.fit_error for f in frames.values()])))
    metrics |= {
        "dense": stats.as_dict() | {"scene_points": total, "fused_points": fused,
                                    "tiles": len(manifest["tiles"]), "tiles_built": _tiles_built(parent, manifest)},
        "dense_tiles": manifest,
        "dense_voxel": voxel,
    }
    return write_tileset(manifest, dense_dir / "tiles.json")


def _tiles_built(parent: dict | None, manifest: dict) -> int:
    old = {t["dir"] for t in parent["tiles"].values()} if parent else set()
    return sum(t["dir"] not in old for t in manifest["tiles"].values())


def write_tileset(manifest: dict, path: Path) -> Path:
    """The viewer's list of the dense tiles' LoDs."""
    return _write_lod_list([rad for rad, _ in lod_files(manifest, DATA_ROOT)], path)


def _write_lod_list(rads: list[Path], path: Path) -> Path:
    """A list of LoDs the viewer streams together, each as its own paged mesh."""
    path.write_text(json.dumps({"tiles": [{"url": files_url(rad)} for rad in rads]}))
    return path


def _training_should_stop(version_id: int) -> bool:
    """Cancelled from the UI, or (automatic jobs only) a new video is waiting: it will
    produce a newer HEAD, which supersedes this training anyway."""
    with session_scope() as session:
        v = session.get(Version, version_id)
        if v is None or v.train_status == "cancelling":
            return True
        if not v.train_auto:
            return False
        return session.exec(
            select(Contribution).where(Contribution.scene_id == v.scene_id, Contribution.status == "queued")
        ).first() is not None


def _update_version(version_id: int, **fields) -> None:
    with session_scope() as session:
        v = session.get(Version, version_id)
        for key, value in fields.items():
            setattr(v, key, value)
        v.updated_at = utcnow()
        session.add(v)
        session.commit()


def _train_global(v: Version, quality: str, steps: int, max_resolution: int, reconstruction: Path,
                  rec: pycolmap.Reconstruction, image_root: Path, work: Path, progress: Progress,
                  timing: TimingModel, show_preview: Callable[[list[Path], int], None]) -> tuple[Path, dict]:
    """One run over the whole scene; returns the LoD and details for the metrics."""
    # Uniform splats can't grow to cover gaps, so they start from the dense cloud (plus
    # a sky shell), each with a fixed on-screen size.
    init_ply, uniform = None, None
    if quality == "uniform":
        if not v.metrics.get("dense_tiles"):
            raise RuntimeError("uniform splats need the version's dense point cloud")
        init_ply = work / "init.ply"
        cloud = sample_cloud(v.metrics["dense_tiles"], DATA_ROOT, UNIFORM_INIT_POINTS)
        uniform = uniform_init(cloud, rec, image_root, init_ply, max_resolution,
                               pixels=UNIFORM_PIXELS, num_points=UNIFORM_INIT_POINTS)

    progress.start("train")
    snapshots: list[Path] = []

    def on_snapshot(s: Snapshot) -> None:
        progress.update(s.step / s.total_steps, detail=f"Step {s.step}/{s.total_steps}")
        if s.step >= s.total_steps:
            return
        preview_dir = work / f"preview_{s.step}"
        show_preview([build_lod(BUILD_LOD_BIN, s.ply_path, preview_dir, quality=False)], s.step)
        snapshots.append(preview_dir)
        while len(snapshots) > 2:
            shutil.rmtree(snapshots.pop(0), ignore_errors=True)

    raw_ply = compute.trainer().train(
        image_root, reconstruction, work / "train", steps, max_resolution, init_ply, TRAIN_PREVIEW_EVERY,
        on_snapshot, should_stop=lambda: _training_should_stop(v.id),
        sh_degree=0 if uniform else 3, freeze_scales=uniform is not None,
    )
    timing.record(f"train.{quality}", steps, progress.finish())

    progress.start("cleanup")
    splat_ply = work / "splat.ply"
    stats = clean_splats(raw_ply, rec, splat_ply)
    shutil.rmtree(work / "train" / "exports", ignore_errors=True)
    timing.record("cleanup", 1, progress.finish())

    progress.start("lod")
    rad = build_lod(BUILD_LOD_BIN, splat_ply, work / "lod", quality=True)
    timing.record("lod", 1, progress.finish())
    return rad, {**({"uniform": uniform} if uniform else {}), "cleanup": stats.as_dict(),
                 "eval": eval_results(work / "train")}


def _train_blocks(v: Version, scene: Scene, quality: str, steps: int, max_resolution: int, reconstruction: Path,
                  rec: pycolmap.Reconstruction, image_root: Path, work: Path, progress: Progress,
                  timing: TimingModel, show_preview: Callable[[list[Path], int], None]) -> tuple[Path, dict]:
    """Each ground block trained on its own frames (worker.pipeline.splat.blocks), cropped to
    its cells, cleaned and built into its own LoD; returns the list of block LoDs. Blocks the
    new videos left alone are taken from the nearest earlier version trained at this quality
    (_plan_blocks), so only the stale ones train. Scene settings can add training frames
    (worker.pipeline.splat.frames) and start each block from the dense cloud."""
    settings = _block_settings(scene)
    bp = _plan_blocks(v, quality, steps, max_resolution, settings, rec)
    geo, blocks = bp.geo, bp.blocks
    _save_block_plan(v.id, bp)
    stale = [block for block in blocks if block.key in bp.stale]
    plan = [(block, steps) for block in stale]
    total = sum(n for _, n in plan) or 1
    # Held out: every 8th keyframe by name, as one global Brush run would hold out, so block
    # and global splats are measured on the same unseen frames.
    names = sorted(image.name for image in rec.images.values() if image.has_pose)
    held_out = set(names[::8])
    full = pycolmap.Reconstruction(reconstruction)
    keyframes = {block.key: (block.core, block.images) for block in stale}

    train_root, frames = image_root, None
    if progress.has("frames"):
        train_root = SCENES_ROOT / scene.slug / "train_images"
        # Only the videos the stale blocks train on need their extra frames.
        prefixes = {name.split("/")[0] for block in stale for name in block.images}
        follows, frames = _training_frames(v, scene, full, rec, image_root, train_root, held_out, progress, timing,
                                           prefixes)
        for block in stale:
            mine = set(block.images)
            block.images = sorted(block.images + [x for x, after in follows.items() if after in mine])
    registered = {image.name for image in full.images.values() if image.has_pose}
    report = {"frames": frames.as_dict() if frames else None, "blocks": {}, "steps": steps,
              "max_resolution": max_resolution}

    rads, results = {}, {}
    for key, old in bp.reused.items():
        rads[key] = DATA_ROOT / old["dir"] / "lod" / "splat-lod.rad"
        results[key] = old | {"reused": True, "from": old.get("from", bp.ancestor.number)}
        sky = DATA_ROOT / old["dir"] / "sky.json"
        if "sky" not in old and sky.exists():  # removed by scripts.remove_sky after training
            results[key]["sky"] = json.loads(sky.read_text())
        report["blocks"][key] = {k: results[key].get(k) for k in ("frames", "train_frames", "eval", "kept", "sky",
                                                                  "reused", "from")}
    _report(v.id, report)

    cloud, dense_init = None, settings["dense_init"]
    if progress.has("init") and stale:
        manifest = v.metrics["dense_tiles"]
        progress.start("init", substeps=[("sample", len(manifest["tiles"]))])
        with progress.step("sample", len(manifest["tiles"])):
            # Blocks overlap and differ in density; 4 per block's share leaves each enough.
            cloud = sample_cloud(manifest, DATA_ROOT, 4 * dense_init * len(stale), on_progress=progress.counts)
        progress.finish()

    sky_every = settings["sky_every"]
    progress.set_estimate("train", timing.estimate(f"train.{quality}", total))
    progress.start("train")
    done = 0
    for i, (block, n) in enumerate(plan, start=1):
        block_dir = work / block.key
        (block_dir / "model").mkdir(parents=True)
        # What it trains on, for the next version's reuse check (keyframes; extras follow them).
        core, images = keyframes[block.key]
        (block_dir / "frames.json").write_text(json.dumps({"core": core, "images": images}))
        model = block_model(full, geo, block)
        model.write(block_dir / "model")
        init_ply, init = None, None
        if cloud is not None:
            progress.update(done / total, detail=f"Block {i}/{len(plan)} · writing its dense start")
            init_ply = block_dir / "init.ply"
            init = block_init(cloud, model, geo, block, dense_init, init_ply)
        snapshots: list[Path] = []

        def on_snapshot(s: Snapshot, block: Block = block, block_dir: Path = block_dir, i: int = i) -> None:
            progress.update((done + s.step) / total, detail=f"Block {i}/{len(plan)} · step {s.step}/{s.total_steps}")
            if s.step >= s.total_steps:
                return
            preview_dir = block_dir / f"preview_{s.step}"
            preview_dir.mkdir()
            crop_to_block(s.ply_path, preview_dir / "splat.ply", geo, block, blocks)
            show_preview(list(rads.values()) + [build_lod(BUILD_LOD_BIN, preview_dir / "splat.ply",
                                                          preview_dir / "lod", quality=False)], done + s.step)
            snapshots.append(preview_dir)
            while len(snapshots) > 2:
                shutil.rmtree(snapshots.pop(0), ignore_errors=True)

        eval_names = sorted(held_out & set(block.images) & registered)
        raw = compute.trainer().train(
            train_root, block_dir / "model", block_dir / "train", n, max_resolution, init_ply, TRAIN_PREVIEW_EVERY,
            on_snapshot, should_stop=lambda: _training_should_stop(v.id), eval_names=eval_names,
        )
        if init_ply is not None:
            init_ply.unlink()
        kept = crop_to_block(raw, block_dir / "cropped.ply", geo, block, blocks)
        shutil.rmtree(block_dir / "train" / "exports", ignore_errors=True)
        stats = clean_splats(block_dir / "cropped.ply", block_model(rec, geo, block), block_dir / "splat.ply")
        (block_dir / "cropped.ply").unlink()
        sky = None
        if sky_every:
            progress.update((done + n) / total, detail=f"Block {i}/{len(plan)} · removing the sky")
            sky = remove_sky(block_dir, model, geo, train_root, SCENES_ROOT / scene.slug / "sky_masks", sky_every)
        rads[block.key] = build_lod(BUILD_LOD_BIN, block_dir / "splat.ply", block_dir / "lod", quality=True)
        done += n
        evals = eval_results(block_dir / "train")
        results[block.key] = {
            "dir": rel(block_dir), "cells": block.cells, "grow": block.grow, "frames": len(block.core),
            "train_frames": len(set(block.images) & registered) - len(eval_names), "eval_frames": len(eval_names),
            "steps": n, "images_hash": block.images_hash, "eval": evals[-1] if evals else None, "kept": kept,
            "cleanup": stats.as_dict(), "stale": bp.stale[block.key], **({"init": init} if init else {}),
            **({"sky": sky} if sky else {}),
        }
        report["blocks"][block.key] = {k: results[block.key].get(k) for k in ("frames", "train_frames", "eval", "kept", "sky")}
        _report(v.id, report)
        log.info("version %s block %s (%d/%d, %s): %s", v.id, block.key, i, len(plan), bp.stale[block.key],
                 results[block.key]["eval"])
    seconds = progress.finish()
    if plan:  # all reused: nothing to learn the step time from
        timing.record(f"train.{quality}", total, seconds)
    # In plan order (the overlay's numbering).
    results = {block.key: results[block.key] for block in blocks}
    return _write_lod_list([rads[block.key] for block in blocks], work / "tiles.json"), {
        "block_frames": TRAIN_BLOCK_FRAMES, "steps_total": sum(n for _, n in plan), "blocks": results,
        "settings": settings, "geo": geo,
        **({"reused_from": bp.ancestor.number} if bp.reused else {}),
        **({"frames": frames.as_dict()} if frames else {}), **({"dense_init": dense_init} if cloud is not None else {})}


@dataclass
class _BlockPlan:
    geo: dict  # the map frame the blocks' cells are in (the first block-trained version's)
    blocks: list[Block]
    ancestor: Version | None  # nearest earlier version trained in blocks at this quality
    reused: dict[str, dict]  # key -> the ancestor's result for a block that stays as it is
    stale: dict[str, str]  # key -> why it trains


def _block_settings(scene: Scene) -> dict:
    """Scene settings a trained block depends on besides its frames, steps and resolution."""
    return {name: scene_setting(scene.settings, setting, default) for name, setting, default in (
        ("shared_cameras", "TRAIN_SHARED_CAMERAS", TRAIN_SHARED_CAMERAS),
        ("frame_side", "TRAIN_FRAME_SIDE", TRAIN_FRAME_SIDE), ("extra_frames", "TRAIN_EXTRA_FRAMES", TRAIN_EXTRA_FRAMES),
        ("dense_init", "TRAIN_DENSE_INIT", TRAIN_DENSE_INIT), ("sky_every", "TRAIN_SKY_EVERY", TRAIN_SKY_EVERY))}


def _trained_ancestor(v: Version, quality: str | None) -> tuple[Version, str] | None:
    """Nearest earlier version trained in blocks at this quality (None: at any, full first)."""
    qualities = [quality] if quality else ["full", "quick"]
    with session_scope() as session:
        parent = session.get(Version, v.parent_id) if v.parent_id else None
        while parent is not None:
            for q in qualities:
                if parent.metrics.get(f"splat_{q}", {}).get("blocks"):
                    return parent, q
            parent = session.get(Version, parent.parent_id) if parent.parent_id else None
    return None


def _moved(a: pycolmap.Image, b: pycolmap.Image, metre: float) -> bool:
    """A frame posed differently in two versions (a re-run of the scene's model; a new video
    leaves earlier frames exactly where they were: Achapnyak v5 -> v6 moved none)."""
    shift = np.linalg.norm(a.projection_center() - b.projection_center()) / metre
    turn = np.degrees(np.arccos(np.clip(np.dot(a.viewing_direction(), b.viewing_direction()), -1, 1)))
    return shift > 0.05 or turn > 0.05


def _trained_frames(result: dict, posed: set[str]) -> set[str]:
    """Keyframes a block was trained on: its frames.json, or (blocks from before it) its model."""
    block_dir = DATA_ROOT / result["dir"]
    if (block_dir / "frames.json").exists():
        return set(json.loads((block_dir / "frames.json").read_text())["images"])
    model = pycolmap.Reconstruction(block_dir / "model")
    return {im.name for im in model.images.values() if im.has_pose} & posed


def _plan_blocks(v: Version, quality: str, steps: int, max_resolution: int, settings: dict,
                 rec: pycolmap.Reconstruction) -> _BlockPlan:
    """The version's blocks, keeping the trained ancestor's (plan_blocks `keep`, in its map
    frame, so its splats and the new blocks' crops meet at the same seams), and which of them
    must train: a new or grown block, one whose frames moved, one with TRAIN_STALE_FRAMES new
    frames of its own, or new frames in its margin beyond TRAIN_STALE_MARGIN of its own frames.
    `rec`: the version's model with its SfM points cleaned, as training uses it."""
    found = _trained_ancestor(v, quality)
    if found is None:
        blocks = plan_blocks(rec, v.metrics["geo"], TRAIN_BLOCK_FRAMES)
        return _BlockPlan(v.metrics["geo"], blocks, None, {}, {block.key: "new" for block in blocks})
    ancestor, _ = found
    old = ancestor.metrics[f"splat_{quality}"]
    geo = old.get("geo") or ancestor.metrics["geo"]
    keep = [{"key": key, "cells": b["cells"], "grow": b["grow"], "frames": b["frames"]} for key, b in old["blocks"].items()]
    blocks = plan_blocks(rec, geo, TRAIN_BLOCK_FRAMES, keep=keep)

    before = pycolmap.Reconstruction(DATA_ROOT / ancestor.reconstruction_path)
    was = {im.name: im for im in before.images.values() if im.has_pose}
    now = {im.name: im for im in rec.images.values() if im.has_pose}
    added = now.keys() - was.keys()
    metre = float(np.linalg.norm(np.array(geo["scene_from_enu"])[:3, 0]))  # scene units per metre
    moved = {name for name in now.keys() & was.keys() if _moved(was[name], now[name], metre)}
    # Blocks from before settings were recorded were trained with the scene's settings then.
    same = old.get("steps") == steps and old.get("max_resolution") == max_resolution and \
        old.get("settings", settings) == settings

    reused, stale = {}, {}
    for block in blocks:
        result = old["blocks"].get(block.key)
        own = len(added & set(block.core))
        margin = len(added & (set(block.images) - set(block.core)))
        if result is None:
            stale[block.key] = "new"
        elif sorted(map(tuple, result["cells"])) != sorted(map(tuple, block.cells)):
            stale[block.key] = "grew"
        elif not same:
            stale[block.key] = "settings changed"
        elif moved and (n := len(moved & _trained_frames(result, set(was)))):
            stale[block.key] = f"{n} frames moved"
        elif own >= TRAIN_STALE_FRAMES:
            stale[block.key] = f"{own} new frames"
        elif margin > TRAIN_STALE_MARGIN * len(block.core):
            stale[block.key] = f"{margin} new frames in its margin"
        else:
            reused[block.key] = result
    return _BlockPlan(geo, blocks, ancestor, reused, stale)


def inherit_blocks(version_id: int) -> None:
    """A new version shows its trained ancestor's block splats straight away: all of them, the
    stale ones outdated until trained (a block that is planned again shows its earlier splats
    where it was). The block plan marks which blocks are reused and which stale; training then
    trains only the stale ones (_train_blocks)."""
    with session_scope() as session:
        v = session.get(Version, version_id)
        scene = session.get(Scene, v.scene_id)
    if not v.metrics.get("geo") or TRAIN_BLOCK_FRAMES <= 0 or v.splat_url:
        return
    found = _trained_ancestor(v, None)
    if found is None:
        return
    ancestor, quality = found
    steps, max_resolution = TRAIN_SETTINGS[quality]
    steps = scene_setting(scene.settings, f"TRAIN_{quality.upper()}_STEPS", steps)
    max_resolution = scene_setting(scene.settings, f"TRAIN_{quality.upper()}_RES", max_resolution)
    rec = pycolmap.Reconstruction(DATA_ROOT / v.reconstruction_path)
    clean_sparse(rec)
    bp = _plan_blocks(v, quality, steps, max_resolution, _block_settings(scene), rec)
    rads = [DATA_ROOT / b["dir"] / "lod" / "splat-lod.rad" for b in ancestor.metrics[f"splat_{quality}"]["blocks"].values()]
    work = SCENES_ROOT / scene.slug / "versions" / f"v{v.number}" / f"{quality}_inherited"
    work.mkdir(parents=True, exist_ok=True)
    tiles = _write_lod_list(rads, work / "tiles.json")
    _save_block_plan(version_id, bp, quality)
    with session_scope() as session:
        v = session.get(Version, version_id)
        if v.splat_url:  # trained meanwhile
            return
        v.splat_path, v.splat_url, v.splat_kind = rel(tiles), files_url(tiles), quality
        v.splats = v.splats | {quality: v.splat_url}
        v.updated_at = utcnow()
        session.add(v)
        session.commit()
    log.info("version %s: %d of %d blocks from v%s, %d to train (%s)", version_id, len(bp.reused), len(bp.blocks),
             ancestor.number, len(bp.stale), ", ".join(f"{k}: {r}" for k, r in bp.stale.items()))


def _save_block_plan(version_id: int, bp: _BlockPlan, quality: str | None = None) -> None:
    """Each block's ground cells in training order and whether it is reused or must train, for
    the viewer's block overlay (kept after training, unlike the report). `quality`: the
    ancestor's splat the version shows until it trains its own (inherit_blocks)."""
    plan = block_plan(bp.blocks)
    for entry in plan:
        if entry["key"] in bp.reused:
            entry |= {"status": "reused", "from": bp.reused[entry["key"]].get("from", bp.ancestor.number)}
        else:
            entry |= {"status": "stale", "reason": bp.stale[entry["key"]]}
    with session_scope() as session:
        v = session.get(Version, version_id)
        v.metrics = v.metrics | {"block_plan": plan, "block_geo": bp.geo["scene_from_map"]} | (
            {"blocks_inherited": {"quality": quality, "from": bp.ancestor.number, "reused": len(bp.reused),
                                  "stale": len(bp.stale)}} if quality and bp.ancestor else {})
        session.add(v)
        session.commit()


def _report(version_id: int, report: dict) -> None:
    """What the running training has found so far (frames prepared, blocks finished), shown
    under the version while it trains; replaced by the splat metrics when it ends."""
    with session_scope() as session:
        v = session.get(Version, version_id)
        v.metrics = v.metrics | {"training_report": report}
        session.add(v)
        session.commit()


def _videos(v: Version) -> list[Video]:
    """The version's videos as named in its reconstruction: frames are c<id>/..., <id> being
    the contribution that uploaded the video (stored as <scene>/uploads/<id>_<file>; a cloned
    scene keeps the original's names and files, so its frames.json is with the original's
    contributions, and a video added to the clone has its own)."""
    with session_scope() as session:
        contributions = [session.get(Contribution, i) for i in v.contribution_ids]
    videos = []
    for c in contributions:
        upload = DATA_ROOT / c.video_path  # absolute paths stay as they are
        prefix = "c" + upload.name.split("_", 1)[0]
        frames = json.loads((upload.parent.parent / "contributions" / prefix / "frames.json").read_text())
        videos.append(Video(prefix, Path(c.video_path), frames, DECODE_THREADS))
    return videos


def _training_frames(v: Version, scene: Scene, full: pycolmap.Reconstruction, rec: pycolmap.Reconstruction,
                     image_root: Path, train_root: Path, held_out: set[str], progress: Progress,
                     timing: TimingModel, prefixes: set[str] | None = None) -> tuple[dict[str, str], FrameStats]:
    """One camera per video, keyframes at full resolution and extra frames, per the scene's
    settings (worker.pipeline.splat.frames), the latter two for the videos in `prefixes` (None:
    all). Changes `full` (and `rec`'s poses to match); returns extra frame -> the keyframe it
    follows, and the stats."""
    side = scene_setting(scene.settings, "TRAIN_FRAME_SIDE", TRAIN_FRAME_SIDE)
    extra = scene_setting(scene.settings, "TRAIN_EXTRA_FRAMES", TRAIN_EXTRA_FRAMES)
    videos = [video for video in _videos(v) if prefixes is None or video.prefix in prefixes]
    stats = FrameStats(keyframes=sum(1 for im in full.images.values() if im.has_pose))
    gaps = sum(len(video.frames) - 1 for video in videos) * extra
    substeps = [("cameras", stats.keyframes)]
    if side or extra:
        substeps += [("pick", sum(candidates(video) for video in videos) if extra else 0),
                     ("decode", (stats.keyframes if side else 0) + gaps)]
    if extra:
        substeps += [("pose", gaps)]
    progress.start("frames", substeps=substeps)

    with progress.step("cameras", stats.keyframes):
        cameras = shared_cameras(full, stats)
        copy_poses(full, rec)
        progress.counts(stats.keyframes, stats.keyframes)

    follows: dict[str, str] = {}
    if side or extra:
        lock, count = threading.Lock(), [0]

        def tick(n: int, total: int) -> None:
            with lock:
                count[0] += n
                progress.counts(min(count[0], total), total)

        metas = {}
        with ThreadPoolExecutor(DECODE_THREADS) as pool:
            total = substeps[1][1]
            with progress.step("pick", total):
                for video in videos:
                    metas[video.prefix] = plan_images(video, train_root / video.prefix, side, extra, pool,
                                                      lambda n: tick(n, total))
            todo = [m for m in metas.values() if not m["decoded"]]
            total = sum((len(m["keyframes"]) if side else 0) + len(m["extras"]) for m in todo)
            count[0] = 0
            with progress.step("decode", total):
                for video in videos:
                    metas[video.prefix] = decode_images(video, metas[video.prefix], train_root / video.prefix,
                                                        image_root, pool, lambda n: tick(n, total))
        stats.full_resolution = stats.keyframes if side else 0
        if extra:
            with progress.step("pose", gaps):
                follows = add_extra_frames(full, videos, metas, cameras, image_root, train_root, held_out, stats,
                                           progress.counts)
    progress.finish()
    log.info("version %s training frames: %s", v.id, json.dumps(stats.as_dict()))
    return follows, stats


def _block_users(v: Version, work: Path) -> list[str]:
    """Other versions whose splats use block LoDs in `work` (reused, not copied), which
    training again would delete."""
    prefix = rel(work) + "/"
    with session_scope() as session:
        others = session.exec(select(Version).where(Version.scene_id == v.scene_id, Version.id != v.id)).all()
    users = []
    for o in others:
        dirs = [b.get("dir", "") for key, m in o.metrics.items() if key.startswith("splat_") and isinstance(m, dict)
                for b in (m.get("blocks") or {}).values()]
        lists = [DATA_ROOT / url.removeprefix("/files/") for url in (o.splats or {}).values() if url.endswith("tiles.json")]
        if any(d.startswith(prefix) for d in dirs) or any(
                path.exists() and f"/files/{prefix}" in path.read_text() for path in lists):
            users.append(f"v{o.number}")
    return users


def train_version(version_id: int) -> None:
    """Train the version's splat at its requested quality, remove floaters, build the LoD
    and swap it in. GPS scenes train in ground blocks (_train_blocks), others in one run."""
    with session_scope() as session:
        v = session.get(Version, version_id)
        scene = session.get(Scene, v.scene_id)
    quality = v.train_quality or "quick"
    steps, max_resolution = TRAIN_SETTINGS[quality]
    steps = scene_setting(scene.settings, f"TRAIN_{quality.upper()}_STEPS", steps)
    max_resolution = scene_setting(scene.settings, f"TRAIN_{quality.upper()}_RES", max_resolution)
    # The frames live with the scene that made the reconstruction (a showcase scene such as
    # office-tsdf borrows another scene's reconstruction and has no images of its own).
    image_root = SCENES_ROOT / Path(v.reconstruction_path).parts[1] / "images"
    blocks = quality != "uniform" and TRAIN_BLOCK_FRAMES > 0 and bool(v.metrics.get("geo"))
    work = SCENES_ROOT / scene.slug / "versions" / f"v{v.number}" / (f"{quality}_blocks" if blocks else quality)
    if users := _block_users(v, work):
        raise RuntimeError(f"its {quality} blocks are reused by {', '.join(users)}; train a newer version instead")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    timing = TimingModel(compute.device())
    stages = [("train", timing.estimate(f"train.{quality}", steps))]
    if blocks:
        # Rough until the stages start and know their counts.
        keyframes = v.metrics.get("registered") or 1000
        if any(scene_setting(scene.settings, name, default) for name, default in (
                ("TRAIN_SHARED_CAMERAS", TRAIN_SHARED_CAMERAS), ("TRAIN_FRAME_SIDE", TRAIN_FRAME_SIDE),
                ("TRAIN_EXTRA_FRAMES", TRAIN_EXTRA_FRAMES))):
            stages.insert(0, ("frames", timing.estimate("frames.decode", 2 * keyframes)))
        if scene_setting(scene.settings, "TRAIN_DENSE_INIT", TRAIN_DENSE_INIT) and v.metrics.get("dense_tiles"):
            stages.insert(-1, ("init", timing.estimate("init.sample", len(v.metrics["dense_tiles"]["tiles"]))))
    else:
        stages += [("cleanup", timing.estimate("cleanup", 1)), ("lod", timing.estimate("lod", 1))]
    progress = Progress(training_sink(v.id), stages, timing)
    previews: list[Path] = []
    started = time.time()
    try:
        # Train from all SfM points (initialising from the cleaned ones scored 0.4 dB lower
        # on the room), but carve floaters against the cleaned ones: a single bad point
        # would carve a tunnel through real geometry.
        reconstruction = DATA_ROOT / v.reconstruction_path
        rec = pycolmap.Reconstruction(reconstruction)
        removed = clean_sparse(rec)

        def show_preview(rads: list[Path], step: int) -> None:
            # Named by step, not by len(previews): that stops growing at 2 once old ones are
            # dropped, so every later preview overwrote the one before and was then deleted.
            preview = work / f"preview_{step}.json"
            _write_lod_list(rads, preview)
            _update_version(v.id, preview_url=files_url(preview), preview_step=step)
            previews.append(preview)
            # Keep the latest two so a client mid-stream on the previous one doesn't 404.
            while len(previews) > 2:
                previews.pop(0).unlink(missing_ok=True)

        if blocks:
            rad, details = _train_blocks(v, scene, quality, steps, max_resolution, reconstruction, rec, image_root,
                                         work, progress, timing, show_preview)
        else:
            rad, details = _train_global(v, quality, steps, max_resolution, reconstruction, rec, image_root, work,
                                         progress, timing, show_preview)

        with session_scope() as session:
            v = session.get(Version, version_id)
            v.metrics = {k: val for k, val in v.metrics.items() if k not in ("training_report", "blocks_inherited")} | {
                f"splat_{quality}": {
                    "steps": steps,
                    "max_resolution": max_resolution,
                    "sparse_points_removed": removed,
                    **details,
                    "seconds": round(time.time() - started, 1),
                },
            }
            v.splat_path, v.splat_url, v.splat_kind = rel(rad), files_url(rad), quality
            v.splats = v.splats | {quality: v.splat_url}
            v.train_status = v.train_detail = v.preview_url = v.preview_step = None
            v.train_progress, v.train_eta, v.updated_at = 1.0, 0.0, utcnow()
            session.add(v)
            session.commit()
        log.info("version %s: %s splat published", version_id, quality)
    except TrainingStopped:
        with session_scope() as session:
            v = session.get(Version, version_id)
            cancelled = v.train_status == "cancelling"
        # Automatic jobs go back in the queue (dropped later if no longer HEAD).
        _update_version(
            version_id, train_status=None if cancelled else "queued", preview_url=None, preview_step=None,
            train_detail="cancelled" if cancelled else "paused for a new video", train_progress=0.0, train_eta=None,
        )
        log.info("version %s training %s", version_id, "cancelled" if cancelled else "paused")
    finally:
        progress.close()
        for preview in previews:
            preview.unlink(missing_ok=True)
        for preview_dir in work.glob("**/preview_*"):
            shutil.rmtree(preview_dir, ignore_errors=True)
