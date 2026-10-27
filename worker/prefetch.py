"""Work on queued contributions that doesn't depend on the scene HEAD, run ahead of the
sequential lane (which links one video at a time into the HEAD the previous one left):

  * frame extraction: several videos at once, all sharing one pool of decode segments;
  * the video's own SfM model (its piece of the scene, worker.pipeline.pieces): one video at
    a time, oldest first;
  * depth prediction: one video at a time on the GPU, oldest first, so the depth network
    runs while the CPU-bound SfM of an earlier video does.

Results are files in the contribution's work dir with a marker written last, so a restarted
worker reuses them. While a job runs ahead, its progress shows as the queued contribution's
detail; once the sequential lane reaches the contribution, it attaches and reports the
job's progress as its own stage.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sqlmodel import select

from backend.app.config import DEPTH_MODEL, SCENES_ROOT, scene_setting
from backend.app.db import session_scope
from backend.app.models import Contribution, Scene, Version
from worker.pipeline.dense import predict_depth
from worker.pipeline.devices.dji import read_track
from worker.pipeline.features import DEFAULT_FEATURES
from worker.pipeline.ingest import extract_keyframes
from worker.pipeline.sfm import InitialResult

log = logging.getLogger(__name__)

INGEST_MARKER = "ingest.json"
DEPTH_DIR = "depth"
DEPTH_MARKER = "done"
PIECE_DIR = "piece"
PIECE_MARKER = "result.json"


def work_dir(c: Contribution, slug: str) -> Path:
    return SCENES_ROOT / slug / "contributions" / c.image_dir_name


class Relay:
    """Progress of a job that may start ahead: reported in the queued contribution's status
    (on_change) until the sequential lane attaches its own reporter."""

    def __init__(self, on_change: Callable[[], None]):
        self.on_change = on_change
        self.target: Callable[[float, str | None], None] | None = None
        self.last: tuple[float, str | None] = (0.0, None)
        self.started: float | None = None  # first progress call: the job is running

    def attach(self, target: Callable[[float, str | None], None]) -> None:
        self.target = target
        target(*self.last)

    def __call__(self, fraction: float, detail: str | None = None) -> None:
        self.started = self.started or time.time()
        self.last = (fraction, detail)
        if self.target is not None:
            self.target(fraction, detail)
        else:
            self.on_change()


def _pending(contribution_id: int) -> bool:
    with session_scope() as session:
        c = session.get(Contribution, contribution_id)
        return c is not None and c.status in ("queued", "processing")


def _set_state(contribution_id: int, detail: str, timeline: list[dict], progress: float) -> None:
    """A queued contribution's work-ahead status (the sequential lane writes its own once it
    gets there, so only while queued)."""
    with session_scope() as session:
        c = session.get(Contribution, contribution_id)
        if c is not None and c.status == "queued":
            c.detail, c.timeline, c.progress = detail, timeline, progress
            session.add(c)
            session.commit()


def _features(c: Contribution) -> str:
    """A scene keeps the feature type it was built with (see worker.pipeline.features)."""
    with session_scope() as session:
        first = session.exec(select(Version).where(Version.scene_id == c.scene_id).order_by(Version.id)).first()
    return first.metrics.get("features", "sift") if first else DEFAULT_FEATURES


class _PriorityLane:
    """One thread running submitted jobs lowest priority value first."""

    def __init__(self, name: str):
        self._heap: list[tuple[int, int, Callable[[], object], Future]] = []
        self._cv = threading.Condition()
        self._seq = 0
        threading.Thread(target=self._run, name=name, daemon=True).start()

    def submit(self, priority: int, fn: Callable[[], object]) -> Future:
        future: Future = Future()
        with self._cv:
            heapq.heappush(self._heap, (priority, self._seq, fn, future))
            self._seq += 1
            self._cv.notify()
        return future

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._heap:
                    self._cv.wait()
                _, _, fn, future = heapq.heappop(self._heap)
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except BaseException as exc:  # noqa: BLE001 - delivered to whoever waits
                    future.set_exception(exc)


def _write_gps(video: Path, frames_json: Path, prefix: str, out: Path) -> Path | None:
    """Per-keyframe GPS positions (devices.dji) for the piece job, if the video has a track."""
    track = read_track(video)
    if track is None:
        return None
    frames = json.loads(frames_json.read_text())
    positions = track.at(np.array([f["timestamp"] for f in frames]))
    path = out / "gps.json"
    path.write_text(json.dumps({
        "sigma": track.sigma, "origin": [track.origin_lat, track.origin_lon],
        "positions": {f"{prefix}/{f['name']}": [round(float(v), 3) for v in xyz] for f, xyz in zip(frames, positions)},
    }))
    return path


class _Jobs:
    """Running piece job processes, which the sequential lane pauses (SIGSTOP) while it needs
    the CPU itself (linking, densifying the video at the front of the queue)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running: set[subprocess.Popen] = set()
        self.pauses = 0

    def add(self, proc: subprocess.Popen) -> None:
        with self.lock:
            self.running.add(proc)
            if self.pauses:
                proc.send_signal(signal.SIGSTOP)

    def remove(self, proc: subprocess.Popen) -> None:
        with self.lock:
            self.running.discard(proc)

    @contextmanager
    def paused(self) -> Iterator[None]:
        with self.lock:
            self.pauses += 1
            if self.pauses == 1:
                for proc in self.running:
                    proc.send_signal(signal.SIGSTOP)
        try:
            yield
        finally:
            with self.lock:
                self.pauses -= 1
                if not self.pauses:
                    for proc in self.running:
                        proc.send_signal(signal.SIGCONT)


_jobs = _Jobs()


def _kill_orphans() -> None:
    """Piece jobs left by a worker that died while they were paused (a stopped process can't
    notice its parent is gone)."""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        pid, ppid, command = line.split(None, 2)
        if ppid == "1" and "worker.pipeline.piece_job" in command:
            log.info("killing orphaned piece job %s", pid)
            os.kill(int(pid), signal.SIGKILL)


def _run_piece_job(image_root: Path, names: list[str], out: Path, features: str, threads: int,
                   relay: Relay, gps: Path | None = None) -> dict:
    """worker.pipeline.piece_job in a subprocess capped at `threads`; its COLMAP log goes to
    out/sfm.log, its progress to `relay`."""
    names_file = out / "names.txt"
    names_file.write_text("\n".join(names))
    with open(out / "sfm.log", "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "worker.pipeline.piece_job", str(image_root), str(names_file), str(out),
             features, str(threads), str(gps or "-")],
            stdout=subprocess.PIPE, stderr=log_file, text=True,
            env=os.environ | {"OMP_NUM_THREADS": str(threads)},
        )
        assert proc.stdout is not None
        _jobs.add(proc)
        result = None
        try:
            for line in proc.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if "result" in message:
                    result = message["result"]
                else:
                    relay(message["fraction"], message["detail"])
        finally:
            _jobs.remove(proc)
        if proc.wait() != 0 or result is None:
            tail = (out / "sfm.log").read_text().splitlines()[-15:]
            raise RuntimeError("reconstruction failed:\n" + "\n".join(tail))
    return result


class _CoreLane:
    """Jobs lowest priority value first, up to `parallel` at once, splitting `cores` between
    them: a job alone in the lane gets every core, with others waiting `cores // parallel`.
    A job starts only when its share is free, so concurrent jobs never oversubscribe the CPU
    (one that started with every core makes later ones wait for it)."""

    def __init__(self, name: str, cores: int, parallel: int, demand: Callable[[], int] = lambda: 0):
        """demand(): jobs expected soon (submitted or not), so the first of a batch doesn't
        take every core just because it was submitted a moment before the others."""
        self.cores, self.parallel, self.used = cores, parallel, 0
        self.demand = demand
        self._heap: list[tuple[int, int, Callable[[int], object], Future]] = []
        self._cv = threading.Condition()
        self._seq = 0
        for k in range(parallel):
            threading.Thread(target=self._run, name=f"{name}-{k}", daemon=True).start()

    def submit(self, priority: int, fn: Callable[[int], object]) -> Future:
        """fn(threads) runs with its share of the cores."""
        future: Future = Future()
        with self._cv:
            heapq.heappush(self._heap, (priority, self._seq, fn, future))
            self._seq += 1
            self._cv.notify_all()
        return future

    def _share(self) -> int | None:
        if not self._heap:
            return None
        alone = self.used == 0 and len(self._heap) == 1 and self.demand() <= 1
        share = self.cores if alone else self.cores // self.parallel
        return share if self.cores - self.used >= share else None

    def _run(self) -> None:
        while True:
            with self._cv:
                while (share := self._share()) is None:
                    self._cv.wait()
                _, _, fn, future = heapq.heappop(self._heap)
                self.used += share
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(fn(share))
                    except BaseException as exc:  # noqa: BLE001 - delivered to whoever waits
                        future.set_exception(exc)
            finally:
                with self._cv:
                    self.used -= share
                    self._cv.notify_all()


class Prefetcher:
    def __init__(self, ingest_videos: int, decode_threads: int, sfm_videos: int = 2):
        _kill_orphans()
        self.decode_pool = ThreadPoolExecutor(decode_threads, thread_name_prefix="decode")
        self.ingest_pool = ThreadPoolExecutor(ingest_videos, thread_name_prefix="ingest")
        self.depth_lane = _PriorityLane("depth")
        self.sfm_lane = _CoreLane("sfm", os.cpu_count() or 8, sfm_videos, demand=self._pieces_needed)
        self.decode_threads = decode_threads
        self._lock = threading.RLock()  # a done-callback may run inside _start
        self._jobs: dict[tuple[str, int], tuple[Future, Relay]] = {}
        self._counts: dict[tuple[str, int], tuple[int, int]] = {}  # e.g. ("piece", id) -> (643, 695) frames
        self._seconds: dict[tuple[str, int], float] = {}  # how long finished jobs took
        self._reported: dict[int, float] = {}

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Pause the SfM models being built ahead (the caller needs every core)."""
        with _jobs.paused():
            self._report_all()
            try:
                yield
            finally:
                self._report_all()

    def _report_all(self) -> None:
        with self._lock:
            ids = {cid for _, cid in self._jobs}
        for cid in ids:
            self._report(cid, force=True)

    def _pieces_needed(self) -> int:
        """Queued or processing videos without their own model yet."""
        with session_scope() as session:
            rows = session.exec(
                select(Contribution, Scene.slug).join(Scene, Scene.id == Contribution.scene_id)
                .where(Contribution.status.in_(["queued", "processing"]))
            ).all()
        return sum(not (work_dir(c, slug) / PIECE_DIR / PIECE_MARKER).exists() for c, slug in rows)

    # (job kind, timeline stage, label, share of the whole contribution's progress); the rest
    # is fitting into the scene and densifying, which wait for the earlier videos.
    STAGES = [("ingest", "ingest", "frames", 0.1), ("piece", "sfm", "3D model", 0.5),
              ("depth", "depth", "depth maps", 0.1)]

    def _report(self, contribution_id: int, force: bool = False) -> None:
        """The queued contribution's work-ahead status: a timeline like a processing one's
        (done / active / pending per stage), the overall progress, and as detail what the
        active stages are doing, e.g. "3D model: Part 2/4: Mapping, global (attempt 1) 3/5"."""
        # Composed and written under the jobs lock: callbacks from different threads must not
        # write their snapshots out of order.
        with self._lock:
            now = time.time()
            if not force and now - self._reported.get(contribution_id, 0.0) < 2.0:
                return
            self._reported[contribution_id] = now
            details, timeline, progress = [], [], 0.0
            for kind, stage, label, share in self.STAGES:
                job = self._jobs.get((kind, contribution_id))
                entry = {"name": stage, "status": "pending", "seconds": None, "estimate": 0, "remaining": None}
                if job is not None and job[0].done():
                    if error := job[0].exception():
                        details.append(f"{label} failed: {error}")
                    else:
                        entry |= {"status": "done", "seconds": self._seconds.get((kind, contribution_id)),
                                  "remaining": 0}
                        if counts := self._counts.get((kind, contribution_id)):
                            entry |= {"done": counts[0], "total": counts[1]}
                        progress += share
                elif job is not None and job[0].running():
                    fraction, detail = job[1].last
                    if kind == "piece" and _jobs.pauses:
                        detail = "paused while an earlier video is fitted into the scene"
                    elapsed = now - (job[1].started or now)
                    entry |= {"status": "active", "seconds": round(elapsed, 1),
                              "remaining": round(elapsed * (1 - fraction) / fraction, 1) if fraction > 0.05 else None}
                    details.append(f"{label}: {detail or 'starting'}")
                    progress += share * fraction
                timeline.append(entry)
            timeline += [{"name": stage, "status": "pending", "seconds": None, "estimate": 0, "remaining": None}
                         for stage in ("register", "dense")]
            _set_state(contribution_id, " · ".join(details), timeline, round(progress, 4))

    def start(self, every: float = 3.0) -> None:
        """Pick up new uploads in the background, also while the sequential lane is busy."""
        def loop() -> None:
            while True:
                try:
                    self.schedule()
                except Exception:  # noqa: BLE001 - keep scheduling
                    log.exception("prefetch scheduling failed")
                time.sleep(every)

        threading.Thread(target=loop, name="prefetch", daemon=True).start()

    def schedule(self) -> None:
        """Start frame extraction (then depth) for every queued contribution."""
        with session_scope() as session:
            rows = session.exec(
                select(Contribution, Scene.slug).join(Scene, Scene.id == Contribution.scene_id)
                .where(Contribution.status.in_(["queued", "processing"])).order_by(Contribution.id)
            ).all()
        for c, slug in rows:
            self.ingest(c, slug)
            self._report(c.id)

    def ingest(self, c: Contribution, slug: str, retry: bool = False) -> tuple[Future, Relay]:
        """Future of (frames as dicts, seconds the extraction took); starts it if needed, and
        with `retry` restarts it if it failed."""
        relay = Relay(lambda: self._report(c.id))

        def run() -> tuple[list[dict], float]:
            work = work_dir(c, slug)
            marker = work / INGEST_MARKER
            if not marker.exists():
                if not _pending(c.id):
                    raise RuntimeError(f"contribution {c.id} is no longer queued")
                images = SCENES_ROOT / slug / "images" / c.image_dir_name
                shutil.rmtree(work, ignore_errors=True)
                shutil.rmtree(images, ignore_errors=True)
                work.mkdir(parents=True)
                started = time.time()
                frames = extract_keyframes(
                    Path(c.video_path), images, pool=self.decode_pool, segments=self.decode_threads,
                    on_progress=lambda f: relay(f, f"Decoding {f:.0%} of the video, picking sharp frames by motion"),
                )
                (work / "frames.json").write_text(json.dumps([f.__dict__ for f in frames]))
                marker.write_text(json.dumps({"seconds": round(time.time() - started, 1)}))
                log.info("contribution %s: %d keyframes in %.0fs", c.id, len(frames), time.time() - started)
            frames = json.loads((work / "frames.json").read_text())
            self._counts[("ingest", c.id)] = (len(frames), len(frames))
            names = [f"{c.image_dir_name}/{f['name']}" for f in frames]
            self.piece(c, slug, names)
            self.depth(c, slug, names)
            return frames, json.loads(marker.read_text())["seconds"]

        return self._start(("ingest", c.id), relay, lambda: self.ingest_pool.submit(run), retry)

    def depth(self, c: Contribution, slug: str, names: list[str], retry: bool = False) -> tuple[Future, Relay]:
        """Future of the directory of depth maps for these frames; starts it if needed, and
        with `retry` restarts it if it failed."""
        relay = Relay(lambda: self._report(c.id))

        def run() -> Path:
            out = work_dir(c, slug) / DEPTH_DIR
            if not (out / DEPTH_MARKER).exists():
                if not _pending(c.id):
                    raise RuntimeError(f"contribution {c.id} is no longer queued")
                shutil.rmtree(out, ignore_errors=True)
                with session_scope() as session:
                    settings = session.get(Scene, c.scene_id).settings
                predict_depth(SCENES_ROOT / slug / "images", names, out, scene_setting(settings, "DEPTH_MODEL", DEPTH_MODEL),
                              on_progress=lambda done, total: relay(done / total, f"Predicting depth {done}/{total}"))
                (out / DEPTH_MARKER).touch()
                if not _pending(c.id):  # rejected meanwhile: nobody will fuse these
                    shutil.rmtree(out, ignore_errors=True)
                    raise RuntimeError(f"contribution {c.id} is no longer queued")
            return out

        return self._start(("depth", c.id), relay, lambda: self.depth_lane.submit(c.id, run), retry)

    def piece(self, c: Contribution, slug: str, names: list[str], retry: bool = False) -> tuple[Future, Relay]:
        """Future of (the video's own SfM result, seconds it took); starts it if needed, and
        with `retry` restarts it if it failed."""
        relay = Relay(lambda: self._report(c.id))

        def run(threads: int) -> tuple[InitialResult, float]:
            out = work_dir(c, slug) / PIECE_DIR
            marker = out / PIECE_MARKER
            if not marker.exists():
                if not _pending(c.id):
                    raise RuntimeError(f"contribution {c.id} is no longer queued")
                shutil.rmtree(out, ignore_errors=True)
                out.mkdir(parents=True)
                started = time.time()
                relay(0.0, f"Starting on {threads} cores")
                gps = _write_gps(Path(c.video_path), work_dir(c, slug) / "frames.json", c.image_dir_name, out)
                result = _run_piece_job(SCENES_ROOT / slug / "images", names, out, _features(c), threads, relay, gps)
                fields = result | {"seconds": round(time.time() - started, 1), "threads": threads}
                marker.write_text(json.dumps(fields))
                log.info("contribution %s: piece of %d/%d frames in %.0fs on %d cores (%s)", c.id,
                         result["num_registered"], result["num_images"], time.time() - started, threads,
                         result["defect"] or "ok")
            fields = json.loads(marker.read_text())
            self._counts[("piece", c.id)] = (fields["num_registered"], fields["num_images"])
            seconds = fields.pop("seconds")
            fields.pop("threads", None)
            fields["reconstruction_path"] = Path(fields["reconstruction_path"])
            fields["database_path"] = Path(fields["database_path"])
            return InitialResult(**fields), seconds

        return self._start(("piece", c.id), relay, lambda: self.sfm_lane.submit(c.id, run), retry)

    def _start(self, key: tuple[str, int], relay: Relay, submit: Callable[[], Future],
               retry: bool) -> tuple[Future, Relay]:
        with self._lock:
            job = self._jobs.get(key)
            created = job is None or (retry and job[0].done() and job[0].exception() is not None)
            if created:
                job = self._jobs[key] = (submit(), relay)
                job[0].add_done_callback(lambda future: self._finished(key, relay, future))
        if created:
            self._report(key[1], force=True)
        return job

    def _finished(self, key: tuple[str, int], relay: Relay, future: Future) -> None:
        if future.exception() is None:
            result = future.result()
            if isinstance(result, tuple):  # (result, seconds), also for work done before a restart
                self._seconds[key] = result[1]
            elif relay.started:
                self._seconds[key] = round(time.time() - relay.started, 1)
        self._report(key[1], force=True)

    def forget(self, c: Contribution, slug: str) -> None:
        """After the sequential lane is done with a contribution: drop its jobs and depth maps
        (a rejected video's were never fused; a depth job still running cleans up itself)."""
        with self._lock:
            for kind in ("ingest", "piece", "depth"):
                job = self._jobs.pop((kind, c.id), None)
        if job is None or job[0].done():
            shutil.rmtree(work_dir(c, slug) / DEPTH_DIR, ignore_errors=True)
