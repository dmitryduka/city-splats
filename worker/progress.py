"""Stage progress + ETA reporting, persisted to a DB row (the API streams it): the
Contribution for video processing, the Version's train_* fields for splat training.

A job is a list of stages (ingest, sfm/register, layers, preview; or train, cleanup, lod). Stages can have
substeps (extract, match_sequential, match_retrieval, map, ...) whose real (done, total)
counts come from COLMAP's log lines (worker.colmap_progress). ETAs use the measured rate
within the current substep, and a per-device timing model (seconds per unit of work)
for everything not started yet; the model is updated after every run, so estimates
converge to the actual machine (Mac vs DGX).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.config import DATA_ROOT
from backend.app.db import session_scope
from backend.app.models import Contribution, Version, utcnow
from worker.colmap_progress import ColmapProgress
from worker.reporting import STEP_LABELS, log_source

# Seconds per unit, measured on an M3 MacBook on battery (CPU SfM, Brush on Metal).
# Substep keys are "<stage>.<substep>.<features>".
DEFAULT_RATES = {
    "ingest": 0.8,  # per second of video
    "sfm": 2.3,  # per keyframe (used before substeps are known)
    "register": 2.4,  # per new keyframe (used before substeps are known)
    "layers": 0.02,  # per keyframe
    "dense": 0.45,  # per keyframe (depth network on MPS, fusion, merge, LoD)
    "train.quick": 0.12,  # per training step (brush-cli on Metal, 1024 px)
    "train.full": 0.2,  # per training step (1600 px)
    "train.uniform": 0.07,  # per training step (1024 px, 500k-point dense start)
    "frames.cameras": 0.05,  # per keyframe (PnP)
    "frames.pick": 0.004,  # per 15 fps candidate (sharpness pass over the video)
    "frames.decode": 0.15,  # per frame decoded at full resolution
    "frames.pose": 0.1,  # per extra frame (optical flow + PnP)
    "init.sample": 1.5,  # per dense tile read
    "cleanup": 10.0,  # per run
    "lod": 20.0,  # per run
    "sfm.extract.aliked": 1.3,  # per frame
    "sfm.match_sequential.aliked": 0.3,  # per frame
    "sfm.match_weak_links.aliked": 0.65,  # per pair (LightGlue on CPU)
    "sfm.map.aliked": 1.0,  # per frame
    "register.extract.aliked": 1.3,
    "register.match_sequential.aliked": 0.3,
    "register.match_retrieval.aliked": 12.0,  # per query frame (~10 LightGlue pairs)
    "register.map.aliked": 1.0,  # per new frame
    "register.submodel_merge.aliked": 2.0,
    "register.gate.aliked": 0.1,
    "sfm.extract.sift": 0.8,
    "sfm.match_sequential.sift": 0.75,
    "sfm.map.sift": 0.6,
    "register.extract.sift": 0.8,
    "register.match_sequential.sift": 0.75,
    "register.match_retrieval.sift": 1.5,
    "register.map.sift": 0.6,
    "register.submodel_merge.sift": 1.0,
    "register.gate.sift": 0.1,
}

STAGE_LABELS = {
    "ingest": "Extracting frames",
    "sfm": "Reconstructing (SfM)",
    "register": "Fitting into scene",
    "layers": "Exporting point layers",
    "dense": "Dense point cloud",
    "frames": "Preparing training frames",
    "init": "Sampling dense cloud for initialisation",
    "train": "Training splats",
    "cleanup": "Removing floaters",
    "lod": "Building LoD for streaming",
}


_RATES_LOCK = threading.Lock()


class TimingModel:
    def __init__(self, device: str):
        self.path = DATA_ROOT / f"timing_{device}.json"
        self.rates = dict(DEFAULT_RATES)
        if self.path.exists():
            self.rates.update(json.loads(self.path.read_text()))

    def estimate(self, key: str, units: float) -> float:
        return self.rates.get(key, 1.0) * units

    def record(self, key: str, units: float, seconds: float, alpha: float = 0.5) -> None:
        if units <= 0:
            return
        with _RATES_LOCK:  # prefetch threads record too
            self.rates[key] = (1 - alpha) * self.rates.get(key, seconds / units) + alpha * (seconds / units)
            self.path.write_text(json.dumps(self.rates, indent=2))


@dataclass
class _Part:
    """A stage or substep: estimated seconds, actual seconds once finished, live counts."""

    name: str
    estimate: float
    units: float = 0.0
    actual: float | None = None
    started: float | None = None
    done: int = 0
    total: int = 0
    count_at: float = 0.0  # elapsed seconds at the last count update
    measured_fraction: float | None = None  # for stages that report a fraction (ingest, train)
    detail: str | None = None
    substeps: list[_Part] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        if self.actual is not None:
            return self.actual
        return time.time() - self.started if self.started else 0.0

    def remaining(self) -> float:
        """Seconds left: measured rate when we have counts, else the estimate."""
        if self.actual is not None:
            return 0.0
        if self.substeps:
            return sum(s.remaining() for s in self.substeps)
        elapsed = self.elapsed
        if self.total and self.done >= 2 and self.count_at > 2:
            rate = self.count_at / self.done
            return max(rate * (self.total - self.done) - (elapsed - self.count_at), 1.0)
        if self.measured_fraction and self.measured_fraction > 0.05 and self.count_at > 3:
            return max(self.count_at / self.measured_fraction - elapsed, 1.0)
        if self.started is None:
            return self.estimate
        return max(self.estimate - elapsed, self.estimate * 0.05)

    def fraction(self) -> float:
        if self.actual is not None:
            return 1.0
        if self.started is None:
            return 0.0
        elapsed, remaining = self.elapsed, self.remaining()
        return min(elapsed / max(elapsed + remaining, 1e-3), 0.99)

    def timeline(self) -> dict:
        status = "done" if self.actual is not None else "active" if self.started else "pending"
        out = {
            "name": self.name,
            "status": status,
            "seconds": round(self.elapsed, 1) if self.started else None,
            "estimate": round(self.estimate, 1),
            "remaining": round(self.remaining(), 1) if status != "done" else 0,
        }
        if self.total:
            out |= {"done": self.done, "total": self.total}
        if self.substeps:
            out["substeps"] = [s.timeline() for s in self.substeps]
        return out


Sink = Callable[[dict], None]


def contribution_sink(contribution_id: int) -> Sink:
    def write(fields: dict) -> None:
        with session_scope() as session:
            c = session.get(Contribution, contribution_id)
            if c is None:
                return
            for key, value in fields.items():
                setattr(c, key, value)
            c.updated_at = utcnow()
            session.add(c)
            session.commit()

    return write


def training_sink(version_id: int) -> Sink:
    def write(fields: dict) -> None:
        with session_scope() as session:
            v = session.get(Version, version_id)
            if v is None:
                return
            v.train_progress = fields["progress"]
            v.train_eta = fields["eta_seconds"]
            v.train_detail = fields["detail"]
            v.train_timeline = fields["timeline"]
            v.updated_at = utcnow()
            session.add(v)
            session.commit()

    return write


class Progress:
    def __init__(self, sink: Sink, plan: list[tuple[str, float]], timing: TimingModel):
        self.sink = sink
        self.timing = timing
        self.stages = [_Part(name, estimate) for name, estimate in plan]
        self.stage: _Part | None = None
        self.sub: _Part | None = None
        self.rate_suffix = ""
        self._last_write = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._tick, daemon=True)
        self._ticker.start()

    def has(self, stage: str) -> bool:
        return any(s.name == stage for s in self.stages)

    def _get(self, name: str) -> _Part:
        return next(s for s in self.stages if s.name == name)

    def set_estimate(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._get(stage).estimate = seconds

    def start(self, stage: str, substeps: list[tuple[str, float]] | None = None, rate_suffix: str = "") -> None:
        """substeps: (name, units of work) -> estimated via the timing model."""
        with self._lock:
            part = self._get(stage)
            part.started = time.time()
            self.rate_suffix = rate_suffix
            if substeps:
                part.substeps = [
                    _Part(n, self.timing.estimate(self._rate_key(stage, n), units), units=units)
                    for n, units in substeps
                ]
                part.estimate = sum(s.estimate for s in part.substeps)
            self.stage, self.sub = part, None
        self._write(force=True)

    def _rate_key(self, stage: str, sub: str) -> str:
        return f"{stage}.{sub}.{self.rate_suffix}" if self.rate_suffix else f"{stage}.{sub}"

    def done_ahead(self, stage: str, seconds: float) -> None:
        """Mark a stage that ran ahead of the job (see worker.prefetch) as done."""
        with self._lock:
            part = self._get(stage)
            part.started, part.actual = time.time() - seconds, seconds
        self._write(force=True)

    def update(self, fraction: float, detail: str | None = None) -> None:
        """A measured stage fraction (ingest decode position, training snapshots)."""
        with self._lock:
            if self.stage is None:
                return
            self.stage.measured_fraction = min(fraction, 1.0)
            self.stage.count_at = self.stage.elapsed
            if detail is not None:
                self.stage.detail = detail
        self._write()

    @contextmanager
    def step(self, name: str, total: int | None = None) -> Iterator[None]:
        """A substep of the current stage; counts come from COLMAP's log lines."""
        with self._lock:
            assert self.stage is not None
            sub = next((s for s in self.stage.substeps if s.name == name), None)
            if sub is None:  # unplanned (e.g. fallback): estimate from the timing model
                sub = _Part(name, self.timing.estimate(self._rate_key(self.stage.name, name), total or 1),
                            units=total or 1)
                self.stage.substeps.append(sub)
            sub.started, sub.total = time.time(), total or 0
            self.sub = sub
        self._write(force=True)

        def on_progress(done: int, total_: int) -> None:
            with self._lock:
                sub.done, sub.total, sub.count_at = done, total_, sub.elapsed
            self._write()

        def on_phase(phase: str) -> None:
            with self._lock:
                sub.detail = phase
            self._write()

        try:
            with ColmapProgress(on_progress, total_hint=total, on_phase=on_phase, source=log_source(name)):
                yield
        finally:
            with self._lock:
                sub.actual = sub.elapsed
                stage_name = self.stage.name if self.stage else ""
            self.timing.record(self._rate_key(stage_name, name), sub.units, sub.actual)
            self._write(force=True)

    def counts(self, done: int, total: int) -> None:
        """Counts for the running substep, for work counted in Python rather than parsed from
        COLMAP's log."""
        with self._lock:
            if self.sub is None:
                return
            self.sub.done, self.sub.total, self.sub.count_at = done, total, self.sub.elapsed
        self._write()

    def finish(self) -> float:
        with self._lock:
            assert self.stage is not None
            self.stage.actual = self.stage.elapsed
            self.sub = None
            actual = self.stage.actual
        self._write(force=True)
        return actual

    def close(self) -> None:
        self._stop.set()

    def _tick(self) -> None:
        while not self._stop.wait(1.0):
            self._write()

    def _detail(self) -> str | None:
        """The running substeps (several when mapping attempts run in parallel)."""
        stage = self.stage
        if stage is None:
            return None
        texts = []
        for sub in stage.substeps:
            if sub.started is None or sub.actual is not None:
                continue
            text = STEP_LABELS.get(sub.name, sub.name)
            if sub.total:
                text += f" {sub.done}/{sub.total}"
            if sub.detail:
                text += f" · {sub.detail}"
            texts.append(text)
        return " | ".join(texts) if texts else stage.detail

    def _write(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_write < 1.0:
            return
        with self._lock:
            if self.stage is None:
                return
            self._last_write = now
            done = sum(s.actual or 0.0 for s in self.stages if s.actual is not None)
            current = self.stage.elapsed if self.stage.actual is None else 0.0
            remaining = sum(s.remaining() for s in self.stages)
            overall = (done + current) / max(done + current + remaining, 1e-3)
            fields = dict(
                stage=self.stage.name,
                stage_progress=self.stage.fraction(),
                progress=min(overall, 0.99),
                eta_seconds=remaining,
                detail=self._detail(),
                timeline=[s.timeline() for s in self.stages],
            )
        self.sink(fields)


def rel(path: Path) -> str:
    return path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
