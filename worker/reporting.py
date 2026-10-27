"""Step reporting shared by the worker (DB-backed Progress) and benchmarks (console).

Pipeline code wraps each blocking COLMAP call in `step(report, name, total)`; whatever
reporter is passed receives real (done, total) counts parsed from COLMAP's log lines.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from typing import ContextManager, Protocol

from worker.colmap_progress import ColmapProgress

STEP_LABELS = {
    "extract": "Extracting features",
    "match_sequential": "Matching within video",
    "match_weak_links": "Re-matching weak links",
    "match_retrieval": "Matching across videos",
    "map": "Mapping",
    "map_global": "Mapping, global (attempt 1)",
    "map_incremental": "Mapping, incremental",
    "map_global_2": "Mapping, global (retry)",
    "map_fisheye": "Mapping, fisheye lens",
    "submodel_merge": "Fitting via submodel (fallback)",
    "gate": "Consistency check",
}


# COLMAP source files whose log lines carry each step's progress (see ColmapProgress.source).
_SOURCES = {
    "extract": "feature_extraction",
    "map_global": "global_mapper|global_pipeline",
    "map_global_2": "global_mapper|global_pipeline",
    "map_fisheye": "global_mapper|global_pipeline",
    "map_incremental": "incremental_pipeline",
}


def log_source(step_name: str) -> str | None:
    if step_name.startswith("match"):
        return "pairing|feature_matching"
    return _SOURCES.get(step_name)


class StepReporter(Protocol):
    def step(self, name: str, total: int | None = None) -> ContextManager[None]: ...


def step(reporter: StepReporter | None, name: str, total: int | None = None) -> ContextManager[None]:
    return reporter.step(name, total) if reporter is not None else nullcontext()


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "–"
    s = int(round(max(seconds, 0)))
    return f"{s}s" if s < 60 else f"{s // 60}m{s % 60:02d}s" if s < 3600 else f"{s // 3600}h{s % 3600 // 60:02d}m"


class ConsoleReporter:
    """Live single-line progress on stdout plus a timing summary (for benchmarks/spikes)."""

    def __init__(self, label: str = ""):
        self.label = label
        self.timings: dict[str, float] = {}

    @contextmanager
    def step(self, name: str, total: int | None = None) -> Iterator[None]:
        started = time.time()
        state = {"done": 0, "total": total, "phase": None}
        title = f"{self.label} {STEP_LABELS.get(name, name)}".strip()

        def render() -> None:
            elapsed = time.time() - started
            done, total_ = state["done"], state["total"]
            if done and total_:
                rate = elapsed / done
                eta = rate * (total_ - done)
                text = f"{done}/{total_}  {rate:.2f}s/it  elapsed {format_duration(elapsed)}  ETA {format_duration(eta)}"
            else:
                text = f"elapsed {format_duration(elapsed)}"
            if state["phase"]:
                text += f"  [{state['phase']}]"
            sys.stdout.write(f"\r\033[K  {title}: {text}")
            sys.stdout.flush()

        def on_progress(done: int, total_: int) -> None:
            state["done"], state["total"] = done, total_
            render()

        def on_phase(phase: str) -> None:
            state["phase"] = phase
            render()

        render()
        with ColmapProgress(on_progress, total_hint=total, on_phase=on_phase):
            yield
        self.timings[name] = time.time() - started
        sys.stdout.write(f"\r\033[K  {title}: done in {format_duration(self.timings[name])}\n")
        sys.stdout.flush()


class FractionReporter:
    """Steps as (fraction, detail) callbacks, for SfM reporting through one line of text
    (a reconstruction run in its own process, worker.pipeline.piece_job): the detail names the
    running steps with COLMAP's counts, the fraction splits the stage roughly between steps."""

    SPANS = {"extract": (0.0, 0.3), "match_sequential": (0.3, 0.4), "match_weak_links": (0.4, 0.45)}
    MAP_SPAN = (0.45, 1.0)

    def __init__(self, relay: Callable[[float, str | None], None]):
        self.relay = relay
        self.fraction = 0.0
        self.active: dict[str, str] = {}
        self.lock = threading.Lock()
        self.map_span, self.prefix = self.MAP_SPAN, ""

    def section(self, index: int | None, count: int | None) -> None:
        """Mapping a long video chunk by chunk: chunk `index` of `count` gets its share of the
        mapping span, and the detail says which chunk (None: back to the whole span)."""
        if index is None or not count:
            self.map_span, self.prefix = self.MAP_SPAN, ""
            return
        start, end = self.MAP_SPAN
        size = (end - start) / count
        self.map_span = (start + size * index, start + size * (index + 1))
        self.prefix = f"Part {index + 1}/{count}: "

    def _show(self, fraction: float) -> None:
        with self.lock:
            detail = " | ".join(self.active.values())
        self.fraction = max(fraction, self.fraction)
        self.relay(self.fraction, detail)

    @contextmanager
    def step(self, name: str, total: int | None = None) -> Iterator[None]:
        mapping = name.startswith("map")
        label = (self.prefix if mapping else "") + STEP_LABELS.get(name, name)
        start, end = self.SPANS.get(name, self.map_span if mapping else (0.0, 0.0))
        state = {"counts": "", "phase": ""}

        def update(fraction: float) -> None:
            with self.lock:
                self.active[name] = " ".join(x for x in (label, state["counts"], state["phase"]) if x)
            self._show(fraction)

        def on_progress(done: int, total_: int) -> None:
            state["counts"] = f"{done}/{total_}"
            update(start + (end - start) * done / max(total_, 1))

        def on_phase(phase: str) -> None:
            state["phase"] = f"· {phase}"
            update(start)

        update(start)
        try:
            with ColmapProgress(on_progress, total_hint=total, on_phase=on_phase, source=log_source(name)):
                yield
        finally:
            with self.lock:
                self.active.pop(name, None)
