"""Real progress from COLMAP's blocking calls, read from its own log lines.

pycolmap calls (extract/match/map) block and report nothing, but COLMAP logs per-item
progress through glog to stderr. ColmapProgress temporarily points fd 2 at a pipe, passes
every line through to the original stderr (so logs are unchanged), and turns known progress
lines into (done, total) callbacks:

    feature_extraction.cc] Processed file [34/85]
    pairing.cc]            Processing image [20/43]          (sequential / vocab tree)
    pairing.cc]            Matching block [3/12]             (imported / exhaustive)
    incremental_pipeline]  Registering image #12 (num_reg_frames=150)
    global_mapper.cc]      === Running rotation averaging ===     (phase i of GLOBAL_MAPPER_PHASES)

usage:
    with ColmapProgress(lambda done, total: ..., total_hint=len(images)):
        pycolmap.extract_features(...)
"""

from __future__ import annotations

import os
import re
import sys
import threading
from collections.abc import Callable

_PATTERNS = [
    re.compile(r"Processed file \[(\d+)/(\d+)\]"),
    re.compile(r"Processing image \[(\d+)/(\d+)\]"),
    re.compile(r"Matching block \[(\d+)/(\d+)"),
]
# Imported pairs are matched in blocks; "[k/n]" is logged as block k starts.
_BLOCK = re.compile(r"Processing block \[(\d+)/(\d+)\]")
_REGISTERED = re.compile(r"num_reg_frames=(\d+)")
_PHASE = re.compile(r"=== Running (.+?) ===")
# Global mapper phases in order; progress is reported as phases completed.
GLOBAL_MAPPER_PHASES = [
    "rotation averaging",
    "track establishment",
    "global positioning",
    "iterative bundle adjustment",
    "iterative retriangulation and refinement",
]

# fd 2 is process-global: one capture is shared by every active ColmapProgress. Concurrent
# COLMAP calls (two mapping attempts, SfM run ahead next to the sequential lane) interleave
# their lines, so each listener can take only lines from its own source file (`source`).
_lock = threading.Lock()
_listeners: list[ColmapProgress] = []
_capture: dict = {}


def _pump(read_fd: int, saved_fd: int) -> None:
    buffer = b""
    while chunk := os.read(read_fd, 65536):
        os.write(saved_fd, chunk)
        buffer += chunk
        *lines, buffer = buffer.split(b"\n")
        listeners = tuple(_listeners)
        for raw in lines:
            line = raw.decode("utf-8", "replace")
            for listener in listeners:
                if listener.source is None or listener.source.search(line):
                    try:
                        listener._handle(line)
                    except Exception:  # noqa: BLE001 - progress must never break the pipeline
                        pass


def _start_capture() -> None:
    sys.stderr.flush()
    saved_fd = os.dup(2)
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    thread = threading.Thread(target=_pump, args=(read_fd, saved_fd), daemon=True)
    thread.start()
    _capture.update(saved_fd=saved_fd, read_fd=read_fd, thread=thread)


def _stop_capture() -> None:
    sys.stderr.flush()
    os.dup2(_capture["saved_fd"], 2)  # closes the pipe's last write end -> reader gets EOF
    _capture["thread"].join(timeout=5)
    os.close(_capture["read_fd"])
    os.close(_capture["saved_fd"])
    _capture.clear()


class ColmapProgress:
    def __init__(
        self,
        on_progress: Callable[[int, int], None],
        total_hint: int | None = None,
        on_phase: Callable[[str], None] | None = None,
        source: str | None = None,
    ):
        """total_hint: denominator for lines without one (registered frames during mapping).
        source: regex on the log line (e.g. "incremental_pipeline") selecting this call's lines."""
        self.on_progress = on_progress
        self.total_hint = total_hint
        self.on_phase = on_phase
        self.source = re.compile(source) if source else None

    def _handle(self, line: str) -> None:
        if match := _PHASE.search(line):
            phase = match.group(1).strip()
            if self.on_phase:
                self.on_phase(phase)
            if phase.lower() in GLOBAL_MAPPER_PHASES:
                self.on_progress(GLOBAL_MAPPER_PHASES.index(phase.lower()), len(GLOBAL_MAPPER_PHASES))
            return
        for pattern in _PATTERNS:
            if match := pattern.search(line):
                self.on_progress(int(match.group(1)), int(match.group(2)))
                return
        if match := _BLOCK.search(line):
            done, blocks = int(match.group(1)) - 1, int(match.group(2))
            if self.total_hint:  # as pairs, the unit the step's total is in
                self.on_progress(round(self.total_hint * done / blocks), self.total_hint)
            else:
                self.on_progress(done, blocks)
            return
        if self.total_hint and (match := _REGISTERED.search(line)):
            self.on_progress(min(int(match.group(1)), self.total_hint), self.total_hint)

    def __enter__(self) -> ColmapProgress:
        with _lock:
            if not _listeners:
                _start_capture()
            _listeners.append(self)
        return self

    def __exit__(self, *exc) -> None:
        with _lock:
            _listeners.remove(self)
            if not _listeners:
                _stop_capture()
