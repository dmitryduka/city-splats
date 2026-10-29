"""Device-specific inputs: DJI telemetry (dji), Insta360 lens rigs (insta360)."""

from __future__ import annotations

from pathlib import Path


def image_names(images: Path, prefix: str, frame_names: list[str]) -> list[str]:
    """A video's image names (relative to the scene's images dir) for its frames: one per
    frame, or every view of each moment for a rig video (images/<prefix>/rig.json)."""
    from worker.pipeline.devices.insta360 import RIG_FILE, view_names

    if (images / prefix / RIG_FILE).exists():
        return view_names(prefix, frame_names)
    return [f"{prefix}/{name}" for name in frame_names]
