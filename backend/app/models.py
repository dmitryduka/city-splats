from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Scene(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    slug: str = Field(unique=True, index=True)
    name: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    head_version_id: int | None = None
    # Overrides of backend.app.config settings for this scene, by name (scene_setting).
    settings: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class Contribution(SQLModel, table=True):
    """One uploaded video and its processing job."""

    id: int | None = Field(default=None, primary_key=True)
    scene_id: int = Field(index=True, foreign_key="scene.id")
    seq: int  # 1-based order within the scene; drives the per-video color in the viewer
    filename: str
    video_path: str
    device: str = "auto"
    # queued | processing | done | rejected | failed
    status: str = Field(default="queued", index=True)
    stage: str | None = None
    stage_progress: float = 0.0
    progress: float = 0.0
    eta_seconds: float | None = None
    preview_url: str | None = None
    preview_step: int | None = None
    # Live progress: one-line detail ("Matching across videos 20/43") and the stage timeline
    # [{name, status, seconds, estimate, remaining, done?, total?, substeps?}].
    detail: str | None = None
    timeline: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    version_id: int | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utcnow)

    @property
    def image_dir_name(self) -> str:
        return f"c{self.id}"


class Version(SQLModel, table=True):
    """Scene snapshot: reconstruction + viewer layers (immutable) and a splat (derived; it is
    upgraded in place: points -> quick -> full training)."""

    id: int | None = Field(default=None, primary_key=True)
    scene_id: int = Field(index=True, foreign_key="scene.id")
    number: int
    parent_id: int | None = None
    contribution_id: int | None = None
    contribution_ids: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    database_path: str
    reconstruction_path: str
    splat_path: str
    splat_url: str
    dense_url: str = ""
    layers_url: str
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    # dense_url: the dense point cloud (LoD-streamed as small splats), every version has one.
    # splat_url/splat_kind: the latest trained splat ("quick" | "full" | "uniform"), only
    # after training was requested; "" when none. ("points"/"draft" are legacy previews.)
    splat_kind: str = ""
    # Every trained kind's LoD URL, so the viewer can switch between them.
    splats: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    # Splat training job for this version: queued | training | failed (None when idle).
    # Automatic jobs are dropped once the version is no longer HEAD and yield to new videos.
    train_status: str | None = None
    train_quality: str | None = None
    train_auto: bool = False
    train_progress: float = 0.0
    train_eta: float | None = None
    train_detail: str | None = None
    train_timeline: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    train_error: str | None = None
    preview_url: str | None = None
    preview_step: int | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
