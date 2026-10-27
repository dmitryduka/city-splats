import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, func, select

from .config import DATA_ROOT, SCENES_ROOT, TRAIN_SETTINGS
from .db import init_db, session_scope
from .models import Contribution, Scene, Version, utcnow

app = FastAPI(title="scenes")
init_db()


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "scene"


def _get_scene(session: Session, slug: str) -> Scene:
    scene = session.exec(select(Scene).where(Scene.slug == slug)).first()
    if scene is None:
        raise HTTPException(404, f"scene {slug!r} not found")
    return scene


def scene_state(session: Session, scene: Scene) -> dict[str, Any]:
    contributions = session.exec(
        select(Contribution).where(Contribution.scene_id == scene.id).order_by(Contribution.seq)
    ).all()
    versions = session.exec(
        select(Version).where(Version.scene_id == scene.id).order_by(Version.number)
    ).all()
    head = next((v for v in versions if v.id == scene.head_version_id), None)
    return {
        "scene": scene.model_dump(mode="json"),
        "head": head.model_dump(mode="json") if head else None,
        "versions": [v.model_dump(mode="json") for v in versions],
        "contributions": [c.model_dump(mode="json") for c in contributions],
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/scenes")
def list_scenes() -> list[dict[str, Any]]:
    with session_scope() as session:
        out = []
        for scene in session.exec(select(Scene).order_by(Scene.created_at)).all():
            counts = dict(
                session.exec(
                    select(Contribution.status, func.count())
                    .where(Contribution.scene_id == scene.id)
                    .group_by(Contribution.status)
                ).all()
            )
            out.append({**scene.model_dump(mode="json"), "counts": counts})
        return out


class SceneCreate(BaseModel):
    name: str


@app.post("/api/scenes", status_code=201)
def create_scene(body: SceneCreate) -> dict[str, Any]:
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "name is required")
    with session_scope() as session:
        slug, n = _slugify(name), 1
        while session.exec(select(Scene).where(Scene.slug == slug)).first():
            n += 1
            slug = f"{_slugify(name)}-{n}"
        scene = Scene(slug=slug, name=name)
        session.add(scene)
        session.commit()
        (SCENES_ROOT / slug).mkdir(parents=True, exist_ok=True)
        return scene.model_dump(mode="json")


@app.get("/api/scenes/{slug}")
def get_scene(slug: str) -> dict[str, Any]:
    with session_scope() as session:
        return scene_state(session, _get_scene(session, slug))


@app.post("/api/scenes/{slug}/videos", status_code=201)
def upload_video(slug: str, file: UploadFile = File(...), device: str = Form("auto")) -> dict[str, Any]:
    with session_scope() as session:
        scene = _get_scene(session, slug)
        seq = (
            session.exec(select(func.max(Contribution.seq)).where(Contribution.scene_id == scene.id)).one()
            or 0
        ) + 1
        filename = Path(file.filename or "video.mp4").name
        contribution = Contribution(
            scene_id=scene.id, seq=seq, filename=filename, video_path="", device=device, status="uploading"
        )
        session.add(contribution)
        session.commit()

        upload_dir = SCENES_ROOT / scene.slug / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        video_path = upload_dir / f"{contribution.id}_{filename}"
        with open(video_path, "wb") as out:
            shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)

        contribution.video_path = str(video_path)
        contribution.status = "queued"
        contribution.updated_at = utcnow()
        scene.updated_at = utcnow()
        session.add(contribution)
        session.add(scene)
        session.commit()
        return contribution.model_dump(mode="json")


class HeadUpdate(BaseModel):
    version_id: int


@app.post("/api/scenes/{slug}/head")
def set_head(slug: str, body: HeadUpdate) -> dict[str, Any]:
    """Revert (or re-apply) by moving HEAD to any existing version of the scene."""
    with session_scope() as session:
        scene = _get_scene(session, slug)
        version = session.get(Version, body.version_id)
        if version is None or version.scene_id != scene.id:
            raise HTTPException(404, "version not found in this scene")
        busy = session.exec(
            select(Contribution).where(
                Contribution.scene_id == scene.id, Contribution.status.in_(["processing", "uploading"])
            )
        ).first()
        if busy:
            raise HTTPException(409, "a video is being processed; wait for it to finish before reverting")
        scene.head_version_id = version.id
        scene.updated_at = utcnow()
        session.add(scene)
        session.commit()
        return scene_state(session, scene)


class TrainRequest(BaseModel):
    quality: str = "full"


def _get_version(session: Session, slug: str, version_id: int) -> tuple[Scene, Version]:
    scene = _get_scene(session, slug)
    version = session.get(Version, version_id)
    if version is None or version.scene_id != scene.id:
        raise HTTPException(404, "version not found in this scene")
    return scene, version


@app.post("/api/scenes/{slug}/versions/{version_id}/train")
def train_version(slug: str, version_id: int, body: TrainRequest) -> dict[str, Any]:
    """Queue splat training for a version (a TRAIN_SETTINGS quality); runs when no video is waiting."""
    if body.quality not in TRAIN_SETTINGS:
        raise HTTPException(400, f"quality must be one of {sorted(TRAIN_SETTINGS)}")
    with session_scope() as session:
        scene, version = _get_version(session, slug, version_id)
        if version.train_status in ("training", "cancelling"):
            raise HTTPException(409, "this version is already training; cancel it first")
        version.train_status, version.train_quality, version.train_auto = "queued", body.quality, False
        version.train_progress, version.train_eta, version.train_detail, version.train_error = 0.0, None, None, None
        version.updated_at = utcnow()
        session.add(version)
        session.commit()
        return scene_state(session, scene)


@app.delete("/api/scenes/{slug}/versions/{version_id}/train")
def cancel_training(slug: str, version_id: int) -> dict[str, Any]:
    with session_scope() as session:
        scene, version = _get_version(session, slug, version_id)
        if version.train_status == "queued":
            version.train_status, version.train_detail = None, "cancelled"
        elif version.train_status == "training":
            version.train_status = "cancelling"  # the worker stops Brush within a few seconds
        version.updated_at = utcnow()
        session.add(version)
        session.commit()
        return scene_state(session, scene)


def _fingerprint(session: Session, scene_id: int) -> tuple:
    scene = session.get(Scene, scene_id)
    latest = session.exec(
        select(func.max(Contribution.updated_at)).where(Contribution.scene_id == scene_id)
    ).one()
    latest_version = session.exec(select(func.max(Version.updated_at)).where(Version.scene_id == scene_id)).one()
    return (scene.updated_at if scene else None, scene.head_version_id if scene else None, latest, latest_version)


@app.get("/api/scenes/{slug}/events")
async def scene_events(slug: str, request: Request) -> StreamingResponse:
    """SSE stream: emits the full scene state whenever it changes (worker writes go via the DB)."""
    with session_scope() as session:
        scene_id = _get_scene(session, slug).id

    async def stream():
        last, idle = None, 0.0
        while not await request.is_disconnected():
            with session_scope() as session:
                fingerprint = _fingerprint(session, scene_id)
                if fingerprint != last:
                    last = fingerprint
                    state = scene_state(session, session.get(Scene, scene_id))
                    yield f"event: state\ndata: {json.dumps(state)}\n\n"
                    idle = 0.0
            await asyncio.sleep(0.5)
            idle += 0.5
            if idle >= 15:
                yield ": keep-alive\n\n"
                idle = 0.0

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# Splat .rad headers/chunks and viewer layers; StaticFiles handles Range requests.
app.mount("/files", StaticFiles(directory=DATA_ROOT), name="files")
