"""Copy a scene's videos and versions into a new scene that shares the original's files (frames,
models, dense tiles, splats are referenced, not copied), e.g. to try another way of training
next to the original. Don't delete the original while the clone uses its files.

  python -m scripts.clone_scene <source-slug> <new-slug> "<new name>"
"""

from __future__ import annotations

import sys

from sqlmodel import select

from backend.app.config import SCENES_ROOT
from backend.app.db import session_scope
from backend.app.models import Contribution, Scene, Version, utcnow

SKIP = {"id", "scene_id"}


def main() -> None:
    source_slug, slug, name = sys.argv[1:4]
    with session_scope() as session:
        source = session.exec(select(Scene).where(Scene.slug == source_slug)).one()
        scene = Scene(slug=slug, name=name)
        session.add(scene)
        session.flush()
        contributions = session.exec(select(Contribution).where(Contribution.scene_id == source.id)
                                     .order_by(Contribution.id)).all()
        versions = session.exec(select(Version).where(Version.scene_id == source.id).order_by(Version.number)).all()

        contribution_ids: dict[int, int] = {}
        for c in contributions:
            copy = Contribution(scene_id=scene.id, **{k: v for k, v in c.model_dump().items() if k not in SKIP})
            session.add(copy)
            session.flush()
            contribution_ids[c.id] = copy.id
        version_ids: dict[int, int] = {}
        for v in versions:
            fields = {k: val for k, val in v.model_dump().items() if k not in SKIP}
            fields |= {
                "parent_id": version_ids.get(v.parent_id),
                "contribution_id": contribution_ids.get(v.contribution_id),
                "contribution_ids": [contribution_ids[i] for i in v.contribution_ids if i in contribution_ids],
                "train_status": None, "train_auto": False, "preview_url": None, "preview_step": None,
            }
            copy = Version(scene_id=scene.id, **fields)
            session.add(copy)
            session.flush()
            version_ids[v.id] = copy.id
        for c in contributions:
            copy = session.get(Contribution, contribution_ids[c.id])
            copy.version_id = version_ids.get(c.version_id)
            session.add(copy)
        scene.head_version_id = version_ids.get(source.head_version_id)
        scene.updated_at = utcnow()
        session.add(scene)
        session.commit()
        print(f"{slug}: {len(contributions)} videos, {len(versions)} versions; head {scene.head_version_id}; "
              f"versions {version_ids}")

    # A video added to the clone links into the scene through its frames, which keep their
    # original names (c<id>/...).
    images = SCENES_ROOT / slug / "images"
    images.mkdir(parents=True, exist_ok=True)
    for frames in (SCENES_ROOT / source_slug / "images").iterdir():
        (images / frames.name).symlink_to(frames.resolve())


if __name__ == "__main__":
    main()
