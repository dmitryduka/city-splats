# Notes for coding agents

## Orientation

- `backend/app/` is a small FastAPI app over SQLite (SQLModel: `Scene`, `Contribution`,
  `Version`). It never runs heavy work; it only writes rows that the worker picks up.
- `worker/main.py` polls the database. `worker/runner.py` processes one contribution
  (ingest → own SfM model → link into the scene → layers → dense cloud → publish) and trains
  splats for a version. `worker/prefetch.py` runs ingest, SfM and depth ahead for queued videos
  in subprocesses (`worker/pipeline/piece_job.py`).
- `worker/pipeline/` stages are functions over directories: `ingest`, `sfm` (chunked mapping,
  GPS checks), `pieces` (linking a video into the scene and composing models), `dense`,
  `depth_net`, `tiles` (the dense cloud as an octree of tiles, each with its own LoD), `lod`,
  `cleanup`, `georef`, `splat/brush.py`, `devices/dji.py` (telemetry).
  `register.py` is the older merge path and is no longer called.
- `frontend/src/App.tsx` holds the view state and keyboard shortcuts;
  `viewer/SplatViewer.tsx` renders (three.js + Spark, on-demand rendering);
  `components/ScenePanel.tsx` shows videos, versions and progress.
- All settings are environment variables in `backend/app/config.py`, each with a comment
  explaining the measured reason for its default. Keep that style when adding one.

## Running and checking

- `scripts/dev.sh` runs the backend (:8000), worker and frontend (:5173); logs are in
  `data/logs/`. The backend reloads on change; the worker does not, so restart it to pick up
  worker changes, but only when nothing is processing or training (check the `contribution`
  and `version` tables), or the run is lost.
- Frontend: `cd frontend && npx tsc --noEmit -p .`.
- Python: run scripts with `PYTHONPATH=. .venv/bin/python ...` from the repo root.
- There is no test suite. Verify on real data under `data/scenes/` and look at results (logs,
  metrics in the `version.metrics` JSON, screenshots). Headless Chrome via Playwright needs
  `--use-angle=swiftshader --enable-unsafe-swiftshader` to render WebGL; add `?lod=0.1&pr=0.5`
  to the URL to keep software rendering fast enough.

## Conventions

- Match the surrounding code: short docstrings that say why, comments for non-obvious
  choices (often with the measurement that motivated them), no speculative abstractions.
- Pipeline outputs live under `data/` (never committed). Paths stored in the database are
  relative to `DATA_ROOT`; files are served at `/files/...` (`files_url`).
- Progress reporting goes through `worker/progress.py` / `worker/reporting.py`; ETAs come
  from `data/timing_<device>.json`, learned from earlier runs.
- COLMAP and Brush are built from source in `tools/` with local patches in `infra/patches/`.
  If you change either source tree, update its patch file.
- Expensive steps (SfM, linking, dense, training) take minutes to hours on the M3; prefer
  testing a change on a subset or an existing intermediate output before a full run.
