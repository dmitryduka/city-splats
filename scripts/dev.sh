#!/usr/bin/env bash
# Run backend (:8000), worker and frontend (:5173) together; Ctrl-C stops all.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p data/logs
trap 'kill 0' EXIT INT TERM

.venv/bin/uvicorn backend.app.main:app --port 8000 --reload --reload-dir backend --timeout-graceful-shutdown 2 >data/logs/backend.log 2>&1 &
.venv/bin/python -m worker.main >data/logs/worker.log 2>&1 &
(cd frontend && npx vite --port 5173) >data/logs/frontend.log 2>&1 &

echo "backend  http://localhost:8000  (data/logs/backend.log)"
echo "worker   (data/logs/worker.log)"
echo "frontend http://localhost:5173  (data/logs/frontend.log)"
wait
