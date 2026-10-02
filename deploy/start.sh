#!/usr/bin/env sh
set -eu
# Migrations are independent of model training and loading.
if [ -n "${DATABASE_URL:-}" ]; then
  python -m alembic -c backend/alembic.ini upgrade head
fi
exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
