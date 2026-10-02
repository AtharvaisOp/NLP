# MahaPulse FastAPI backend

This directory is the API boundary for the future MahaPulse analysis service.
The current scaffold exposes health and preprocessing-preview endpoints only;
model inference and PostgreSQL persistence are intentionally not wired yet.

## Local run

From the repository root, create a virtual environment, install
`backend/requirements.txt`, copy `.env.example` to `.env`, and run:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is designed to be deployed as a Render web service. Keep the existing
static site independent so its current routes and GitHub Pages behavior remain
unchanged while the API evolves.
