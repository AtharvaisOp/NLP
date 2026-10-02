# MahaPulse FastAPI backend

This directory is the API boundary for the MahaPulse analysis service. Phase 2
defines the frontend-facing response contract and wires deterministic stubs;
no Hugging Face model is loaded or downloaded and PostgreSQL is not used.

## Endpoints

- `GET /health` — lightweight process liveness.
- `GET /ready` — API/preprocessing readiness plus mocked model-service states.
- `GET /v1/model-info` — model/service metadata and readiness states.
- `POST /v1/analyze` — stable analysis response contract over the two text paths.
- `POST /api/v1/analyze/preview` — Phase 1 preprocessing-preview compatibility route.

## Local run

From the repository root, create a virtual environment, install
`backend/requirements.txt`, copy `.env.example` to `.env`, and run:

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Run the contract tests from the repository root with:

```bash
pytest -q
```

For test dependencies, install `backend/requirements-dev.txt` as well.

The API is designed to be deployed as a Render web service. Keep the existing
static site independent so its current routes and GitHub Pages behavior remain
unchanged while the API evolves.
