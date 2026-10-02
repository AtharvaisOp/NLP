# MahaPulse FastAPI backend

This directory is the API boundary for the MahaPulse analysis service. The
frontend-facing response contract remains unchanged. Keyword, topic, and
summary services are still deterministic stubs; PostgreSQL is not used.

## Endpoints

- `GET /health` — lightweight process liveness.
- `GET /ready` — API/preprocessing readiness plus model-service states.
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

## Sentiment backend

The default `SENTIMENT_BACKEND=mock` keeps tests and lightweight development
free of model loading. To use a local MuRIL artifact, configure:

```dotenv
SENTIMENT_BACKEND=muril
SENTIMENT_MODEL_PATH=ml/artifacts/sentiment/muril-mahasent-md-smoke-v4
ALLOW_SMOKE_MODEL=true
MODEL_DEVICE=auto
```

The service loads `AutoTokenizer` and `AutoModelForSequenceClassification`
from the local artifact only; request handling never downloads from Hugging
Face. It validates the manifest, canonical label mapping, model/tokenizer
files, preprocessing version, and optional integrity hashes. Models are lazy
loaded once per process, set to evaluation mode, and run under
`torch.inference_mode()`.

`MODEL_DEVICE=auto` selects CUDA only when the installed PyTorch build reports
CUDA availability. Explicit `cuda` configuration reports a not-ready service
when CUDA is unavailable. Smoke artifacts require `ALLOW_SMOKE_MODEL=true`;
when enabled, `/ready` remains degraded and `/v1/model-info` reports
`smoke_test=true` and `production_ready=false`. Smoke predictions are not
project performance and are not production-ready.

To swap in a future full artifact, change only `SENTIMENT_MODEL_PATH`, for
example:

```dotenv
SENTIMENT_MODEL_PATH=ml/artifacts/sentiment/muril-mahasent-md-v1
ALLOW_SMOKE_MODEL=false
```

The API response shape does not change. The classifier receives the shared
`model_text` preprocessing path; `analysis_text` remains reserved for later
keyword/topic services.

### Local smoke measurement

On the local Windows development environment, loading and validating the smoke
artifact took about 14 seconds on the first `/ready` request. After loading,
four sequential `/v1/analyze` requests measured approximately 29–71 ms
end-to-end, with service-level MuRIL inference logs around 24–66 ms. These are
local smoke-development observations, not production latency guarantees.

The API is designed to be deployed as a Render web service. Keep the existing
static site independent so its current routes and GitHub Pages behavior remain
unchanged while the API evolves.
