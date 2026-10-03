# MahaPulse FastAPI backend

This directory is the API boundary for the MahaPulse analysis service. The
frontend-facing response contract remains unchanged. MuRIL sentiment is the
required service; keyword, topic, and summary enrichments are independently
selectable and may be mocked or disabled. Phase 6 adds normalized SQLAlchemy
persistence, bounded CSV batches, analytics, and export.

## Endpoints

- `GET /health` — lightweight process liveness.
- `GET /ready` — API/preprocessing readiness plus model-service states.
- `GET /v1/model-info` — model/service metadata and readiness states.
- `POST /v1/analyze` — stable analysis response contract over the two text paths.
- `POST /v1/analyze/batch` — bounded synchronous UTF-8 CSV processing.
- `GET /v1/analyses/{session_id}` — paginated persisted results.
- `GET /v1/analyses/{session_id}/analytics` — deterministic session aggregates.
- `GET /v1/analyses/{session_id}/export?format=csv|json` — safe result export.
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

The default `SENTIMENT_BACKEND=mock` uses the small offline **rule-based demo**
(`rule-demo-v1`), not MuRIL. Whole-word Marathi/English sentiment cues, short-range
negation and contrast words such as `पण`/`but` make results input-dependent.
Unknown wording or balanced cues produce neutral; sarcasm, complex negation and
context are not reliably handled. There are no model downloads or GPU requirements.

The existing `confidence`/`probabilities` API fields hold normalized rule weights
for compatibility, **not calibrated probabilities or model accuracy**. The UI
calls them demo scores; metadata remains `state=mocked`, `production_ready=false`
and explicitly names the lexicon provider. Keywords remain mock fixtures unless
a real provider is configured. A missing optional topic or single-sentence summary
does not mean the API failed. This demo never substitutes for a failing MuRIL service.

To use a local MuRIL artifact, configure:

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

To use the full artifact after lifecycle promotion, configure:

```dotenv
SENTIMENT_MODEL_PATH=ml/artifacts/sentiment/muril-mahasent-md-v1
ALLOW_SMOKE_MODEL=false
```

Full artifacts initially report `production_ready=false`. Promotion requires
complete training, final held-out test evidence, independent reload, SHA-256
integrity and real API integration. See [the validation record](../docs/VALIDATION.md)
and `python -m ml.cli promote-artifact --help`. A loaded non-smoke classifier
alone does not satisfy promotion. Optional disabled topics can keep the overall
`/ready` response degraded even when sentiment is production-ready.

The API response shape does not change. The classifier receives the shared
`model_text` preprocessing path; `analysis_text` remains reserved for later
topic/EDA processing, while KeyBERT deliberately receives the conservative
`model_text` path.

## Optional enrichment services

Install the optional packages only when enabling a real enrichment backend:

```bash
pip install -r ml/requirements-enrichment.txt
```

Configuration is explicit:

```dotenv
KEYWORD_BACKEND=keybert
KEYWORD_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
TOPIC_BACKEND=bertopic
TOPIC_MODEL_PATH=ml/artifacts/topics/bertopic-mahasent-md-v1
SUMMARY_BACKEND=extractive
```

KeyBERT uses the cached [multilingual MiniLM sentence-transformer](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), selected for its documented 50-language coverage including Marathi-compatible use. KeyBERT's [embedding-based extraction API](https://maartengr.github.io/KeyBERT/guides/embeddings.html) is used directly. The
embedding model must already be cached locally; request handling never
downloads it. KeyBERT receives `model_text` so meaningful Marathi phrases,
negation, punctuation context, and Roman code-mixing are not discarded.

BERTopic is corpus-level. Train it offline with
`python -m ml.cli topics train --processed-dir ml/data/processed/mahasent-md`
and load the resulting versioned artifact online. Topic IDs are specific to
that artifact version; BERTopic outlier `-1` is returned as a null topic.

The extractive summary provider selects source sentences deterministically and
does not claim generative behavior. A one-sentence input returns a null summary
with provider `extractive`; a future provider can implement the same
`SummaryService` interface without changing the route or response schema.

Optional enrichment failures are isolated: sentiment remains available,
failed fields become empty/null, and `meta.warnings` identifies the affected
service. Synchronous timeout settings are observability budgets; they do not
create background workers or pretend to cancel a running model call.

`/ready` and `/v1/model-info` distinguish ready, mocked, disabled, and
unavailable states. Loading the current MuRIL smoke artifact still keeps the
overall service degraded and is never production readiness or model
performance.

## Persistence and batches

Production persistence uses PostgreSQL through SQLAlchemy 2.x and psycopg. The
schema is managed with Alembic; no tables are created automatically at API
startup:

```bash
alembic -c backend/alembic.ini upgrade head
alembic -c backend/alembic.ini downgrade base
```

`DATABASE_URL` enables database-backed history, batches, analytics, and export.
Render-style `postgres://` and `postgresql://` URLs are normalized to the
installed `postgresql+psycopg` driver. `/ready` verifies both connectivity and
the required tables; an unmigrated database is reported unavailable.
When it is unset, `/v1/analyze` remains stateless and `/ready` reports the
database as disabled/not-required. `PERSIST_SINGLE_ANALYSIS=false` is the
default; enabling it requires a configured database and persists one session
without changing the response body. Batch/history/analytics/export require a
database and return safe service errors when it is unavailable.

`POST /v1/analyze/batch` accepts a `.csv` multipart upload with a default
`text` column. Use `?text_column=review` for another column. Uploads are
bounded by `MAX_UPLOAD_BYTES` and `MAX_BATCH_ROWS`, parsed as UTF-8 (including
an optional BOM), and processed synchronously through the same
`AnalysisOrchestrator` as single analysis. Rows fail independently; session
status is `completed`, `partial`, or `failed`.
Batch work runs in a thread pool so its synchronous inference/database work
does not block the ASGI event loop. The HTTP layer limits bodies before JSON
and multipart parsing: uploads allow the configured file bytes plus 64 KiB
of multipart overhead, while text requests allow the configured character
limit encoded as JSON plus bounded overhead. Oversized HTTP bodies return
413; bounded uploads or inputs that exceed their content limits return 422.

The normalized schema stores sessions, documents, sentiment, keywords, topics,
and summaries. It does not store weights or embeddings. `STORE_RAW_TEXT=true`
stores the original input for review/export; when false, `original_text` is
null while the documented processed/result metadata may remain. User text is
not written to normal application logs. CSV export protects cells beginning
with `=`, `+`, `-`, or `@` against spreadsheet formula execution and preserves
Marathi UTF-8.
Export also protects formulas after leading whitespace/control characters.
Exception logs record exception classes instead of provider/SQL exception
contents, which can contain user input. CORS requires an explicit list of
frontend origins and accepts either a comma-separated environment value or
a JSON list; wildcard origins are rejected.

On this local Windows CPU environment, the first KeyBERT load took about 13.4
seconds and increased resident memory by about 702 MB. The two requested local
smoke extractions took about 110 ms and 43 ms after loading. These are
development observations, not deployment guarantees; the memory cost is an
important Render sizing constraint when combined with MuRIL.
The extractive summary smoke remained below 1 ms per input in the same local
process and has no model-memory cost.

### Local smoke measurement

On the local Windows development environment, loading and validating the smoke
artifact took about 14 seconds on the first `/ready` request. After loading,
four sequential `/v1/analyze` requests measured approximately 29–71 ms
end-to-end, with service-level MuRIL inference logs around 24–66 ms. These are
local smoke-development observations, not production latency guarantees.

The API is designed to be deployed as a Render web service. Keep the existing
static site independent so its current routes and GitHub Pages behavior remain
unchanged while the API evolves.

Final integration measurements, the actual OpenAPI contract, and remaining
model/deployment gates are recorded in [docs/VALIDATION.md](../docs/VALIDATION.md).
