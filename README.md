# MahaPulse · महापल्स

MahaPulse analyzes Marathi and Marathi–English text with a FastAPI NLP service
and a vanilla HTML/CSS/JavaScript dashboard. It supports single-text analysis,
bounded CSV batches, persisted documents, session analytics, and backend CSV/JSON
downloads. Educational routes, shared navigation, search, and themes remain
available alongside `/analyzer/`.

**Development/demo model gate:** the validated artifact is
`muril-mahasent-md-smoke-v4`, with `smoke_test=true` and
`production_ready=false`. It proves loading, Unicode handling, inference, and
API integration. Its metrics are **not project accuracy**. Full model training
remains pending until complete training and untouched test evaluation produce
a non-smoke artifact. [The validation record](docs/VALIDATION.md) contains actual
runtime/deployment results and outstanding gates.

The selected free cloud demo uses **mock sentiment and mock keywords**,
disabled topics, real extractive summary, and PostgreSQL when configured.
Local smoke MuRIL/KeyBERT checks remain separate from the hosted demo.
The real local pipeline measured about 1.45 GB resident memory; a paid instance
was not selected. Neither hosted mock predictions nor local smoke results
establish model quality.

## Architecture and repository

```text
Static dashboard (Vercel)
  → raw text / UTF-8 CSV over HTTPS
  → FastAPI validation → dual preprocessing → MuRIL sentiment
  → KeyBERT keywords → saved BERTopic transform → extractive summary
  → SQLAlchemy persistence (PostgreSQL)
  → analytics / paginated documents / CSV & JSON export
```

Sentiment is required. Keywords, topics, and summaries are independent optional
services; a failure yields a safe warning and empty/null enrichment while
preserving sentiment. MuRIL failure never silently switches to mocks.
[Architecture](docs/ARCHITECTURE.md), [backend](backend/README.md), and
[ML](ml/README.md) guides describe the boundaries.

| Directory | Purpose |
| --- | --- |
| `backend/app/` | HTTP schemas/routes, service adapters, orchestration, storage |
| `backend/alembic/`, `backend/tests/` | Database migrations and backend tests |
| `ml/`, `ml/tests/fixtures/` | Preprocessing, data, training/evaluation, offline topics, small fixtures |
| `ml/data/`, `ml/artifacts/` | Ignored generated data and model artifacts |
| `analyzer/`, `assets/` | Dashboard and shared vanilla frontend |
| `frontend-dev/` | Deterministic contract mock and frontend tests |
| `docs/`, `scripts/`, `.github/workflows/` | Documentation and lightweight checks |
| Other `*/index.html` | Preserved educational/static routes |

## Single and batch flows

Single mode sends `{"text":"हा मोबाईल खूप चांगला आहे."}` to
`POST /v1/analyze`. The response includes original/prepared text, script ratios,
sentiment probabilities/confidence, keywords, topic, summary, model version,
processing time, and warnings. Single analysis is stateless by default.

CSV mode uploads multipart field `file` to `POST /v1/analyze/batch` with
`text_column` in the **query string**. The backend validates UTF-8 CSV and its
header, analyzes/persists each row, and returns session ID,
`completed`/`partial`/`failed`, total/success/failure counts, time, and version.
An empty text row is a stored row failure. The dashboard fetches server
analytics and bounded document pages; export controls download server files.
It does not run NLP or rebuild aggregates/exports in the browser.

Defaults are 5,000,000 upload bytes, 1,000 rows, a `text` column, and 100 rows
per result page maximum. Batches run synchronously; begin with small batches
on CPU/staging and set request timeouts for the chosen runtime.

## Preprocessing and services

`model_text` applies NFC, control/whitespace cleanup, and URL/email/mention
placeholders while retaining Marathi/Latin mixing, negation, emojis, hashtags,
and punctuation. It is not stop-word filtered or lemmatized. MuRIL uses the
matching saved tokenizer and truncates at the artifact's recorded token limit.

`analysis_text` removes placeholders and joins case-folded Devanagari, Latin,
and numeric tokens for topic analysis/EDA. KeyBERT uses `model_text` to preserve
phrase context; BERTopic uses `analysis_text`. Educational stop-word and lemma
examples are reference techniques, not mandatory classifier preprocessing.

| Service | Implementation |
| --- | --- |
| Sentiment | Fine-tuned `google/muril-base-cased`; local artifact/label/manifest/integrity checks; lazy load |
| Keywords | KeyBERT with cached `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; no request-time download |
| Topics | BERTopic trained offline and transformed online; artifact-specific labels; outlier `-1` becomes null |
| Summary | Deterministic source-sentence extraction; one-sentence input returns null |

Optional services also support `mock`/`disabled`; metadata distinguishes them
from ready or unavailable real providers. Analytics uses successful rows for
sentiment percentages and includes confidence, code mixing, keyword counts and
average scores, topics/null count, and total/success/failure. Aggregate summary
is intentionally unavailable because the API returns null. Confidence is not
correctness or calibrated certainty. MahaBERT/IndicBERT pages remain educational
comparisons; MuRIL is the implemented sentiment adapter.

## Local setup

Use Python 3.11+ and Node.js 20+ from the repository root. Create/activate a
virtual environment with `python -m venv .venv`. Copy `.env.example` to ignored
`.env`; replace the placeholder database URL with
`DATABASE_URL=sqlite:///./mahapulse.db` for a persisted local demo, or leave
`DATABASE_URL` empty for stateless analysis. Retain explicit mock defaults until
a validated artifact is provisioned.

```bash
python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
python -m alembic -c backend/alembic.ini upgrade head
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal, run `python -m http.server 5500 --bind 127.0.0.1` and open
`http://localhost:5500/analyzer/`. Include that exact origin in `ALLOWED_ORIGINS`.
Live API schema/docs are `/openapi.json` and `/docs`. Runtime dependencies
include PyTorch/Transformers; fixture-only tests can use the smaller
`scripts/requirements-ci.txt` instead.

Frontend configuration is centralized in `assets/js/api-config.js` and accepts
`MAHAPULSE_STATIC_CONFIG` or `MAHAPULSE_RUNTIME_CONFIG` before it loads. Only
public settings belong there, including the backend API base URL. See the
[frontend guide](frontend-dev/README.md) for mock usage and browser checks.

## Environment variables

`.env.example` documents names/defaults. Credentials belong in ignored local
environment files or provider secrets.

| Variables | Meaning/default |
| --- | --- |
| `APP_ENV`, `ALLOWED_ORIGINS` | Environment label and exact frontend origins |
| `DATABASE_URL` | PostgreSQL `postgresql+psycopg://…` or local SQLite; empty allows stateless mode |
| `STORE_RAW_TEXT`, `PERSIST_SINGLE_ANALYSIS` | Original storage true / single persistence false |
| `MAX_UPLOAD_BYTES`, `MAX_BATCH_ROWS`, `DEFAULT_TEXT_COLUMN` | 5,000,000 / 1,000 / `text` |
| `MAX_PAGINATION_LIMIT`, `MAX_TEXT_LENGTH` | 100 rows/page / 100,000 characters |
| `SENTIMENT_BACKEND`, `SENTIMENT_MODEL_PATH` | `mock` or `muril`; validated local artifact |
| `ALLOW_SMOKE_MODEL`, `MODEL_DEVICE` | false; `auto`, `cpu`, `cuda` |
| `LOW_CONFIDENCE_THRESHOLD` | 0.60 |
| `KEYWORD_BACKEND`, `KEYWORD_MODEL_NAME` | `mock`, `keybert`, `disabled`; cached embedding model |
| `KEYWORD_TOP_N`, `KEYWORD_NGRAM_MIN`, `KEYWORD_NGRAM_MAX` | 5 / 1 / 2 |
| `KEYWORD_USE_MMR`, `KEYWORD_DIVERSITY`, `KEYWORD_TIMEOUT_SECONDS` | false / 0.5 / 2.0 |
| `TOPIC_BACKEND`, `TOPIC_MODEL_PATH`, `TOPIC_EMBEDDING_MODEL` | `mock`, `bertopic`, `disabled`; topic artifact/matching embeddings |
| `TOPIC_TIMEOUT_SECONDS` | 2.0 |
| `SUMMARY_BACKEND`, `SUMMARY_MAX_SENTENCES`, `SUMMARY_TIMEOUT_SECONDS` | `mock`, `extractive`, `disabled`; 2 / 1.0 |

Enrichment time budgets report elapsed warnings, not cancellation.
`API_HOST`, `API_PORT`, and `MURIL_MODEL_NAME` in the example are descriptive
deployment/training values selected explicitly by Uvicorn/ML commands.
`RESULT_RETENTION_DAYS` is a policy placeholder; no purge job is implemented.

`STORE_RAW_TEXT=false` hides only `original_text` in persisted retrieval/export.
**Prepared text still contains user content and is stored.** It is not
anonymization; stateless responses also return submitted text. The dashboard
shows an intentional omitted-original state.

## API

| Method/path | Input/output |
| --- | --- |
| `GET /health` | Lightweight process liveness |
| `GET /ready` | Ready/degraded plus model/enrichment/database states |
| `GET /v1/model-info` | Provider/version/device, lifecycle, labels, readiness |
| `POST /v1/analyze` | JSON `{text}` → analysis |
| `POST /v1/analyze/batch?text_column=text` | Multipart `file` → persisted batch summary |
| `GET /v1/analyses/{session_id}?limit=50&offset=0` | Session and bounded documents |
| `GET /v1/analyses/{session_id}/analytics` | Server session aggregates |
| `GET /v1/analyses/{session_id}/export?format=csv` | UTF-8 CSV attachment; JSON supported/default |
| `POST /api/v1/analyze/preview` | Compatibility preprocessing preview |

Persisted endpoints need a migrated database. Failures use safe JSON errors and
request IDs; CSV cells are protected from spreadsheet formula injection.
Sessions are retrieved by ID: there is no global history listing or
authentication/multi-tenant ownership boundary. Restrict access to a controlled
demo/staging audience.

## Dataset, training and artifacts

Preparation uses the official [L3Cube MarathiNLP source](https://github.com/l3cube-pune/MarathiNLP),
`L3Cube-MahaSent-MD/MahaSent_All`. Preserve its Train/Val/Test splits and map
`-1/0/1` to `negative=0/neutral=1/positive=2`. `dataset_report.json` records
provenance/revision, filtering, deduplication, split counts and script statistics.
Upstream terms are CC BY-NC-SA 4.0 with research/non-commercial restrictions;
preserve attribution. Local sources need explicit `--local-path`.

```bash
python -m ml.cli prepare
python -m ml.cli train --smoke --model-version muril-mahasent-md-smoke-v4
python -m ml.cli train --full --processed-dir ml/data/processed/mahasent-md --artifact-root ml/artifacts --model-version muril-mahasent-md-v1 --train-batch-size 4 --eval-batch-size 8
python -m ml.cli evaluate --artifact-dir ml/artifacts/sentiment/muril-mahasent-md-v1 --processed-dir ml/data/processed/mahasent-md
python -m ml.cli inspect-artifact ml/artifacts/sentiment/muril-mahasent-md-v1
```

Before full training, verify complete splits, disk, PyTorch CUDA availability,
GPU name and VRAM. Use a separate training environment for CUDA changes and
keep working runtimes intact. Do not blindly start multi-hour CPU training.
Checkpoint selection uses validation data. Only complete non-smoke training
and untouched test evaluation may report accuracy, macro precision/recall/F1,
weighted F1, per-class metrics, and confusion matrix.

Versioned artifacts include tokenizer/model files, label map, training config,
manifest, dataset report, metrics and predictions. Validate hashes and reload
independently, then run an API smoke before promotion. Transfer artifacts via
controlled external storage/build-time download or a persistent deployment disk.
Do not commit weights, data, caches, databases or credentials. A fresh clone
requires separate artifact provisioning for real inference. No training runs
in web-service startup.

BERTopic has a separate offline lifecycle:

```bash
python -m pip install -r ml/requirements-enrichment.txt
python -m ml.cli topics train --processed-dir ml/data/processed/mahasent-md --topic-version bertopic-mahasent-md-v1
```

## Testing and CI

```bash
python -m pip install -r scripts/requirements-ci.txt
python -m pytest -q backend/tests ml/tests
python -m compileall -q backend ml
node scripts/check-static.mjs
node scripts/run-frontend-tests.mjs
git diff --check
```

CI runs fixture backend/ML tests, compile/import, JS/inline-script syntax,
frontend Node tests, static route/link checks, and whitespace checks. It does
not install PyTorch, download weights, or train. Optional browser checks use
`node frontend-dev/tests/browser-smoke.mjs`; unavailable browser support is
reported explicitly and the manual checklist remains required.

## Render and Vercel

Inspect existing resources before creating dedicated MahaPulse resources. Keep
Render PostgreSQL's internal URL in backend `DATABASE_URL` secrets; use its
external URL only for authorized validation from outside Render. Select the
SQLAlchemy psycopg dialect. Apply migrations before serving:

```bash
python -m alembic -c backend/alembic.ini upgrade head
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
```

Use `/health` for liveness, `/ready` and `/v1/model-info` for actual state. Start
with one worker: each worker duplicates model RAM, and MuRIL plus multilingual
embeddings may exceed small instances. Keep optional services lazy and disable
missing topic artifacts. Provision validated model files deterministically.

Deploy the static site on Vercel without a framework migration, publish its
public API URL through frontend config, retain all directory routes, and limit
CORS to exact frontend domains. `vercel.json` runs
`node scripts/build-static.mjs`, publishes allowlisted website files to `dist/`,
and generates public runtime config from `MAHAPULSE_API_BASE_URL`. Backend
source, `.env`, data, caches and artifacts are excluded from that output.

`render.yaml` describes dedicated staging API/database resources with explicit
mock sentiment/keyword defaults, disabled topics and extractive summaries.
`deploy/build.sh` installs the lightweight demo dependencies. Opting into real
MuRIL installs CPU inference dependencies and provisions a model with
`deploy/fetch-model.py` using secret `MODEL_ARTIFACT_URL` and a pinned
`MODEL_ARTIFACT_SHA256`. `deploy/start.sh` applies migrations when a database is
configured and launches one worker. These files prepare deployment; actual
success must be established by runtime checks.

Smoke deployments must visibly show **Development/Smoke Model — Not Production
Ready**, use a demo/staging service name and `ALLOW_SMOKE_MODEL=true`.
Production requires a full artifact, PostgreSQL runtime validation, and deployed
single/batch/export/browser checks. Liveness or passing mock tests alone does
not satisfy these gates. Actual URLs/restart/timing outcomes are documented in
[VALIDATION.md](docs/VALIDATION.md).

## Limitations

Full MuRIL training/evaluation is pending. Real topics require a provisioned
BERTopic artifact. Aggregate summaries, queued/streaming batches,
authentication/ownership, automatic retention, and global history listing are
not implemented. Sentiment truncates to the artifact's tokenizer limit even
within the character limit. Educational examples, mocks and smoke predictions
do not establish model quality.
