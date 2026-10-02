# MahaPulse architecture

## Scope and invariants

MahaPulse has two intentionally separate parts:

1. **Part 1 — existing static site.** The current vanilla HTML/CSS/JS site is
   preserved as-is, including its generated navigation, page-local scripts,
   interactive workflows, search, theme switching, and every existing route.
2. **Part 2 — analysis service.** A Python FastAPI service will provide the
   NLP pipeline and persistence boundary. It is scaffolded here without
   changing the Part 1 UI or migrating it to React/Next.

The target pipeline is:

```text
Marathi / Marathi-English input
        │
        ├── model_text: NFC + conservative whitespace/noise normalization
        │       └── MuRIL sentiment classifier → Positive / Negative / Neutral
        │
        └── analysis_text: deeper cleaning + tokens
                ├── BERTopic artifact transform → topics (optional)
                └── EDA / analytics inputs

model_text → KeyBERT candidate generation (optional; preserves useful phrases)

model_text + derived results → provider-isolated summarization → PostgreSQL result record
PostgreSQL aggregates → analytics/dashboard API → export (JSON / CSV)
```

`model_text` must preserve sentiment-bearing context: Devanagari, Roman
code-mixing, negation, hashtags, emojis, and useful punctuation. It is NFC
normalized and has obvious control/noise boundaries normalized, but it is not
stop-word filtered, lemmatized, or aggressively stripped before MuRIL.

`analysis_text` is a separate, deeper-cleaned/tokenized representation for
KeyBERT, BERTopic, and exploratory analysis. The shared preprocessor does not
silently remove stop words or lemmatize this path either; those are explicit,
downstream analysis choices and must never leak into the classifier path.

## API boundaries

The FastAPI app in `backend/app` is the only service boundary for frontend
integration. The response contract is stable. The sentiment service can now
select either the explicit `mock` implementation or a local MuRIL artifact;
KeyBERT, BERTopic, and summarization are independently selectable optional
services with mock and disabled modes retained for lightweight development.

| Endpoint | Responsibility | Current state |
| --- | --- | --- |
| `GET /health` | Lightweight Render liveness check | Implemented; no model dependency |
| `GET /ready` | Report API/preprocessing readiness and model-service states | Implemented; reflects mock, ready, or not-ready MuRIL state |
| `GET /v1/model-info` | Return MuRIL/KeyBERT/BERTopic/summary metadata and labels | Implemented; real artifact metadata when configured |
| `POST /v1/analyze` | Validate input, run both text paths, call injected services, return stable result | Implemented with configurable mock or MuRIL sentiment |
| `POST /api/v1/analyze/preview` | Phase 1 preprocessing-preview compatibility route | Preserved |
| `POST /v1/analyze/batch` | Bounded synchronous CSV analysis and persistence | Implemented; database required |
| `GET /v1/analyses/{session_id}` | Paginated persisted session documents | Implemented; database required |
| `GET /v1/analyses/{session_id}/analytics` | Deterministic successful-result aggregates | Implemented; database required |
| `GET /v1/analyses/{session_id}/export` | Session JSON/CSV export | Implemented; database required |

The analysis response always contains `request_id`, original and prepared text,
language ratios/code-mixing metadata, sentiment label/confidence/probabilities,
keywords, topic, summary, and processing metadata/warnings. The frontend sends
raw text; the service owns validation, preprocessing, model execution,
persistence, and export formats; the static frontend owns presentation and
user interaction.

### Service boundaries

`backend/app/services/interfaces.py` defines Protocols for
`SentimentService`, `KeywordService`, `TopicService`, and `SummaryService`.
`AnalysisOrchestrator` receives those interfaces through dependency injection.
`services/factory.py` selects each explicit backend independently. The MuRIL
implementation in `services/sentiment/muril.py` validates and lazily loads one
local artifact per process. KeyBERT, BERTopic, and summary implementations are
lazy or local deterministic services; `services/mocks.py` remains available for
lightweight tests.

Sentiment is required for `/v1/analyze`. Keyword, topic, and summary calls are
separate optional boundaries. Each failure leaves sentiment and successful
enrichments intact, returns `[]` or null for the failed component, and adds a
safe service-specific warning. The synchronous timeout settings are elapsed
time budgets for observability, not a background queue or cancellation system.

KeyBERT uses the cached multilingual
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
embedding model, selected because its model documentation covers 50 languages
and it is already compatible with the repository's local sentence-transformer
cache. KeyBERT's [embedding-based extraction API](https://maartengr.github.io/KeyBERT/guides/embeddings.html) is used directly. Keyword
candidate generation uses `model_text` to preserve meaningful Marathi and
Marathi-English phrases. The optional embedding package/model is never loaded
in mock mode and is never downloaded on a request path.

BERTopic is corpus-level, so training lives in `ml/topics.py` and the CLI:
`python -m ml.cli topics train ...`. The online service only loads a saved
`ml/artifacts/topics/<topic_version>/` artifact and calls `transform`. Topic
IDs and stored labels are artifact-version-specific; outlier topic `-1` is
represented as a null result. No topic model is fitted per request.

Summary providers implement the same `SummaryService` protocol. The current
extractive provider selects source sentences deterministically and returns
null for one-sentence inputs; it does not fabricate or call a generative API.
This isolates a future LLM provider without changing routes or schemas.

The local CPU KeyBERT smoke added approximately 702 MB resident memory and
took about 13.4 seconds to load; two post-load extractions took about 110 ms
and 43 ms. This is a development observation and an explicit Render sizing
risk when MuRIL and BERTopic are loaded in the same process.

MuRIL artifact loading is local-only and validates project/task, the
`google/muril-base-cased` base model, canonical labels, preprocessing version,
model version, tokenizer/model files, smoke policy, and recorded integrity
hashes. `model_text` is passed to the tokenizer with the artifact's recorded
maximum length; `analysis_text` is never sent to MuRIL. Real inference uses
the selected device, `model.eval()`, and `torch.inference_mode()`.

`SENTIMENT_BACKEND=mock|muril`, `SENTIMENT_MODEL_PATH`,
`ALLOW_SMOKE_MODEL`, and `MODEL_DEVICE=auto|cpu|cuda` control the backend.
Smoke artifacts can be enabled only explicitly and report
`smoke_test=true`, `production_ready=false`; readiness remains degraded even
when the smoke service is operational. A failed configured MuRIL service does
not fall back to mock sentiment.

The local CPU smoke integration measured approximately 24–66 ms for the
service-level MuRIL inference after the one-time artifact load. This is a
development observation, not a production performance claim.

Input validation rejects blank text and applies `MAX_TEXT_LENGTH`. CORS is
controlled by `ALLOWED_ORIGINS`, and `LOW_CONFIDENCE_THRESHOLD` controls the
response `low_confidence` flag. Request IDs are generated or safely propagated
through `X-Request-ID`, logged as structured JSON, and returned in both normal
responses and safe JSON errors. Stack traces are logged server-side only.

## Phase 6 persistence and batch boundary

Persistence is an explicit SQLAlchemy 2.x/Alembic layer under
`backend/app/db` and `backend/app/storage`. PostgreSQL is the production target;
temporary SQLite is supported for deterministic local tests. The initial
migration creates normalized `analysis_sessions`, `analysis_documents`,
`sentiment_results`, `keyword_results`, `topic_results`, and `summary_results`
tables with session, label, timestamp, and topic indexes. Application startup
does not create tables; deployment runs `alembic upgrade head`.

`POST /v1/analyze` remains stateless by default. `PERSIST_SINGLE_ANALYSIS=true`
opts into a single session/document write without changing its response schema.
`POST /v1/analyze/batch` requires a configured database, accepts bounded UTF-8
CSV, and processes each row through the existing `AnalysisOrchestrator`.
Individual failures are stored safely and produce `completed`, `partial`, or
`failed` session status. Batch/history/analytics/export endpoints return safe
database service errors if the configured database is unavailable.

`GET /v1/analyses/{session_id}` provides bounded pagination;
`/analytics` calculates percentages against successful documents only, and
`/export` supports JSON and UTF-8 CSV. Null topics are counted separately,
empty keywords do not affect aggregates, and zero-success sessions return null
confidence aggregates with zero percentages. CSV cells beginning with
spreadsheet formula prefixes are apostrophe-prefixed before export.

`STORE_RAW_TEXT=false` omits only `original_text`; model/analysis paths and
result metadata remain available as documented for review. Weights and
embeddings are never stored. Normal logs contain request/session identifiers,
not full user text. Uploads and generated database/export files are ignored by
Git.

## Deployment topology

```text
Vercel (future static frontend)
        │ HTTPS / JSON
        ▼
Render Web Service (FastAPI)
        │ private/managed connection
        ▼
Render PostgreSQL (persistent analysis results and aggregates)
```

The existing static site can continue to be served independently while the
FastAPI service is developed and deployed on Render. Model checkpoints and
runtime configuration belong to the backend deployment, not to the static
frontend. Production CORS origins, database URLs, and model identifiers must
be supplied as deployment environment variables; `.env.example` documents the
names only.

## Audit findings and report/UI mismatches

The repository audit was completed before scaffolding. The current tree is a
static knowledge repository with HTML entry points, shared CSS, shared
JavaScript components, inline page scripts, and favicon assets. It has no
Python service, dependency manifest, database layer, model artifact, or
automated test runner. The current static route set is:

`/`, `/404.html`, `/ai-usage/`, `/applications/`, `/category/`, `/compare/`,
`/concept/`, `/design-system/`, `/references/`, `/reflection/`,
`/repository/`, `/research/`, `/sustainability/`, `/team/`, and `/workflows/`.

The existing `MahaPulse_NLP_Topics.md`, landing page, concept data, and
workflow infographic describe an eight-step educational pipeline that ends in
MahaBERT/IndicBERT and includes stop-word removal, lemmatization, removal of
Roman/mixed-script content, and aggressive noise/punctuation removal. The
canonical implementation target instead uses MuRIL for sentiment and adds
KeyBERT, BERTopic, summarization, analytics/dashboard, and export. Those
materials are documentation/infographic mismatches to reconcile in a later
UI/content pass; this commit intentionally does not rewrite them.

## Directory plan

```text
backend/
  __init__.py              Backend package marker
  app/
    main.py                 FastAPI application entry point
    config.py               Environment-backed runtime settings
    schemas.py              HTTP request/response contracts
    db/                     SQLAlchemy engine and normalized models
    storage/                Repository, persistence, and batch boundaries
    dependencies.py         FastAPI service dependency providers
    api/routes.py           Thin health, readiness, and analysis routes
    services/interfaces.py  Protocols for future real model services
    services/mocks.py       Deterministic no-model stubs
    services/factory.py      Configurable service assembly
    services/sentiment/      Validated local MuRIL inference service
    services/enrichment/     Optional KeyBERT, BERTopic, disabled, and extractive services
    services/orchestrator.py Two-path analysis orchestration
    errors.py               Centralized safe JSON exception handlers
    middleware.py           Request IDs and structured access logs
  alembic.ini               Migration configuration
  alembic/                  Versioned database migrations
  requirements.txt          Backend runtime dependencies
  requirements-dev.txt      Pytest/httpx test dependencies
  tests/                    FastAPI contract tests
ml/
  preprocessing.py          Two-path deterministic preprocessing contract
  dataset.py                Verified MahaSent-MD acquisition, validation, and manifests
  config.py                 Centralized MuRIL training configuration
  reproducibility.py        Seeds and runtime/package provenance
  topics.py                 Offline BERTopic training and artifact inspection
  metrics.py                Held-out evaluation and prediction records
  train.py                  Lazy-dependency MuRIL Trainer pipeline
  evaluate.py               Artifact evaluation on the held-out test split
  cli.py                    prepare/train/evaluate/topics/inspect commands
  requirements.txt          Optional training dependencies
  tests/                    Dependency-light pipeline tests and Marathi fixtures
ml/data/                    Ignored raw and processed dataset outputs
ml/artifacts/               Ignored versioned tokenizer/model artifacts
docs/
  ARCHITECTURE.md           System, API, deployment, and mismatch record
assets/, */index.html       Existing Part 1 static site; routes preserved
.env.example                Local/deployment configuration names
```

Future model adapters should be added under `ml/` behind explicit interfaces
for MuRIL, KeyBERT, BERTopic, summarization, and analytics rather than making
the FastAPI routes import model internals directly. Optional enrichment
dependencies are listed separately in `ml/requirements-enrichment.txt` so the
default test/API installation stays lightweight.

## Phase 3 dataset and training boundary

The reproducible preparation command is `python -m ml.cli prepare`. With no
`--local-path`, it clones only the verified L3Cube MarathiNLP upstream source
and selects `L3Cube-MahaSent-MD/MahaSent_All`. The inspected official files are
`MahaSent_All_Train.csv`, `MahaSent_All_Val.csv`, and `MahaSent_All_Test.csv`,
with `Unnamed: 0,text,label` columns and raw labels `-1`, `0`, and `1`.
Those authoritative splits are preserved. For an explicit local source without
all three split markers, the pipeline uses deterministic stratified 80/10/10
splitting and records the seed and strategy.

The upstream README identifies CC BY-NC-SA 4.0 and research-only,
non-commercial share-alike attribution terms. Each preparation writes ignored
raw/processed output, split manifests, canonical `negative=0`, `neutral=1`,
`positive=2` mapping, and a `dataset_report.json` containing provenance,
revision, filenames, timestamp, schema mappings, validation removals,
duplicates, label counts, and Devanagari/Latin statistics. The pipeline refuses
ambiguous CSV groups and does not silently substitute another dataset.

MuRIL training is deliberately separate from FastAPI. The intended checkpoint
`google/muril-base-cased` was verified against the MuRIL model evaluated in the
L3Cube MahaSent-MD research paper. `ml/train.py` uses
`AutoTokenizer`, `AutoModelForSequenceClassification`, and Hugging Face
`Trainer` with the verified `google/muril-base-cased` checkpoint. It consumes
the shared conservative `model_text` path, never stopword-removing or
lemmatizing classifier input. `python -m ml.cli train --smoke` is a tiny
plumbing check and writes `smoke_test=true`; `--full` uses the prepared train
split, validation-based early stopping, and held-out test evaluation. Neither
mode is connected to `/v1/analyze` in this phase.

Versioned artifacts are written under ignored
`ml/artifacts/sentiment/<model_version>/` with tokenizer/model files,
`training_config.json`, `label_mapping.json`, `model_manifest.json`,
`dataset_report.json`, `metrics.json`, and prediction records. The manifest
records the model source, dataset revision, split strategy, preprocessing
version, seed, runtime package versions, and whether the artifact is smoke-only.
Phase 4 may load this artifact behind the existing service protocol without
changing the frontend-facing API schema.
