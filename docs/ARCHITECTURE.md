# MahaPulse architecture

## Delivered boundaries

The vanilla HTML/CSS/JavaScript site keeps shared navigation, search, themes,
concept pages and directory routes. `/analyzer/` adds single/CSV modes, service
status, session analytics, paginated documents and server-generated downloads.
FastAPI owns NLP, validation, persistence, aggregation and export formats.

```text
Marathi / Marathi-English text or bounded UTF-8 CSV
  → FastAPI validation → shared dual preprocessing
       ├─ model_text → MuRIL sentiment → KeyBERT → extractive summary
       └─ analysis_text → saved BERTopic transform
  → SQLAlchemy session/document persistence (PostgreSQL)
  → analytics / bounded retrieval / CSV and JSON files
  → static dashboard
```

The orchestrator calls sentiment, keywords, topics and summary sequentially.
Sentiment is required; each enrichment has a separate failure boundary.
MuRIL is the implemented classifier. MahaBERT/IndicBERT pages are educational
comparisons rather than deployment configuration.

## Two representations

`ml/preprocessing.py` returns `PreparedText(model_text, analysis_text,
analysis_tokens)` with version `model-text-v1`. `model_text` applies NFC,
control cleanup, URL/email/mention placeholders and collapsed whitespace while
preserving Devanagari, Roman code mixing, negation, emojis, hashtags and
punctuation. MuRIL uses its matching artifact tokenizer; classifier input is
not stop-word filtered, lemmatized or restricted to Devanagari.

`analysis_text` removes placeholders and joins case-folded Devanagari, Latin
and numeric tokens for BERTopic/EDA. KeyBERT candidates use `model_text` to keep
phrase context; extractive summaries also use it. Educational aggressive
cleaning and lemma examples are not the shared runtime preprocessor.

## Services and artifacts

`services/interfaces.py` defines the protocols, `services/factory.py` selects
explicit providers, and `AnalysisOrchestrator` composes them. Real models load
once lazily per process; mocks/disabled services avoid model-package imports.

MuRIL validates project/task, base model, canonical labels, preprocessing/model
version, model/tokenizer files, lifecycle flags and recorded integrity hashes.
Loading is local-only; inference uses `model.eval()` and
`torch.inference_mode()`. Configured failures never switch to mock sentiment.

The historical `muril-mahasent-md-smoke-v4` artifact has `smoke_test=true`,
`production_ready=false`. It requires `ALLOW_SMOKE_MODEL=true` and leaves
readiness degraded; its fixture metrics establish integration only.
The real `muril-mahasent-md-v1` has now completed three epochs on all 47,730
train rows and one final evaluation on the untouched 6,744-row test split.
Validation macro F1 selected epoch-2 checkpoint 23,866; final test accuracy is
0.801008 and macro F1 0.800593. Its full lifecycle gates have passed and its
manifest records `smoke_test=false`, `production_ready=true`. These local
results are separate from smoke evidence and do not imply a cloud deployment.
Full artifacts start with `smoke_test=false`, `production_ready=false`.
Training selects on validation macro F1, then performs a single final held-out
test prediction pass. `ml.cli promote-artifact` verifies required files,
complete SHA-256 coverage, held-out prediction evidence and independent local
reload. Its required `--validation-report` binds a passing prepromotion report
from `scripts/validate-production-model.py` to the exact artifact directory,
version, hashes and manifest. The report exercises real FastAPI startup,
Marathi/code-mixed inference, migrated SQLite persistence, analytics, both
exports, OpenAPI, CORS and request limits. Only after those checks pass does
promotion record all lifecycle gates and
`production_ready=true`. Weights/artifacts/data stay outside Git and need
explicit deployment provisioning. Disabled optional topics keep overall
readiness degraded even when the required sentiment service is ready.

KeyBERT uses cached
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` embeddings.
BERTopic is fitted offline in `ml/topics.py`; online requests only load a saved
artifact and call `transform`. IDs/labels depend on that artifact; outlier `-1`
becomes null. Missing artifacts are unavailable states. No request path
downloads weights or fits a model.

The local `bertopic-mahasent-md-v1` was fitted on all 47,730 train documents
after sentiment promotion and saved outside Git, with 492 non-outlier topic
IDs. Independent offline reload and four-sample backend-provider transformation
passed. Compact safetensors reload uses nearest-topic embedding cosine similarity;
its bounded API `probability` is an assigned-topic score, not calibrated
membership probability. Training-time UMAP/HDBSCAN density/outlier behavior is
not promised after compact reload. The adapter distinguishes compact cosine
score matrices from full membership matrices when an outlier column exists.
Primary sentiment API validation deliberately leaves topics disabled; enabling
the optional provider needs its artifact, cached embeddings and dependencies.

Extractive summary selects source sentences deterministically and returns null
for one-sentence input; no generative API is called. Future summary providers
can implement the same protocol. Enrichment failures preserve other results
and add safe warnings. Time budgets report elapsed work; they are not
cancellation limits or a queue.

## HTTP contract

The executable `/openapi.json` is authoritative.

| Route | Signature/responsibility |
| --- | --- |
| `GET /health` | Process liveness independent of model load |
| `GET /ready` | Overall ready/degraded, service and database state |
| `GET /v1/model-info` | Provider/version/device, labels, lifecycle/readiness |
| `POST /v1/analyze` | JSON `text`; full single result |
| `POST /v1/analyze/batch` | Multipart `file`, optional **query** `text_column` |
| `GET /v1/analyses/{session_id}` | Query `limit=50`, `offset=0`; bounded document page |
| `GET /v1/analyses/{session_id}/analytics` | Successful-row aggregates and session counts |
| `GET /v1/analyses/{session_id}/export` | Query `format=json|csv`, default JSON; attachment |
| `POST /api/v1/analyze/preview` | Compatibility preprocessing-only preview |

Single results include request ID, original/prepared text, language ratios,
sentiment probabilities/confidence/low-confidence flag, keywords, topic,
summary, model version, time and warnings. Service state distinguishes mocked,
ready, disabled, not loaded/not ready and unavailable. Database status comes
from `/ready`; no database is required for default stateless single mode.

## Persistence and batches

SQLAlchemy 2.x defines `analysis_sessions`, `analysis_documents`,
`sentiment_results`, `keyword_results`, `topic_results`, and `summary_results`.
Alembic owns schema changes; application startup does not create tables.
PostgreSQL is the deployment target; SQLite supports deterministic local tests.

Single analysis is stateless unless `PERSIST_SINGLE_ANALYSIS=true`; its response
schema does not change. Persisted routes require a migrated configured database.
UTF-8 CSV accepts a BOM and validates filename extension, size, selected column,
row count and row text. Defaults are 5,000,000 bytes, 1,000 rows, `text` column,
and 100 rows/page maximum. Batches run synchronously with stored per-row success
or failure. Sessions are completed (all succeed), partial (both outcomes), or
failed (no successes). Errors have safe JSON bodies/request IDs.

Analytics uses successful documents for sentiment percentages and includes
confidence average/min/max and low-confidence count, code-mixed count/percentage,
keyword occurrences/average score, topic distribution and null count. No-success
sessions return zero percentages and null confidence aggregates. Aggregate
`summary` is null and the UI shows an unavailable state. The browser does not
calculate competing aggregates.

Results use limit/offset. Backend-generated CSV/JSON preserve UTF-8; formula
prefixes are neutralized for spreadsheets. `STORE_RAW_TEXT=false` hides original
text in persisted retrieval/export, but model/analysis text still contains
content and is stored. It is not anonymization. There is no automatic purge,
global history listing, authentication or per-user session ownership boundary.

## Deployment and gates

```text
Vercel static publish directory
  │ HTTPS; exact-origin CORS
  ▼
Render FastAPI (one worker initially)
  ├─ local validated artifact / cached optional embeddings
  └─ Render PostgreSQL (internal private URL)
```

Environment settings choose services explicitly. Credentials belong in secrets
or ignored `.env`; frontend config holds only public settings such as API URL.
Publish output excludes backend source, models, data, caches and secrets.
Run `python -m alembic -c backend/alembic.ini upgrade head` before serving.
Uvicorn binds `0.0.0.0`/`$PORT`; each extra worker duplicates model memory.
Training is offline and never runs in startup.

The selected free cloud demo explicitly uses mock sentiment/keywords, disabled
topics and real extractive summary with PostgreSQL. The full locally trained
artifact and historical smoke-model validation are separate results; no
full-model deployment is implied. The earlier local model/enrichment process
peaked at 1,452 MiB RSS, beyond the 512 MiB free instance; the user selected the
free mock demo instead of paying for a larger instance. There is no silent
sentiment fallback or promotion of the smoke artifact.

Smoke/mocked services are demos even if liveness is HTTP 200. Smoke readiness
correctly stays degraded. Production requires a full non-smoke artifact,
PostgreSQL runtime checks and deployed end-to-end validation. Actual deployment
URLs, hardware diagnosis, timings and blockers live in
[VALIDATION.md](VALIDATION.md); config alone does not establish deployment.

## Reproducibility and quality

Dataset preparation selects official L3Cube MahaSent-MD `MahaSent_All`
Train/Val/Test files and records revision, label/schema mapping, validation
removals, duplicates and splits. Terms are research/non-commercial CC BY-NC-SA
4.0. Provided splits stay separate for validation selection and untouched test
evaluation. Explicit local input lacking original split markers uses recorded
deterministic stratified splitting.

Versioned sentiment artifacts contain model/tokenizer files, training config,
label map, manifest, data report, metrics and predictions. Smoke metrics prove
plumbing only. A complete run records accuracy, macro precision/recall/F1,
weighted F1, per-class metrics and confusion matrix, then needs independent
reload and real API smoke before promotion.

CI runs fixture backend/ML tests, Python compile/import, JS and inline-script
syntax, frontend Node/DOM guards, local routes/links and Git whitespace checks.
It installs no deep-learning runtime, downloads no weights and trains nothing.
Real-model, PostgreSQL and live-browser verification are separate runtime gates.
