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
                ├── KeyBERT → keywords
                ├── BERTopic → topics
                └── EDA / analytics inputs

model_text + derived results → summarization → PostgreSQL result record
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
integration. Phase 2 implements this stable contract with deterministic stubs;
later model adapters can replace those stubs without changing the response
shape.

| Endpoint | Responsibility | Current state |
| --- | --- | --- |
| `GET /health` | Lightweight Render liveness check | Implemented; no model dependency |
| `GET /ready` | Report API/preprocessing readiness and model-service states | Implemented; model services are `mocked` |
| `GET /v1/model-info` | Return MuRIL/KeyBERT/BERTopic/summary metadata and labels | Implemented; no models loaded |
| `POST /v1/analyze` | Validate input, run both text paths, call injected services, return stable result | Implemented with deterministic stubs |
| `POST /api/v1/analyze/preview` | Phase 1 preprocessing-preview compatibility route | Preserved |
| `GET /v1/results/{result_id}` | Retrieve one persisted analysis | Future PostgreSQL phase |
| `GET /v1/analytics` | Return sentiment/topic/keyword aggregates | Future PostgreSQL phase |
| `GET /v1/export` | Stream filtered JSON or CSV result data | Future PostgreSQL phase |

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
The default implementations in `services/mocks.py` are deterministic and
report `mocked`/`not-loaded` state; they do not claim model performance.

Input validation rejects blank text and applies `MAX_TEXT_LENGTH`. CORS is
controlled by `ALLOWED_ORIGINS`, and `LOW_CONFIDENCE_THRESHOLD` controls the
response `low_confidence` flag. Request IDs are generated or safely propagated
through `X-Request-ID`, logged as structured JSON, and returned in both normal
responses and safe JSON errors. Stack traces are logged server-side only.

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
    dependencies.py         FastAPI service dependency providers
    api/routes.py           Thin health, readiness, and analysis routes
    services/interfaces.py  Protocols for future real model services
    services/mocks.py       Deterministic no-model stubs
    services/orchestrator.py Two-path analysis orchestration
    errors.py               Centralized safe JSON exception handlers
    middleware.py           Request IDs and structured access logs
  requirements.txt          Backend runtime dependencies
  requirements-dev.txt      Pytest/httpx test dependencies
  tests/                    FastAPI contract tests
ml/
  preprocessing.py          Two-path deterministic preprocessing contract
  tests/                    Dependency-light contract tests
docs/
  ARCHITECTURE.md           System, API, deployment, and mismatch record
assets/, */index.html       Existing Part 1 static site; routes preserved
.env.example                Local/deployment configuration names
```

Future model adapters should be added under `ml/` behind explicit interfaces
for MuRIL, KeyBERT, BERTopic, summarization, and analytics rather than making
the FastAPI routes import model internals directly.
