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

The FastAPI app in `backend/app` is the only service boundary for the future
frontend integration. Planned endpoints are:

| Endpoint | Responsibility | Persistence |
| --- | --- | --- |
| `GET /health` | Render liveness check | None |
| `POST /api/v1/analyze` | Validate input, run both text paths and all model stages, persist a result | Write |
| `POST /api/v1/analyze/preview` | Current scaffold: return deterministic preprocessing outputs | None |
| `GET /api/v1/results/{result_id}` | Retrieve one persisted analysis | Read |
| `GET /api/v1/analytics` | Return sentiment/topic/keyword aggregates for dashboard views | Read |
| `GET /api/v1/export` | Stream filtered JSON or CSV result data | Read |

The frontend should send raw text and receive a versioned response containing
the result identifier, sentiment label/probabilities, keywords, topics,
summary, and analytics metadata. The service owns validation, preprocessing,
model execution, persistence, and export formats; the static frontend owns
presentation and user interaction.

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
  app/
    main.py                 FastAPI application entry point
    schemas.py              HTTP request/response contracts
    api/routes.py           Thin health and preprocessing-preview routes
  requirements.txt          Backend runtime dependencies
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
