# MahaPulse staging deployment record

Validated 3 October 2026 (Asia/Calcutta). Classification: **DEMO/STAGING READY**.
The user selected the free hosted mock demo. This deployment does not load
MuRIL weights or KeyBERT embeddings. Historical smoke-model checks and the
separate full-model local lifecycle are recorded in
[VALIDATION.md](VALIDATION.md).

## Live application

| Resource | Address / identity |
| --- | --- |
| Frontend | [mahapulse-staging.vercel.app](https://mahapulse-staging.vercel.app/) |
| Analyzer | [Single and CSV dashboard](https://mahapulse-staging.vercel.app/analyzer/) |
| Backend health | [mahapulse-staging-api.onrender.com](https://mahapulse-staging-api.onrender.com/health) |
| API schema | [OpenAPI](https://mahapulse-staging-api.onrender.com/openapi.json) |
| Render web service | `mahapulse-staging-api`, `srv-davvttdg1s2s73c2j8cg` |
| Render PostgreSQL | `mahapulse-staging-db`, `dpg-davvmlfavr4c73dje56g-a`, PostgreSQL 17 |
| Vercel project | `mahapulse-staging`, `prj_kKALODMnIEp5kHitKLz3W2Uhme2x` |

At that deployment validation, both deployments tracked `feat/mahapulse-final`
and `main` had not yet received the integration. The subsequent repository
release is tracked separately in [VALIDATION.md](VALIDATION.md); this record
does not assert that a new model was deployed. Existing unrelated Render
resources were preserved. The connected Vercel deployment tool was unavailable;
the official authenticated Vercel CLI deployed the existing static project.

## Render configuration and PostgreSQL evidence

The web service and dedicated free database use Singapore. The service runs
`sh deploy/build.sh`, followed by `sh deploy/start.sh`. Startup applies
`python -m alembic -c backend/alembic.ini upgrade head` before one Uvicorn
worker binds `0.0.0.0:$PORT`. No training runs at build or startup.

The internal PostgreSQL connection string is held only in Render's secret
environment configuration. External database access remains disabled; no
password or connection string is committed. The hosted read-only SQL connector
cannot reach that private endpoint, so runtime verification used the deployed
backend over its internal database connection.

Deployment logs recorded `Context impl PostgresqlImpl` and
`Running upgrade -> 0001_initial_analysis_schema`. The following then passed
against that PostgreSQL-backed service:

- Connection and migrated-schema readiness.
- CSV batch inserts, including a persisted invalid row.
- Session retrieval with bounded `limit` / `offset` pages.
- Sentiment, confidence, language, keyword and null-topic analytics.
- Backend CSV and JSON exports with Marathi UTF-8 content.

Runtime settings:

| Setting | Value |
| --- | --- |
| `APP_ENV` | `staging` |
| `SENTIMENT_BACKEND` | `mock` |
| `KEYWORD_BACKEND` | `mock` |
| `TOPIC_BACKEND` | `disabled` |
| `SUMMARY_BACKEND` | `extractive` |
| `ALLOW_SMOKE_MODEL` | `false` |
| `STORE_RAW_TEXT` | `false` |
| `ALLOWED_ORIGINS` | `https://mahapulse-staging.vercel.app` |
| `OMP_NUM_THREADS` | `1` |
| Health check | `/health` |

`GET /health`, `GET /ready`, and `GET /v1/model-info` returned HTTP 200.
Readiness is intentionally `degraded`: API, preprocessing, summary and database
are ready; sentiment/keywords are mocked and topics disabled. Model information
and the dashboard explicitly say development mocks are not production ready.
The model version is `mock-v0`; no smoke classifier is deployed on the free tier.

`STORE_RAW_TEXT=false` omits original text from persisted documents. Derived
preprocessed text, extracted phrases and summaries can still contain source
content. This is a storage policy, not a promise to remove all derived text.
The public demo has no user-account/session ownership authorization; use
non-sensitive demonstration inputs.

## Vercel configuration

`vercel.json` builds with `node scripts/build-static.mjs` and publishes only
`dist`. The build copies the existing vanilla site and writes the public
`assets/js/runtime-config.js` from
`MAHAPULSE_API_BASE_URL=https://mahapulse-staging-api.onrender.com`.
Secrets, Python sources, models and datasets are excluded from published output.
Trailing-slash routes and shared navigation/search/theme behavior are retained.
Preview deployment protection was preserved; the staging production alias is
public. No React, Next.js or Vite migration was introduced.

All 16 published HTML routes returned 200, including `/analyzer/`, `/concept/`,
`/workflows/`, `/compare/`, and the educational routes. The static checker also
validated 208 local links/assets, 22 inline scripts, and 23 JavaScript files.

## Deployed end-to-end evidence

Run the reproducible API check:

```bash
python scripts/smoke-api.py https://mahapulse-staging-api.onrender.com
```

The check performs liveness/readiness/model metadata; the four requested Marathi
and code-mixed examples; a long valid input; safe empty/whitespace validation;
CSV upload; persisted pagination; analytics; and backend CSV/JSON exports.
It verifies contract behavior without asserting mocked or smoke-model quality.

The five-row deployed batch returned `partial`, **5 total / 4 successful /
1 failed**. Retrieval returned pages of 2, 2 and 1 documents. Original text was
null for every stored row. Both exports included all five rows, including the
failure. The browser displayed the same session counts, server analytics,
privacy message and failed-row state. Optional unavailable topics did not stop
analysis, persistence or export. Aggregate summary is intentionally unavailable
because analytics returns null; individual extractive summaries remain supported.

Live browser checks covered single output, CSV upload/loading, service states,
partial dashboard, session reopening, export controls, shared search and theme.
An additional 31-row browser upload returned 30 successes and one failure.
The first page displayed rows 1–25; the next page displayed rows 26–31,
including the invalid final row. Reopening the session fetched a fresh bounded
page. Its multi-sentence documents displayed real extractive summaries even
though sentiment and keywords were mocked. Backend-generated files were independently
verified by HTTP; the in-app browser does not expose reliable blob download
events for confirming a file's destination on disk.

A manual backend redeployment without a code change replaced the process
(`dep-db004c1srm7s73djj15g`). After it became live, database readiness returned
ready and the pre-restart five-row session was retrieved successfully. Migrations
were idempotent and persisted sessions survived service replacement.

## Resource observations and remaining gates

Render mock + PostgreSQL process memory measured about **88 MiB** against a
512 MiB limit. Historical local smoke MuRIL + KeyBERT warmed to about
1,248 MiB, with a 1,452 MiB peak. The newly trained full model's isolated local
CUDA API measured about 1,637.61 MiB warm working set and 1,809.28 MiB peak;
its separate reload-plus-API validation process peaked at about 2,410 MiB.
These are local observations, not cloud inference measurements.
One worker and disabled unavailable services are already configured. A larger
instance must be measured before enabling real model inference in the cloud.

The free PostgreSQL database expires **2 November 2026, 00:13 IST**
(`2026-11-01T18:43:33Z`). Preserve/export needed demonstration sessions before
expiry, or explicitly upgrade the dedicated database. Free web-service cold
starts can delay the first request after inactivity.

The non-smoke `muril-mahasent-md-v1` training and untouched held-out evaluation
are now complete locally: test accuracy is 0.801008 and macro F1 0.800593.
The detailed validation record separates these real-model results from this
hosted mock demo. The full artifact passed independent reload, integrity and
real local API gates and is promoted with `production_ready=true`; cloud
production additionally requires
immutable artifact provisioning, a suitable sustained PostgreSQL/runtime
deployment and real deployed end-to-end checks. The existing 512 MiB free
service is not a safe real-model target, remains explicitly mocked and was not
upgraded. Topics are optional and a fitted corpus artifact is not a sentiment
promotion prerequisite.

For a future real deployment, upload model files to immutable external artifact
storage; configure `MODEL_ARTIFACT_URL` and `MODEL_ARTIFACT_SHA256` in Render.
`deploy/fetch-model.py` validates HTTPS, checksum, archive bounds and the model
manifest at build time. Set `SENTIMENT_BACKEND=muril`, the matching model version,
and `ALLOW_SMOKE_MODEL=false` for production. A smoke-only staging deployment
requires explicit `ALLOW_SMOKE_MODEL=true` and visible smoke/non-production UX.
Do not commit weights or run training in the service startup.

`render.yaml` documents the same dedicated resources. Reuse the existing
resources rather than applying another Blueprint that creates duplicates.
