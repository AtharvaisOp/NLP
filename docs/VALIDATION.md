# MahaPulse final integration validation

Recorded on **3 October 2026 (Asia/Calcutta)**. These observations describe
this Windows development machine. They do not claim deployment performance
or model quality.

## Readiness and model gate

The only validated classifier is `muril-mahasent-md-smoke-v4`, based on
`google/muril-base-cased`. Its manifest has `smoke_test=true`; the live API
correctly reports `production_ready=false` and overall readiness `degraded`.
It was trained on the local fixture, not the full MahaSent-MD corpus. No
accuracy, precision, recall, F1, or confusion-matrix values from this smoke
artifact are project performance.

**FULL MODEL TRAINING PENDING.** A complete non-smoke
`muril-mahasent-md-v1` artifact and untouched test evaluation remain required
before production classification. Deployment/runtime evidence must also
include PostgreSQL and deployed end-to-end checks; see
[DEPLOYMENT.md](DEPLOYMENT.md) for the final cloud record.

## Actual API contract

Inspected `create_app().openapi()` and the live `/openapi.json`; implemented
routes are:

| Method | Route | Request contract |
| --- | --- | --- |
| GET | `/health` | Process liveness; no model required |
| GET | `/ready` | Service states, smoke/production flags, database connectivity/schema |
| GET | `/v1/model-info` | Model and enrichment metadata/readiness |
| POST | `/v1/analyze` | JSON `{ "text": "…" }` |
| POST | `/v1/analyze/batch` | Multipart field `file`; optional query `text_column` |
| GET | `/v1/analyses/{session_id}` | Query `limit` (default 50), `offset` (default 0) |
| GET | `/v1/analyses/{session_id}/analytics` | Persisted session aggregates |
| GET | `/v1/analyses/{session_id}/export` | Query `format=csv` or `format=json` |

The configured pagination maximum defaults to 100 despite the absolute
OpenAPI upper bound of 1,000. Upload limits default to 5,000,000 file bytes
and 1,000 rows. Analytics directly returns sentiment count/percentage,
confidence average/minimum/maximum and low-confidence count, code-mixed
count/percentage, keyword count/average score, assigned topics, null-topic
count, and total/successful/failed counts. Aggregate `summary` is intentionally
`null`; the dashboard must not invent one. Error responses use the documented
safe `ErrorResponse` envelope.

## Lightweight verification

`python -m pytest backend/tests ml/tests -q`: **66 passed, zero failures**
(52 backend tests, 14 ML tests). The installed Starlette emits one TestClient
deprecation warning concerning its httpx integration; it does not affect the
checks. Tests do not download model weights.

`python -m compileall -q backend ml` and `git diff --check` passed. Tests cover
the contract, dual preprocessing, mocked/injected real services, artifact
labels/integrity paths, empty and overlong input, enrichment isolation,
SQLite persistence, row failure, pagination, analytics, UTF-8 exports,
privacy policy, request bounds, CSV formulas, safe logs, and CORS/Render URL
configuration.

## Real local API matrix

The independently loaded smoke artifact served real FastAPI inference on:

- `हा मोबाईल खूप चांगला आहे.`
- `ही सेवा अत्यंत खराब आहे.`
- `आज दुकान सकाळी दहा वाजता उघडले.`
- `हा phone चांगला आहे पण battery backup खराब आहे.`
- A multi-sentence extractive-summary input.
- A valid approximately 4,400-character input truncated only at the model's
  configured token window, while API preprocessing remained complete.

Every valid request returned 200 with canonical labels and probabilities
summing to one. Empty input, whitespace-only input, and 100,001-character
input returned safe 422 responses. All four named sentiment examples yielded
the smoke classifier's `positive` label with low confidence. This demonstrates
the fixture model's unreliability and is not a successful semantic-quality
evaluation.

The small real-model CSV contained the four examples plus a deliberately
blank row. It returned `partial`, total 5, successful 4, failed 1. Three
bounded retrieval requests returned pages of 2, 2, and 1 documents, retaining
the failed row. Analytics counted all session outcomes and one successful
code-mixed row. Backend-generated CSV and JSON exports returned 200 and
preserved Marathi UTF-8. `STORE_RAW_TEXT=false` is also covered by regression
tests; original text becomes null, while the documented processed/results
metadata may remain.

SQLite `alembic upgrade head` was run before this matrix. Local smoke/browser
API uses one worker and an ignored `exports/local-demo.db`; schema creation
does not occur automatically at web startup.
A subsequent live `scripts/smoke-api.py http://127.0.0.1:8000` check enabled
real KeyBERT alongside MuRIL and extractive summary. It passed the complete
matrix after an API restart: the same 4-success/1-failure batch took 235 ms,
keywords persisted into analytics, and CSV/JSON each exported five rows.

## Enrichment evidence

Cached real KeyBERT loaded
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. A real
extraction initially exposed the default vectorizer splitting Marathi
combining marks into syllable fragments. The final service uses the shared
Unicode-safe tokenizer for candidate n-grams and still embeds conservative
`model_text`. Revalidation returned intact phrases such as `हा मोबाईल`,
`मोबाईल`, `चांगला आहे`, and `battery backup`.
After the tokenizer fix, the two extractions measured 98 ms and 34 ms.
Offline BERTopic's vectorizer uses the same Unicode-safe candidates, and its
artifact provenance reads the actual prepared dataset metadata key.

BERTopic is **unavailable** locally: no fitted corpus artifact exists and the
package is not installed in the working runtime. The API returned null topic
with `Topic enrichment unavailable.` while sentiment, summary, persistence,
analytics, and export still worked. Assigned-topic and outlier/null behavior
are tested with fixtures. No real BERTopic latency or quality claim is made.

The real extractive summarizer returned deterministic source sentences on a
multi-sentence input and intentional null summary for single sentences.
Independent optional keyword/topic/summary failures are tested and do not
turn required sentiment inference into a system failure.

## Local resource observations

First measurement, using the working CPU PyTorch environment and one model
process (file/system caches influence subsequent starts):
The single/batch latency measurements below used real MuRIL and extractive
summary with keywords disabled; KeyBERT load/extraction was measured separately
in the same process. The browser API enables real KeyBERT as well.

| Observation | Measured value |
| --- | --- |
| App construction including streamed artifact integrity | 2.32 s |
| Cold `/ready` and MuRIL load | 11.12 s |
| Lightweight health request | 7.4 ms |
| First sentiment HTTP request | 58 ms |
| Warm short sentiment HTTP requests | 19–24 ms |
| Long valid input HTTP request | 134 ms |
| Real smoke batch, 4 successes + 1 failure | 130 ms |
| KeyBERT cached load | 2.01 s |
| KeyBERT two real extractions | 218 ms / 44 ms |
| Extractive summary average over 100 calls | 0.086 ms |
| Mock batch with 100 persisted rows | 112 documents/s |
| Resident memory after lazy MuRIL load | 508 MiB |
| Resident memory after warm MuRIL + KeyBERT | 1,248 MiB |
| Peak working set in this process | 1,452 MiB |

The model weights are memory-mapped, so initial resident memory is lower than
warm working memory. A 512 MiB Render process is insufficient for real
MuRIL plus KeyBERT. One worker with a larger plan must be validated under
actual Linux staging load; optional services can be disabled to reduce
memory. No alternative classifier, quantization, or ONNX substitution was
introduced. Streamed hashing avoids a weights-sized (~950 MB) extra buffer
at startup.

## Full-data and GPU feasibility

Hardware inspection found NVIDIA GeForce RTX 3050 Laptop GPU, 6,144 MiB VRAM,
driver 596.36. NVIDIA-SMI advertises driver CUDA capability 13.2. The working
Python 3.13 environment contains `torch 2.13.0+cpu`, whose own
`torch.version.cuda` is null and `torch.cuda.is_available()` is false. Driver
capability alone does not establish a CUDA-enabled PyTorch runtime.

An isolated environment under the user's Codex directory was created with
system-site-package access; the working interpreter/packages were preserved.
The official `torch 2.11.0+cu128` wheel installed successfully there. In that
environment, `torch.version.cuda` is `12.8`,
`torch.cuda.is_available()` is **true**, and the GPU reports
6,441,926,656 bytes of total VRAM. The working Python remains
`torch 2.13.0+cpu` with CUDA unavailable. Official wheel guidance:
[PyTorch installation commands](https://pytorch.org/get-started/previous-versions/).

A bounded feasibility check loaded the cached MuRIL base checkpoint and ran
only three training optimizer steps against train-split examples, batch size
4, in a temporary process. It created no trained artifact and used no test
examples. The model contains 237,558,531 parameters. Steps at sequence lengths
32, 19, and 256 took 0.807 s, 0.155 s, and 0.266 s respectively. Peak CUDA
allocation was 4,795,203,584 bytes and peak reservation 5,016,387,584 bytes.
The three-step median projects approximately **2.65 hours for three full
epochs before validation/test evaluation and checkpoint I/O**; this is a
rough resource estimate, not a promised completion time.

**Decision: full training remains pending.** CUDA and a batch-size-4 memory
check now work. The resource assessment identified a dedicated multi-hour
training/evaluation job, which was deferred during final delivery.
No full CPU job was started,
no `muril-mahasent-md-v1` artifact exists, and no full evaluation metrics are
reported. The safe isolated environment can be used for that later run;
its executable is
`C:\Users\athar\.codex\mahapulse-training-cuda\Scripts\python.exe`.
The installation and feasibility processes have exited; only the local
demonstration API remains running for browser verification.

The original local data was only the 20-row fixture. The official upstream
data was then acquired from
[L3Cube MarathiNLP](https://github.com/l3cube-pune/MarathiNLP) at
`8ee29fa1329d6a841030eb46659d3c10614b5e59`. The acquisition code now explicitly
checks out HEAD after its sparse `--no-checkout` clone. Prepared official data
contains 60,396 validated records: **47,730 train / 5,922 validation / 6,744
test**, with all three canonical classes represented in every split.
Normalized-text intersections between all split pairs are zero. Upstream
boundaries were retained, with empty rows, duplicates, and contradictory
duplicate labels removed and recorded in ignored `dataset_report.json`.
The source license is CC BY-NC-SA 4.0 with the upstream research-use note.
About 39 GB disk space was available before CUDA installation. Data and model
weights remain ignored by Git.

The verified full-run command is:

```powershell
python -m ml.cli train --full --processed-dir ml/data/processed/mahasent-md --artifact-root ml/artifacts --model-version muril-mahasent-md-v1 --train-batch-size 4 --eval-batch-size 8
```

Only a genuine full run, independent reload, untouched test evaluation, and
real API checks can replace the current smoke gate.

## Security changes verified

- Explicit CORS origins; documented comma-separated and JSON-list syntax.
- PostgreSQL URLs select the installed psycopg driver without printing secrets.
- Database readiness requires the migrated schema, not merely `SELECT 1`.
- Bounded HTTP bodies before multipart/JSON parsing, including chunked upload.
- CSV formula protection includes leading whitespace and control characters.
- SQLAlchemy parameterized queries; no dynamic SQL from session IDs/CSV text.
- Safe error envelopes and exception-type logs without SQL/user parameters.
- Artifact config labels match manifest labels; integrity files cannot escape
  the artifact directory, and hashing is incremental.
- Optional failures retain successful sentiment and visible warning states.

Public demo sessions are addressed by UUID; no account authentication or
per-user authorization product is implemented. The raw-text policy controls
original-text storage rather than promising deletion of all derived text.

## Cloud validation ownership

Render PostgreSQL, migrations on that target, service configuration, live
health/readiness, Vercel static routes, browser flows, deployment restarts,
and deployed export checks are recorded by the final integration agent in
[DEPLOYMENT.md](DEPLOYMENT.md). Local SQLite results do not establish
PostgreSQL runtime success. Local smoke inference does not establish a live
cloud MuRIL deployment or production readiness.
