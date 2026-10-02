# Frontend development and testing

This directory is frontend-only. It does not replace or modify the FastAPI
service, ML pipeline, or production deployment.

## Run the local fixture

In one terminal, from the repository root:

```powershell
node frontend-dev/mock-api/server.js
```

In a second terminal, serve the static site:

```powershell
python -m http.server 8765
```

Open <http://127.0.0.1:8765/analyzer/>. The frontend default points to
`http://localhost:8000`, so no production URL or secret is needed. The fixture
supports the final Phase 6 routes and deterministic analyzer cases. Include these
phrases in the textarea to exercise cases:

`assets/js/api-config.js` resolves configuration in this order: runtime
`window.MAHAPULSE_RUNTIME_CONFIG`, optional deployment-provided
`window.MAHAPULSE_STATIC_CONFIG`, then the safe localhost defaults. It strips
trailing slashes and never embeds a Render URL or secret.

Deployment builds generate the nonsecret API URL in `assets/js/runtime-config.js`,
loaded before the API boundary. Public batch UI configuration keys are
`MAX_UPLOAD_BYTES` (5,000,000), `MAX_BATCH_ROWS` (1,000), `DEFAULT_TEXT_COLUMN`
(`text`), `PAGE_SIZE` (25, capped at 100), and `BATCH_TIMEOUT_MS` (600,000).
Match any changed limits to the backend environment. They provide feedback;
the server validates the actual file, UTF-8 content, column and row limits.

For local FastAPI CORS, provide the list as JSON to `pydantic-settings`, for
example: `ALLOWED_ORIGINS=["http://127.0.0.1:8765","http://localhost:8765"]`.
This is a run-time environment setting; the frontend does not weaken CORS.

| Input phrase | Fixture behavior |
| --- | --- |
| `हे उत्पादन छान आहे` | positive Marathi |
| `सेवा वाईट आहे` | negative Marathi |
| `अनुभव ठीक आहे` | neutral Marathi |
| `हा app मस्त आहे` | Marathi-English code mixed |
| `low-confidence` | low-confidence warning |
| `empty-keywords` | empty keyword state |
| `topic-null` | null topic state |
| `summary-null` | null summary state |
| `topic-assigned` | optional topic payload |
| `summary-available` | optional summary payload |
| `partial` | partial enrichment warning |
| `keyword-error` | sentiment succeeds with empty keywords and a keyword warning |
| `topic-error` | sentiment succeeds with a null topic and a topic warning |
| `summary-error` | sentiment succeeds with a null summary and a summary warning |
| `multi-enrichment-error` | all optional services fail while sentiment remains available |
| `topic-outlier` | BERTopic-style unassigned/null topic |
| `services-disabled` | analysis fixture with disabled optional results |
| `trigger-422` | HTTP 422 safe error |
| `trigger-500` | HTTP 500 safe error |
| `timeout` | delayed response; default frontend timeout is 12 seconds |

The mock server is intentionally deterministic and must never be presented as
model performance. Readiness/model-info state fixtures are also available with
`/ready?case=services-disabled`, `/ready?case=services-unavailable`, and the
corresponding `/v1/model-info` query parameters. They are development-only
fixtures and do not represent deployment health.

## CSV dashboard fixture

Switch to **CSV batch**, select a UTF-8 CSV with a `text` header, and analyze it.
The fixture implements multipart `POST /v1/analyze/batch?text_column=text`,
paginated `GET /v1/analyses/{session_id}?limit=25&offset=0`, analytics, and
backend-generated CSV/JSON export. Uploading the same filename/content returns
the same deterministic session ID. Sessions live in process memory and reset
when the fixture stops; production persistence remains the FastAPI database.
The `USE_MOCK_API` flag supplies only the existing single-text browser fixtures.
Batch requests always use the HTTP API, including this fixture server; the
browser does not parse CSVs or run NLP.

Create a small CSV such as:

```csv
text
हा मोबाईल खूप चांगला आहे.
ही सेवा अत्यंत खराब आहे.
आज दुकान सकाळी दहा वाजता उघडले.
हा phone चांगला आहे पण battery backup खराब आहे.
""
```

The empty row fails, giving a **partial** session with four successful and one
failed document. `row-error` also deliberately fails a fixture row. A batch
containing only invalid rows is **failed**; all valid rows are **completed**.
Use the same enrichment trigger phrases as single text. Aggregates are returned
by the fixture API, never independently calculated by the dashboard. The API
does not provide an aggregate summary or a Marathi-only count, so the dashboard
communicates those limits explicitly.

Set `$env:MAHAPULSE_MOCK_STORE_RAW_TEXT='false'` before starting the fixture to
test privacy behavior: `original_text` is null in retrieval and exports while
analysis results remain present. CSV exports include a UTF-8 BOM and neutralize
spreadsheet formula prefixes. JSON exports preserve Marathi Unicode.

Run the lightweight frontend checks from the repository root:

```powershell
node scripts/run-frontend-tests.mjs
```

The new batch tests cover upload, completed/partial/failed sessions, validation,
pagination, privacy, analytics, exports, formula protection, safe DOM rendering,
and download failure. They require Node 20+ and no browser/model downloads.

## Browser checklist

Use a current browser with the browser console open and verify:

1. `/analyzer/` loads without console errors; shared nav, Analyzer CTA, theme
   toggle, and sidebar remain functional.
2. Marathi text updates the character counter; example chips populate the
   textarea; Ctrl/⌘+Enter submits; Clear resets input and results.
3. While a request is pending, Analyze and Clear are disabled.
4. Positive, negative, neutral, code-mixed, low-confidence, empty-keyword,
   null-topic, null-summary, and partial cases render their intentional states.
5. Empty input shows validation; `trigger-422`, `trigger-500`, and a stopped
   server show safe errors; `timeout` shows a timeout state.
6. Keyboard-only tab navigation reaches textarea, chips, Analyze, Clear, theme,
   and navigation controls; focus is visible.
7. At a mobile viewport, there is no horizontal overflow and the sidebar menu
   opens/closes; at desktop, the two-column analyzer layout is intact.
8. Switching between Single text and CSV batch preserves their input/results.
   Choose/drop a `.csv`, check filename and size feedback, change the text column,
   submit, and confirm upload progress switches to backend-processing state.
9. Empty, non-CSV, zero-byte and oversized files show validation. Clear CSV
   resets selection and results. Analyze/Clear are disabled while processing.
10. Verify the partial batch shows total/successful/failed counts, backend
    sentiment percentages, confidence ranges, code mixing, keywords, topics and
    an intentional unavailable aggregate summary. Failed rows are visible.
11. For a collection longer than 25 rows, Previous/Next request limited pages.
    Reopen its session ID and confirm retrieval. Test the raw-text policy fixture.
12. Export CSV/JSON downloads the server-generated file; an unavailable backend
    shows a download error. Service indicators distinguish mocks, disabled and
    unavailable enrichments, database readiness, and a smoke sentiment model.
    Smoke mode must visibly say **Development/Smoke Model · Not Production Ready**.

If an optional Playwright installation is present, run
`node frontend-dev/tests/browser-smoke.mjs`; it will run the browser flow.
Without it, the script reports that the checklist is required rather than
claiming browser automation passed.
