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
supports the Phase 2 routes and deterministic analyzer cases. Include these
phrases in the textarea to exercise cases:

`assets/js/api-config.js` resolves configuration in this order: runtime
`window.MAHAPULSE_RUNTIME_CONFIG`, optional deployment-provided
`window.MAHAPULSE_STATIC_CONFIG`, then the safe localhost defaults. It strips
trailing slashes and never embeds a Render URL or secret.

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
| `trigger-422` | HTTP 422 safe error |
| `trigger-500` | HTTP 500 safe error |
| `timeout` | delayed response; default frontend timeout is 12 seconds |

The mock server is intentionally deterministic and must never be presented as
model performance.

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

If an optional Playwright installation is present, run
`node frontend-dev/tests/browser-smoke.mjs`; it will run the browser flow.
Without it, the script reports that the checklist is required rather than
claiming browser automation passed.
