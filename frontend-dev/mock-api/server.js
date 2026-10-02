/*
 * MahaPulse frontend-only mock API.
 * Development/test use only; sessions are in-memory fixtures, with no models.
 */
const http = require('node:http');
const { createHash } = require('node:crypto');

const LABELS = ['positive', 'negative', 'neutral'];

function json(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type, Accept',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

function service(name, overrides = {}) {
  return {
    name,
    version: 'mock-v0',
    device: 'not-loaded',
    state: 'mocked',
    smoke_test: null,
    backend: 'mock',
    provider: name,
    production_ready: null,
    base_model: null,
    preprocessing_version: null,
    embedding_model: null,
    ...overrides,
  };
}

function stateServices(mode = '') {
  const optional = mode === 'services-disabled'
    ? { state: 'disabled', version: 'disabled', backend: 'disabled', production_ready: null }
    : mode === 'services-unavailable'
      ? { state: 'unavailable', version: 'unavailable', production_ready: false }
      : {};
  return {
    api: { state: 'ready', detail: 'API process is running' },
    preprocessing: { state: 'ready', detail: 'Deterministic preprocessing is available' },
    sentiment: { state: 'mocked', detail: 'Service is mocked and no model is loaded', smoke_test: null, production_ready: null },
    keywords: { state: optional.state || 'mocked', detail: optional.state ? 'Fixture state' : 'Service is mocked and no model is loaded', smoke_test: null, production_ready: null },
    topics: { state: optional.state || 'mocked', detail: optional.state ? 'Fixture state' : 'Service is mocked and no model is loaded', smoke_test: null, production_ready: null },
    summary: { state: optional.state || 'mocked', detail: optional.state ? 'Fixture state' : 'Service is mocked and no model is loaded', smoke_test: null, production_ready: null },
    database: { state: 'ready', detail: 'In-memory development fixture only; sessions reset when this process stops', smoke_test: null, production_ready: null },
  };
}

function modelInfo(mode = '') {
  const disabled = mode === 'services-disabled';
  const unavailable = mode === 'services-unavailable';
  const optional = (name, overrides = {}) => service(name, Object.assign(disabled
    ? { state: 'disabled', version: 'disabled', backend: 'disabled', provider: null, production_ready: null }
    : unavailable
      ? { state: 'unavailable', version: 'unavailable', production_ready: false }
      : {}, overrides));
  return {
    sentiment_model: service('MuRIL', { provider: 'MuRIL' }),
    labels: LABELS,
    keyword_service: optional('KeyBERT'),
    topic_service: optional('BERTopic'),
    summary_service: optional('summarization', { provider: null }),
    readiness: stateServices(mode),
  };
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let body = '';
    req.setEncoding('utf8');
    req.on('data', chunk => {
      body += chunk;
      if (body.length > 250000) reject(new Error('request too large'));
    });
    req.on('end', () => {
      try { resolve(JSON.parse(body || '{}')); }
      catch { reject(new Error('invalid JSON')); }
    });
    req.on('error', reject);
  });
}

function makeAnalysis(text) {
  const lower = text.toLocaleLowerCase();
  const negative = /वाईट|नाही|खराब|निराश|bad|hate|negative/.test(lower);
  const positive = /छान|चांगल|आवड|मस्त|उत्कृष्ट|good|love|positive/.test(lower);
  const lowConfidence = /low[- ]?confidence|uncertain/.test(lower);
  const codeMixed = /[A-Za-z]/.test(text) && /[\u0900-\u097f]/.test(text);
  const sentiment = lowConfidence || (!positive && !negative)
    ? { label: 'neutral', confidence: lowConfidence ? 0.42 : 0.66, probabilities: lowConfidence ? { positive: 0.31, negative: 0.27, neutral: 0.42 } : { positive: 0.14, negative: 0.20, neutral: 0.66 } }
    : positive
      ? { label: 'positive', confidence: 0.93, probabilities: { positive: 0.93, negative: 0.02, neutral: 0.05 } }
      : { label: 'negative', confidence: 0.91, probabilities: { positive: 0.03, negative: 0.91, neutral: 0.06 } };
  const letters = [...text].filter(char => /[A-Za-z\u0900-\u097f]/.test(char));
  const devanagari = letters.filter(char => /[\u0900-\u097f]/.test(char)).length;
  const latin = letters.filter(char => /[A-Za-z]/.test(char)).length;
  const keywordError = /keyword[- ]?error|enrichment[- ]?warning|multi[- ]?enrichment[- ]?error/.test(lower);
  const topicError = /topic[- ]?error|enrichment[- ]?warning|multi[- ]?enrichment[- ]?error/.test(lower);
  const summaryError = /summary[- ]?error|enrichment[- ]?warning|multi[- ]?enrichment[- ]?error/.test(lower);
  const servicesDisabled = /services[- ]?disabled/.test(lower);
  const topicOutlier = /topic[- ]?outlier|topic[- ]?null|no[- ]?topic/.test(lower);
  const nullTopic = topicOutlier || !/topic[- ]?assigned/.test(lower) || topicError || servicesDisabled;
  const nullSummary = !/summary[- ]?available/.test(lower) || /summary[- ]?null|no[- ]?summary/.test(lower) || summaryError || servicesDisabled;
  const emptyKeywords = /empty[- ]?keywords|no[- ]?keywords/.test(lower);
  const partial = /partial|enrichment[- ]?warning/.test(lower);
  const warnings = ['ML service outputs are deterministic mocks; no models are loaded.'];
  if (keywordError) warnings.push('Keyword enrichment unavailable.');
  if (topicError) warnings.push('Topic enrichment unavailable.');
  if (summaryError) warnings.push('Summary enrichment unavailable.');
  if (partial && !keywordError && !topicError && !summaryError) warnings.push('Optional enrichment is partial in the development fixture.');
  return {
    request_id: `mock-${Buffer.from(text).toString('base64url').slice(0, 16) || 'empty'}`,
    original_text: text,
    model_text: text.replace(/\s+/g, ' ').trim(),
    analysis_text: text.replace(/[^A-Za-z\u0900-\u097f0-9]+/g, ' ').trim().toLocaleLowerCase(),
    language: { primary: 'mr', devanagari_ratio: letters.length ? devanagari / letters.length : 0, latin_ratio: letters.length ? latin / letters.length : 0, is_code_mixed: codeMixed },
    sentiment: { ...sentiment, low_confidence: lowConfidence },
    keywords: emptyKeywords || keywordError || servicesDisabled ? [] : [{ text: 'अनुभव', score: 0.91 }, { text: codeMixed ? 'app' : 'उत्पादन', score: 0.73 }],
    topic: nullTopic ? { id: null, label: null, probability: null } : { id: 7, label: 'उत्पादन अनुभव', probability: 0.82 },
    summary: nullSummary ? { text: null, provider: null } : { text: 'Mock extractive summary for frontend verification.', provider: 'extractive' },
    meta: { model_version: 'mock-v0', processing_ms: 4, warnings },
  };
}

async function handleAnalyze(req, res, delayMs) {
  let payload;
  try { payload = await readJson(req); }
  catch (error) { json(res, 422, { error: { code: 'validation_error', message: error.message }, request_id: 'mock-invalid' }); return; }
  const text = typeof payload.text === 'string' ? payload.text : '';
  const lower = text.toLocaleLowerCase();
  if (!text.trim()) { json(res, 422, { error: { code: 'validation_error', message: 'text must not be empty or whitespace-only' }, request_id: 'mock-invalid' }); return; }
  if (/http[- ]?422|trigger[- ]?422/.test(lower)) { json(res, 422, { error: { code: 'validation_error', message: 'Mock validation failure.' }, request_id: 'mock-422' }); return; }
  if (/http[- ]?500|trigger[- ]?500/.test(lower)) { json(res, 500, { error: { code: 'internal_error', message: 'Mock backend failure.' }, request_id: 'mock-500' }); return; }
  const respond = () => json(res, 200, makeAnalysis(text));
  if (/timeout|delayed|slow/.test(lower)) setTimeout(respond, delayMs);
  else respond();
}

function parseCSV(text) {
  const rows = [];
  let row = []; let value = ''; let quoted = false;
  const source = text.replace(/^\uFEFF/, '');
  for (let i = 0; i < source.length; i++) {
    const char = source[i];
    if (char === '"') {
      if (quoted && source[i + 1] === '"') { value += '"'; i++; }
      else if (!quoted && value.length) throw new Error('CSV content is malformed');
      else quoted = !quoted;
    } else if (char === ',' && !quoted) { row.push(value); value = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && source[i + 1] === '\n') i++;
      row.push(value); rows.push(row); row = []; value = '';
    } else value += char;
  }
  if (quoted) throw new Error('CSV content is malformed');
  if (value || row.length) { row.push(value); rows.push(row); }
  return rows;
}

async function readUpload(req) {
  const parts = [];
  let size = 0;
  for await (const chunk of req) { size += chunk.length; if (size > 5100000) throw new Error('upload exceeds the configured maximum size'); parts.push(chunk); }
  // Use the platform's multipart parser; no hand-built boundary parsing.
  const body = new Response(Buffer.concat(parts), { headers: { 'Content-Type': req.headers['content-type'] || '' } });
  const form = await body.formData();
  const file = form.get('file');
  if (!file || !/\.csv$/i.test(file.name || '')) throw new Error('upload must be a CSV file');
  if (file.size > 5000000) throw new Error('upload exceeds the configured maximum size');
  if (!file.size) throw new Error('CSV upload is empty');
  return { name: file.name, text: new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer()) };
}

function aggregateSession(item) {
  const successful = item.documents.filter(row => row.status === 'success');
  const denominator = successful.length || 1;
  const confidences = successful.map(row => row.sentiment.confidence);
  const keywordScores = new Map(); const topics = new Map();
  successful.forEach(row => {
    row.keywords.forEach(keyword => { const scores = keywordScores.get(keyword.text) || []; scores.push(keyword.score); keywordScores.set(keyword.text, scores); });
    if (row.topic.id !== null) {
      const previous = topics.get(row.topic.id) || { id: row.topic.id, label: row.topic.label, count: 0 };
      previous.count++; topics.set(row.topic.id, previous);
    }
  });
  const codeMixedCount = successful.filter(row => row.language.is_code_mixed).length;
  return {
    session_id: item.session.id, total: item.session.total_documents, successful: item.session.successful_documents, failed: item.session.failed_documents,
    sentiment: Object.fromEntries(LABELS.map(label => { const count = successful.filter(row => row.sentiment.label === label).length; return [label, { count, percentage: count * 100 / denominator }]; })),
    confidence: { average: confidences.length ? confidences.reduce((a, b) => a + b, 0) / confidences.length : null, minimum: confidences.length ? Math.min(...confidences) : null, maximum: confidences.length ? Math.max(...confidences) : null, low_confidence_count: successful.filter(row => row.sentiment.low_confidence).length },
    language: { code_mixed_count: codeMixedCount, code_mixed_percentage: codeMixedCount * 100 / denominator },
    keywords: [...keywordScores].map(([text, scores]) => ({ text, count: scores.length, average_score: scores.reduce((a, b) => a + b, 0) / scores.length })).sort((a, b) => b.count - a.count || a.text.localeCompare(b.text)).slice(0, 50),
    topics: [...topics.values()].map(topic => ({ ...topic, percentage: topic.count * 100 / denominator })),
    null_topic_count: successful.filter(row => row.topic.id === null).length, summary: null,
  };
}

function makeBatch(upload, column, storeRawText) {
  const rows = parseCSV(upload.text);
  if (!rows.length) throw new Error('CSV header is missing');
  const header = rows.shift().map(value => value.trim());
  const index = header.indexOf(column);
  if (index < 0) throw new Error('configured text column is missing');
  if (!rows.length) throw new Error('CSV contains no data rows');
  if (rows.length > 1000) throw new Error('CSV row count exceeds the configured maximum');
  const id = `mock-batch-${createHash('sha256').update(upload.name + column + upload.text).digest('hex').slice(0, 16)}`;
  const documents = rows.map((row, row_index) => {
    const original = (row[index] || '').trim();
    const base = { id: `${id}-${row_index}`, row_index, original_text: storeRawText ? original : null, model_text: null, analysis_text: null, language: null, sentiment: null, keywords: [], topic: { id: null, label: null, probability: null }, summary: { text: null, provider: null }, processing_ms: null, warnings: [], error_code: null, error_message: null };
    if (!original || original.length > 100000 || /row-error|trigger-500/i.test(original)) return { ...base, status: 'failed', error_code: original ? 'validation_error' : 'validation_error', error_message: original ? 'Row could not be analyzed by this development fixture.' : 'text must not be empty or whitespace-only' };
    const analysis = makeAnalysis(original);
    return { ...base, status: 'success', model_text: analysis.model_text, analysis_text: analysis.analysis_text, language: analysis.language, sentiment: analysis.sentiment, keywords: analysis.keywords, topic: analysis.topic, summary: analysis.summary, processing_ms: analysis.meta.processing_ms, warnings: analysis.meta.warnings };
  });
  const successful = documents.filter(row => row.status === 'success').length;
  const failed = documents.length - successful;
  const session = { id, source_type: 'batch', filename: upload.name, created_at: '2026-10-03T00:00:00+00:00', completed_at: '2026-10-03T00:00:01+00:00', status: failed ? successful ? 'partial' : 'failed' : 'completed', total_documents: documents.length, successful_documents: successful, failed_documents: failed, model_version: 'mock-v0' };
  return { session, documents };
}

function exportRow(row, sessionId) {
  return {
    document_id: row.id, session_id: sessionId, row_index: row.row_index, status: row.status,
    original_text: row.original_text, model_text: row.model_text, analysis_text: row.analysis_text,
    sentiment: row.sentiment?.label ?? null, confidence: row.sentiment?.confidence ?? null,
    positive_probability: row.sentiment?.probabilities.positive ?? null, negative_probability: row.sentiment?.probabilities.negative ?? null, neutral_probability: row.sentiment?.probabilities.neutral ?? null,
    low_confidence: row.sentiment?.low_confidence ?? null, keywords: row.keywords.map(keyword => `${keyword.text} (${keyword.score.toFixed(6)})`).join('; '),
    topic_id: row.topic.id, topic_label: row.topic.label, topic_probability: row.topic.probability,
    summary: row.summary.text, language: row.language?.primary ?? null, is_code_mixed: row.language?.is_code_mixed ?? null,
    processing_ms: row.processing_ms, error_code: row.error_code, error_message: row.error_message,
  };
}

function csvCell(value) {
  let text = value === null || value === undefined ? '' : String(value);
  if (/^[=+\-@]/.test(text)) text = "'" + text;
  return `"${text.replace(/"/g, '""')}"`;
}

function createMockServer({ port = Number(process.env.MAHAPULSE_MOCK_PORT || 8000), delayMs = Number(process.env.MAHAPULSE_MOCK_DELAY_MS || 13000), storeRawText = process.env.MAHAPULSE_MOCK_STORE_RAW_TEXT !== 'false' } = {}) {
  const sessions = new Map();
  const server = http.createServer(async (req, res) => {
    if (req.method === 'OPTIONS') { res.writeHead(204, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type, Accept', 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS' }); res.end(); return; }
    const requestUrl = new URL(req.url, 'http://127.0.0.1');
    const mode = requestUrl.searchParams.get('case') || '';
    if (req.method === 'GET' && requestUrl.pathname === '/health') { json(res, 200, { status: 'ok', service: 'mahapulse-api', pipeline: 'phase-2' }); return; }
    if (req.method === 'GET' && requestUrl.pathname === '/ready') {
      json(res, 200, { status: 'degraded', services: stateServices(mode) }); return;
    }
    if (req.method === 'GET' && requestUrl.pathname === '/v1/model-info') {
      json(res, 200, modelInfo(mode)); return;
    }
    if (req.method === 'POST' && req.url === '/v1/analyze') { await handleAnalyze(req, res, delayMs); return; }
    if (req.method === 'POST' && requestUrl.pathname === '/v1/analyze/batch') {
      try {
        const upload = await readUpload(req);
        const item = makeBatch(upload, (requestUrl.searchParams.get('text_column') || 'text').trim(), storeRawText);
        sessions.set(item.session.id, item);
        const session = item.session;
        json(res, 200, { session_id: session.id, status: session.status, total_documents: session.total_documents, successful_documents: session.successful_documents, failed_documents: session.failed_documents, processing_ms: item.documents.length * 4, model_version: session.model_version });
      } catch (error) { json(res, 422, { error: { code: 'validation_error', message: error.message }, request_id: 'mock-invalid-batch' }); }
      return;
    }
    const sessionRoute = /^\/v1\/analyses\/([^/]+)(?:\/(analytics|export))?$/.exec(requestUrl.pathname);
    if (req.method === 'GET' && sessionRoute) {
      const item = sessions.get(decodeURIComponent(sessionRoute[1]));
      if (!item) { json(res, 404, { error: { code: 'not_found', message: 'analysis session was not found' }, request_id: 'mock-404' }); return; }
      if (sessionRoute[2] === 'analytics') { json(res, 200, aggregateSession(item)); return; }
      if (sessionRoute[2] === 'export') {
        const format = requestUrl.searchParams.get('format') || 'json';
        if (!['csv', 'json'].includes(format)) { json(res, 422, { error: { code: 'validation_error', message: 'Choose csv or json format' }, request_id: 'mock-422' }); return; }
        const rows = item.documents.map(row => exportRow(row, item.session.id));
        const columns = rows.length ? Object.keys(rows[0]) : ['document_id', 'status'];
        const content = format === 'json' ? JSON.stringify({ session: item.session, documents: rows }) : '\uFEFF' + [columns.map(csvCell).join(','), ...rows.map(row => columns.map(column => csvCell(row[column])).join(','))].join('\r\n');
        res.writeHead(200, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Expose-Headers': 'Content-Disposition', 'Content-Type': format === 'json' ? 'application/json; charset=utf-8' : 'text/csv; charset=utf-8', 'Content-Disposition': `attachment; filename="mahapulse-${item.session.id}.${format}"` });
        res.end(content); return;
      }
      const limit = Number(requestUrl.searchParams.get('limit') || 50); const offset = Number(requestUrl.searchParams.get('offset') || 0);
      if (!Number.isInteger(limit) || limit < 1 || limit > 100 || !Number.isInteger(offset) || offset < 0) { json(res, 422, { error: { code: 'validation_error', message: 'Invalid limit or offset' }, request_id: 'mock-422' }); return; }
      json(res, 200, { session: item.session, documents: item.documents.slice(offset, offset + limit), total_documents: item.documents.length, limit, offset }); return;
    }
    json(res, 404, { error: { code: 'not_found', message: 'Mock route not found.' }, request_id: 'mock-404' });
  });
  server.listen(port, () => console.log(`MahaPulse mock API listening on http://127.0.0.1:${server.address().port}`));
  return server;
}

if (require.main === module) createMockServer();

module.exports = { createMockServer, makeAnalysis, makeBatch, aggregateSession, parseCSV };
