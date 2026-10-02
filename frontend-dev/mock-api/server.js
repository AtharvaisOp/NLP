/*
 * MahaPulse frontend-only mock API.
 * Development/test use only; this server has no persistence or model code.
 */
const http = require('node:http');

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

function service(name) {
  return { name, version: 'mock-dev-v1', device: 'not-loaded', state: 'mocked' };
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
  const positive = /छान|चांगले|आवड|मस्त|उत्कृष्ट|good|love|positive/.test(lower);
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
  const nullTopic = /topic[- ]?null|no[- ]?topic/.test(lower);
  const nullSummary = /summary[- ]?null|no[- ]?summary/.test(lower);
  const emptyKeywords = /empty[- ]?keywords|no[- ]?keywords/.test(lower);
  const partial = /partial|enrichment[- ]?warning/.test(lower);
  const warnings = partial ? ['Keyword/topic enrichment is partial in the development fixture.'] : [];
  return {
    request_id: `mock-${Buffer.from(text).toString('base64url').slice(0, 16) || 'empty'}`,
    original_text: text,
    model_text: text.replace(/\s+/g, ' ').trim(),
    analysis_text: text.replace(/[^A-Za-z\u0900-\u097f0-9]+/g, ' ').trim().toLocaleLowerCase(),
    language: { primary: 'mr', devanagari_ratio: letters.length ? devanagari / letters.length : 0, latin_ratio: letters.length ? latin / letters.length : 0, is_code_mixed: codeMixed },
    sentiment: { ...sentiment, low_confidence: lowConfidence },
    keywords: emptyKeywords ? [] : [{ text: 'अनुभव', score: 0.91 }, { text: codeMixed ? 'app' : 'उत्पादन', score: 0.73 }],
    topic: nullTopic ? { id: null, label: null, probability: null } : { id: 7, label: 'उत्पादन अनुभव', probability: 0.82 },
    summary: nullSummary ? { text: null, provider: null } : { text: 'Mock summary for frontend verification.', provider: 'mock-summary' },
    meta: { model_version: 'mock-dev-v1', processing_ms: 4, warnings },
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

function createMockServer({ port = Number(process.env.MAHAPULSE_MOCK_PORT || 8000), delayMs = Number(process.env.MAHAPULSE_MOCK_DELAY_MS || 13000) } = {}) {
  const server = http.createServer(async (req, res) => {
    if (req.method === 'OPTIONS') { res.writeHead(204, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'Content-Type, Accept', 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS' }); res.end(); return; }
    if (req.method === 'GET' && req.url === '/health') { json(res, 200, { status: 'ok', service: 'mahapulse-api', pipeline: 'phase-2-mock' }); return; }
    if (req.method === 'GET' && req.url === '/ready') {
      json(res, 200, { status: 'degraded', services: { preprocessing: { state: 'ready', detail: 'available' }, sentiment: { state: 'mocked', detail: 'fixture only' }, keywords: { state: 'mocked', detail: 'fixture only' }, topics: { state: 'mocked', detail: 'fixture only' }, summary: { state: 'mocked', detail: 'fixture only' } } }); return;
    }
    if (req.method === 'GET' && req.url === '/v1/model-info') {
      json(res, 200, { sentiment_model: service('MuRIL'), labels: LABELS, keyword_service: service('KeyBERT'), topic_service: service('BERTopic'), summary_service: service('summarization'), readiness: {} }); return;
    }
    if (req.method === 'POST' && req.url === '/v1/analyze') { await handleAnalyze(req, res, delayMs); return; }
    json(res, 404, { error: { code: 'not_found', message: 'Mock route not found.' }, request_id: 'mock-404' });
  });
  server.listen(port, () => console.log(`MahaPulse mock API listening on http://127.0.0.1:${server.address().port}`));
  return server;
}

if (require.main === module) createMockServer();

module.exports = { createMockServer, makeAnalysis };
