const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Node {
  constructor() { this.children = []; this.hidden = false; this.dataset = {}; this.style = {}; this.attributes = {}; this.value = ''; this.disabled = false; this._text = ''; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(node => node.textContent).join(''); }
  replaceChildren(...nodes) { this._text = ''; this.children = nodes; }
  append(...nodes) { this.children.push(...nodes); }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; }
  focus() {}
}
const nodes = new Map();
const document = { addEventListener() {}, getElementById(id) { if (!nodes.has(id)) nodes.set(id, new Node()); return nodes.get(id); }, createElement() { return new Node(); } };
let calls = [];
const session = { session: { id: 'session-1', status: 'partial', total_documents: 26, successful_documents: 25, failed_documents: 1, model_version: 'smoke-v4' }, total_documents: 26, limit: 25, offset: 0, documents: [{ row_index: 0, status: 'success', original_text: null, sentiment: { label: 'positive', confidence: .8, low_confidence: false }, keywords: [{ text: '<script>unsafe</script>', score: .7 }], topic: { id: null }, summary: { text: null }, processing_ms: 12, language: { primary: 'mr', is_code_mixed: true } }] };
const analytics = { session_id: 'session-1', total: 26, successful: 25, failed: 1, sentiment: { positive: { count: 12, percentage: 48 }, neutral: { count: 8, percentage: 32 }, negative: { count: 5, percentage: 20 } }, confidence: { average: .8, minimum: .42, maximum: .95, low_confidence_count: 3 }, language: { code_mixed_count: 10, code_mixed_percentage: 40 }, keywords: [{ text: '<img src=x onerror=bad()>', count: 12, average_score: .9 }], topics: [], null_topic_count: 25, summary: null };
const api = { session: async (id, limit, offset) => { calls.push({ id, limit, offset }); return { ...session, offset, documents: offset ? [{ row_index: 25, status: 'failed', original_text: '', error_code: 'validation_error', error_message: '<script>failure</script>' }] : session.documents }; }, analytics: async () => analytics };
const window = { NLP_COMPONENTS_READY: false, MahaPulseAPI: api, MAHAPULSE_CONFIG: { PAGE_SIZE: 25, MAX_UPLOAD_BYTES: 5000000, MAX_BATCH_ROWS: 1000 } };
vm.runInNewContext(fs.readFileSync('assets/js/batch-analyzer.js', 'utf8'), { window, document });
const batch = window.MahaPulseBatch;
let assertions = 0;
const equal = (actual, expected, message) => { assert.equal(actual, expected, message); assertions++; };
async function main() {
  equal(batch.fileError(null), 'Choose a CSV file before analyzing.');
  assert.match(batch.fileError({ name: 'x.txt', size: 1 }), /\.csv/); assertions++;
  assert.match(batch.fileError({ name: 'x.csv', size: 0 }), /empty/); assertions++;
  assert.match(batch.fileError({ name: 'x.csv', size: 5000001 }), /exceeds/); assertions++;
  equal(batch.fileError({ name: 'मराठी.CSV', size: 200 }), null);
  equal(batch.validBatch({ session_id: 'x', status: 'partial', total_documents: 2, successful_documents: 1, failed_documents: 1, processing_ms: 1, model_version: 'mock' }), true);
  equal(batch.validBatch({ session_id: 'x', status: 'completed', total_documents: 2, successful_documents: 1, failed_documents: 0, processing_ms: 1, model_version: 'mock' }), false, 'inconsistent counts rejected');
  equal(batch.validSession(session), true); equal(batch.validAnalytics(analytics), true); equal(batch.validAnalytics({ ...analytics, total: 99 }), false);
  batch.renderDocuments(session);
  equal(nodes.get('raw-text-note').hidden, false);
  assert.match(nodes.get('document-rows').textContent, /privacy policy/); assertions++;
  assert.match(nodes.get('document-rows').textContent, /<script>unsafe<\/script>/); assertions++;
  equal(nodes.get('document-rows').children[0].children[3].children.length, 0, 'untrusted keywords are text-only');
  batch.renderAnalytics(analytics);
  equal(nodes.get('confidence-average').textContent, '80.0%'); equal(nodes.get('batch-failed').textContent, '1');
  assert.match(nodes.get('batch-language').textContent, /10 code-mixed documents · 40.0%/); assertions++;
  equal(nodes.get('batch-keywords').children[0].children[0].children.length, 0, 'keyword markup never becomes DOM');
  equal(nodes.get('batch-sentiment-list').children[0].children[1].children[0].style.width, '48%', 'server percentages rendered directly');
  batch.switchMode('batch'); equal(nodes.get('single-analysis-panel').hidden, true); equal(nodes.get('batch-analysis-panel').hidden, false);
  await batch.loadDashboard('session-1');
  equal(calls[0].limit, 25); equal(calls[0].offset, 0); equal(nodes.get('next-page-button').disabled, false);
  await batch.loadPage(25);
  equal(calls[1].offset, 25); equal(nodes.get('previous-page-button').disabled, false); equal(nodes.get('next-page-button').disabled, true);
  assert.match(nodes.get('document-rows').textContent, /Failed.*validation_error/); assertions++;
  assert.match(nodes.get('document-page-info').textContent, /26–26 of 26/); assertions++;
  api.exportSession = async () => { throw new Error('Download unavailable'); };
  await batch.exportResults('csv');
  equal(nodes.get('export-state').textContent, 'Export failed: Download unavailable');
  equal(nodes.get('export-csv-button').disabled, false);
  batch.clearCSV(); equal(nodes.get('batch-results').hidden, true);
  console.log(`Batch dashboard: ${assertions} assertions passed.`);
}
main().catch(error => { console.error(error); process.exitCode = 1; });
