const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const context = {
  console,
  document: { addEventListener() {}, getElementById() { return null; } },
  window: { NLP_COMPONENTS_READY: false },
};
vm.runInNewContext(fs.readFileSync('assets/js/analyzer.js', 'utf8'), context);
const validate = context.window.MahaPulseAnalyzer.validateAnalysisResponse;
const serviceLabel = context.window.MahaPulseAnalyzer.serviceLabel;
const enrichmentSummary = context.window.MahaPulseAnalyzer.enrichmentSummary;
const valid = {
  request_id: 'request-1', original_text: 'हे छान आहे', model_text: 'हे छान आहे', analysis_text: 'हे छान आहे',
  language: { primary: 'mr', devanagari_ratio: 1, latin_ratio: 0, is_code_mixed: false },
  sentiment: { label: 'positive', confidence: 0.9, probabilities: { positive: 0.9, negative: 0.05, neutral: 0.05 }, low_confidence: false },
};
assert.equal(validate(valid), true, 'optional enrichment fields may be absent');
assert.equal(validate({ ...valid, keywords: '<malformed>', topic: '<malformed>', summary: 42 }), true, 'malformed optional enrichment does not invalidate core analysis');
assert.equal(validate({ ...valid, sentiment: { ...valid.sentiment, probabilities: { positive: 2, negative: -1, neutral: 0 } } }), false, 'malformed probabilities rejected');
assert.equal(validate({ ...valid, sentiment: { ...valid.sentiment, label: 'unknown' } }), false, 'invalid sentiment label rejected');
assert.match(serviceLabel({ name: 'MuRIL', version: 'smoke-v4', state: 'ready', smoke_test: true, production_ready: false }), /not production-ready/, 'smoke lifecycle is explicit');
assert.doesNotMatch(serviceLabel({ name: 'Rule-based demo', version: 'rule-demo-v1', state: 'mocked', smoke_test: false, production_ready: false }), /smoke/, 'a demo is not a smoke model');
assert.match(serviceLabel({ name: 'Rule-based demo', version: 'rule-demo-v1', state: 'mocked', smoke_test: false, production_ready: false }), /demo.*not production-ready/, 'demo lifecycle is explicit');
assert.doesNotMatch(serviceLabel({ name: 'MuRIL', version: 'unpromoted-v1', state: 'ready', smoke_test: false, production_ready: false }), /smoke/, 'an unpromoted full model is not a smoke model');
assert.equal(enrichmentSummary({ keywords: { state: 'ready' }, topics: { state: 'ready' }, summary: { state: 'ready' } }, null).label, 'Enrichment ready');
assert.equal(enrichmentSummary({ keywords: { state: 'disabled' }, topics: { state: 'disabled' }, summary: { state: 'disabled' } }, null).label, 'Enrichment disabled');
assert.equal(enrichmentSummary({ keywords: { state: 'ready' }, topics: { state: 'disabled' }, summary: { state: 'ready' } }, null).label, 'Enrichment ready · some optional services disabled', 'disabled optional topics are not a failure');
assert.equal(enrichmentSummary({ keywords: { state: 'ready' }, topics: { state: 'unavailable' }, summary: { state: 'ready' } }, null).label, 'Enrichment partial');
assert.equal(enrichmentSummary({ keywords: { state: 'mocked' }, topics: { state: 'mocked' }, summary: { state: 'mocked' } }, null).label, 'Development mocks active');
console.log('Analyzer response guard checks passed.');
