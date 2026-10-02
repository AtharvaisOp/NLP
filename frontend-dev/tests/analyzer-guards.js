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
const valid = {
  request_id: 'request-1', original_text: 'हे छान आहे', model_text: 'हे छान आहे', analysis_text: 'हे छान आहे',
  language: { primary: 'mr', devanagari_ratio: 1, latin_ratio: 0, is_code_mixed: false },
  sentiment: { label: 'positive', confidence: 0.9, probabilities: { positive: 0.9, negative: 0.05, neutral: 0.05 }, low_confidence: false },
};
assert.equal(validate(valid), true, 'optional enrichment fields may be absent');
assert.equal(validate({ ...valid, sentiment: { ...valid.sentiment, probabilities: { positive: 2, negative: -1, neutral: 0 } } }), false, 'malformed probabilities rejected');
assert.equal(validate({ ...valid, sentiment: { ...valid.sentiment, label: 'unknown' } }), false, 'invalid sentiment label rejected');
console.log('Analyzer response guard checks passed.');
