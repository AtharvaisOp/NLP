const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class FakeNode {
  constructor() {
    this.children = [];
    this.hidden = false;
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this._text = '';
  }

  set textContent(value) { this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  removeAttribute() {}
}

const nodes = new Map();
const document = {
  addEventListener() {},
  getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, new FakeNode());
    return nodes.get(id);
  },
  createElement() { return new FakeNode(); },
};
const context = { console, document, window: { NLP_COMPONENTS_READY: false } };
vm.runInNewContext(fs.readFileSync('assets/js/analyzer.js', 'utf8'), context);

const response = {
  request_id: 'partial-1',
  original_text: 'हा phone चांगला आहे',
  model_text: 'हा phone चांगला आहे',
  analysis_text: 'हा phone चांगला आहे',
  language: { primary: 'mr', devanagari_ratio: 0.5, latin_ratio: 0.5, is_code_mixed: true },
  sentiment: {
    label: 'positive',
    confidence: 0.8,
    probabilities: { positive: 0.8, negative: 0.1, neutral: 0.1 },
    low_confidence: false,
  },
  keywords: [],
  topic: { id: null, label: null, probability: null },
  summary: { text: null, provider: 'extractive' },
  meta: {
    model_version: 'muril-smoke-v4',
    processing_ms: 20,
    warnings: ['Keyword enrichment unavailable.', 'Topic enrichment unavailable.'],
  },
};

context.window.MahaPulseAnalyzer.renderResponse(response);

assert.equal(nodes.get('sentiment-label').textContent, 'positive');
assert.match(nodes.get('analysis-state').textContent, /Partial enrichment result/);
assert.match(nodes.get('warning-list').textContent, /Keyword enrichment unavailable/);
assert.equal(nodes.get('results-panel').hidden, false);
assert.equal(nodes.get('sentiment-score-label').textContent, 'model score');

const demo = { ...response, meta: { ...response.meta, model_version: 'rule-demo-v1', warnings: ['Rule-based demo; no trained sentiment model is loaded. Scores are normalized rule weights, not calibrated probabilities.'] } };
context.window.MahaPulseAnalyzer.renderResponse(demo);
assert.equal(nodes.get('analysis-state').dataset.state, 'success', 'null topics, summary and keywords are normal optional outputs');
assert.match(nodes.get('analysis-state').textContent, /Demo analysis complete/);
assert.equal(nodes.get('sentiment-score-label').textContent, 'demo score');
assert.match(nodes.get('warning-list').textContent, /not calibrated/, 'demo disclosure remains visible');
assert.match(nodes.get('sentiment-probabilities').attributes['aria-label'], /not calibrated probabilities/);
assert.match(nodes.get('prob-positive-bar').attributes['aria-label'], /demo score/);

context.window.MahaPulseAnalyzer.renderResponse({ ...demo, meta: { ...demo.meta, model_version: 'mock-v0' } });
assert.equal(nodes.get('sentiment-score-label').textContent, 'demo score', 'legacy mocks also do not show model confidence');
context.window.MahaPulseAnalyzer.renderResponse({ ...demo, meta: { ...demo.meta, warnings: [...demo.meta.warnings, 'Keyword enrichment unavailable.'] } });
assert.equal(nodes.get('analysis-state').dataset.state, 'partial', 'an actual optional-service failure keeps the warning banner');
context.window.MahaPulseAnalyzer.renderResponse({ ...response, meta: { ...response.meta, model_version: 'muril-mahasent-md-v1', warnings: [] } });
assert.equal(nodes.get('analysis-state').dataset.state, 'success');
assert.equal(nodes.get('sentiment-score-label').textContent, 'model score', 'real artifact output restores the model label');
assert.match(nodes.get('prob-positive-bar').attributes['aria-label'], /model probability/);
console.log('Analyzer partial-result rendering checks passed.');
