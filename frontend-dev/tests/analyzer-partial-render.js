const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class FakeNode {
  constructor() {
    this.children = [];
    this.hidden = false;
    this.dataset = {};
    this.style = {};
    this._text = '';
  }

  set textContent(value) { this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  setAttribute() {}
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

context.window.MahaPulseAnalyzer.renderResponse({
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
});

assert.equal(nodes.get('sentiment-label').textContent, 'positive');
assert.match(nodes.get('analysis-state').textContent, /Partial enrichment result/);
assert.match(nodes.get('warning-list').textContent, /Keyword enrichment unavailable/);
assert.equal(nodes.get('results-panel').hidden, false);
console.log('Analyzer partial-result rendering checks passed.');
