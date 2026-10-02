const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.hidden = false;
    this.attributes = {};
    this._text = '';
  }

  set textContent(value) { this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
}

const ids = [
  'keyword-list', 'topic-empty', 'topic-data', 'topic-id', 'topic-label',
  'topic-probability', 'summary-empty', 'summary-data', 'summary-text',
  'summary-provider', 'warning-list',
];
const nodes = Object.fromEntries(ids.map(id => [id, new FakeNode('div')]));
const context = {
  console,
  document: {
    addEventListener() {},
    getElementById(id) { return nodes[id] || null; },
    createElement(tag) { return new FakeNode(tag); },
  },
  window: { NLP_COMPONENTS_READY: false },
};
vm.runInNewContext(fs.readFileSync('assets/js/analyzer.js', 'utf8'), context);
const analyzer = context.window.MahaPulseAnalyzer;
const unsafe = '<script>alert(1)</script>';

analyzer.renderKeywords([{ text: unsafe, score: 0.91 }]);
assert.match(nodes['keyword-list'].children[0].textContent, new RegExp(unsafe.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')), 'keyword is text-only');
analyzer.renderTopic({ id: 3, label: unsafe, probability: 0.8 });
assert.equal(nodes['topic-label'].textContent, unsafe, 'topic label is text-only');
analyzer.renderSummary({ text: unsafe, provider: 'extractive' });
assert.equal(nodes['summary-text'].textContent, unsafe, 'summary is text-only');
analyzer.renderWarnings([unsafe]);
assert.equal(nodes['warning-list'].children[0].textContent, unsafe, 'warning is text-only');
analyzer.renderKeywords('<malformed>');
assert.match(nodes['keyword-list'].textContent, /No keywords/, 'malformed keywords degrade safely');
analyzer.renderTopic('<malformed>');
assert.equal(nodes['topic-empty'].hidden, false, 'malformed topic degrades safely');
analyzer.renderSummary(42);
assert.equal(nodes['summary-empty'].hidden, false, 'malformed summary degrades safely');
console.log('Analyzer enrichment and text-safety checks passed.');
