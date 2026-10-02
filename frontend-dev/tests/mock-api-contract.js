const assert = require('node:assert/strict');
const { createMockServer } = require('../mock-api/server');

async function request(base, path, options) {
  const response = await fetch(`${base}${path}`, options);
  const body = await response.json();
  return { response, body };
}

async function main() {
  const server = createMockServer({ port: 0, delayMs: 20 });
  await new Promise(resolve => server.once('listening', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    for (const path of ['/health', '/ready', '/v1/model-info']) {
      const { response } = await request(base, path);
      assert.equal(response.status, 200, path);
    }
    const samples = [
      'हे उत्पादन छान आहे', 'सेवा वाईट आहे', 'अनुभव ठीक आहे', 'हा app मस्त आहे',
      'low-confidence', 'empty-keywords', 'topic-null', 'topic-outlier',
      'summary-null', 'summary-available', 'partial', 'keyword-error',
      'topic-error', 'summary-error', 'multi-enrichment-error', 'services-disabled',
    ];
    for (const text of samples) {
      const { response, body } = await request(base, '/v1/analyze', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text }) });
      assert.equal(response.status, 200, text);
      assert.equal(typeof body.request_id, 'string');
      assert.equal(typeof body.sentiment.label, 'string');
      assert.ok(Math.abs(Object.values(body.sentiment.probabilities).reduce((sum, value) => sum + value, 0) - 1) < 1e-9);
    }
    const assigned = await request(base, '/v1/analyze', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: 'topic-assigned summary-available' }) });
    assert.equal(assigned.body.topic.id, 7);
    assert.equal(assigned.body.topic.label, 'उत्पादन अनुभव');
    assert.equal(assigned.body.summary.provider, 'extractive');
    for (const [text, warning] of [['keyword-error', 'Keyword enrichment unavailable.'], ['topic-error', 'Topic enrichment unavailable.'], ['summary-error', 'Summary enrichment unavailable.'], ['multi-enrichment-error', 'Summary enrichment unavailable.']]) {
      const partial = await request(base, '/v1/analyze', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text }) });
      assert.equal(partial.body.sentiment.label.length > 0, true);
      assert.ok(partial.body.meta.warnings.includes(warning), `${text} warning`);
      if (text === 'multi-enrichment-error') {
        assert.equal(partial.body.keywords.length, 0);
        assert.equal(partial.body.topic.id, null);
        assert.equal(partial.body.summary.text, null);
        assert.equal(partial.body.meta.warnings.filter(item => /enrichment unavailable/.test(item)).length, 3);
      }
    }
    for (const mode of ['services-disabled', 'services-unavailable']) {
      const ready = await request(base, `/ready?case=${mode}`);
      const info = await request(base, `/v1/model-info?case=${mode}`);
      assert.equal(ready.body.services.topics.state, mode === 'services-disabled' ? 'disabled' : 'unavailable');
      assert.equal(info.body.topic_service.state, mode === 'services-disabled' ? 'disabled' : 'unavailable');
    }
    for (const [text, status] of [['trigger-422', 422], ['trigger-500', 500]]) {
      const { response } = await request(base, '/v1/analyze', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text }) });
      assert.equal(response.status, status);
    }
    const delayed = await request(base, '/v1/analyze', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: 'timeout' }) });
    assert.equal(delayed.response.status, 200);
    console.log('Mock API contract checks passed.');
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
