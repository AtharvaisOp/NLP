const assert = require('node:assert/strict');
const { createMockServer } = require('../mock-api/server');

async function main() {
  const server = createMockServer({ port: 0, storeRawText: false });
  await new Promise(resolve => server.once('listening', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  let assertions = 0;
  const equal = (actual, expected, message) => { assert.equal(actual, expected, message); assertions++; };
  const upload = async (csv, name = 'मराठी.csv', column = 'text') => {
    const body = new FormData(); body.append('file', new Blob([csv], { type: 'text/csv' }), name);
    return fetch(`${base}/v1/analyze/batch?text_column=${column}`, { method: 'POST', body });
  };
  try {
    const response = await upload('text,id\n"हा मोबाईल खूप चांगला आहे.",1\n"ही सेवा अत्यंत खराब आहे.",2\n"आज दुकान सकाळी दहा वाजता उघडले.",3\n"हा phone चांगला आहे पण battery खराब आहे.",4\n"",5\n"=formula",6\n');
    equal(response.status, 200, 'multipart upload succeeds');
    const batch = await response.json();
    equal(batch.status, 'partial'); equal(batch.total_documents, 6); equal(batch.successful_documents, 5); equal(batch.failed_documents, 1);
    const id = batch.session_id;
    const page = await (await fetch(`${base}/v1/analyses/${id}?limit=2&offset=2`)).json();
    equal(page.documents.length, 2); equal(page.documents[0].row_index, 2); equal(page.offset, 2); equal(page.total_documents, 6);
    equal(page.documents[0].original_text, null, 'raw-text privacy enforced');
    const failedPage = await (await fetch(`${base}/v1/analyses/${id}?limit=2&offset=4`)).json();
    equal(failedPage.documents[0].status, 'failed'); equal(failedPage.documents[0].sentiment, null);
    const analytics = await (await fetch(`${base}/v1/analyses/${id}/analytics`)).json();
    equal(analytics.total, 6); equal(analytics.successful, 5); equal(analytics.failed, 1);
    equal(Object.values(analytics.sentiment).reduce((sum, item) => sum + item.count, 0), 5);
    equal(analytics.language.code_mixed_count, 1); equal(analytics.null_topic_count, 5); equal(analytics.summary, null);
    const csvResponse = await fetch(`${base}/v1/analyses/${id}/export?format=csv`);
    equal(csvResponse.status, 200); equal(csvResponse.headers.get('content-type'), 'text/csv; charset=utf-8');
    assert.match(csvResponse.headers.get('content-disposition'), /\.csv/); assertions++;
    const csv = await csvResponse.text();
    assert.match(csv, /document_id/); assertions++;
    assert.match(csv, /'\=formula/, 'CSV formula injection prevented for normalized text'); assertions++;
    const jsonResponse = await fetch(`${base}/v1/analyses/${id}/export?format=json`);
    const exported = await jsonResponse.json();
    equal(exported.documents.length, 6); equal(exported.documents[0].original_text, null); equal(exported.documents[0].sentiment, 'positive');
    equal((await fetch(`${base}/v1/analyses/${id}?limit=101`)).status, 422);
    equal((await fetch(`${base}/v1/analyses/missing`)).status, 404);
    equal((await fetch(`${base}/v1/analyses/${id}/export?format=html`)).status, 422);
    for (const [content, filename, column] of [['text\nhello\n', 'bad.txt', 'text'], ['', 'empty.csv', 'text'], ['other\nhello\n', 'wrong.csv', 'text'], ['text\n', 'no-rows.csv', 'text'], ['text\n"unterminated', 'invalid.csv', 'text'], ['text\n' + 'hello\n'.repeat(1001), 'too-many.csv', 'text']]) {
      equal((await upload(content, filename, column)).status, 422, filename);
    }
    const complete = await (await upload('comment\n"खूप छान, आवडले"\n', 'complete.csv', 'comment')).json();
    equal(complete.status, 'completed');
    const failed = await (await upload('text\n""\n', 'failed.csv')).json();
    equal(failed.status, 'failed'); equal(failed.successful_documents, 0); equal(failed.failed_documents, 1);
    console.log(`Batch API contract: ${assertions} assertions passed.`);
  } finally { await new Promise(resolve => server.close(resolve)); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
