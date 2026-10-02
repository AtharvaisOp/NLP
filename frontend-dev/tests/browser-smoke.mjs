import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';

let playwright;
try {
  playwright = await import('playwright');
} catch {
  console.log('Browser automation unavailable: run frontend-dev/README.md browser checklist manually.');
  process.exit(0);
}

const staticUrl = process.env.MAHAPULSE_ANALYZER_URL || 'http://127.0.0.1:8765/analyzer/';
const mockProcess = process.env.MAHAPULSE_SKIP_MOCK ? null : spawn(process.execPath, ['frontend-dev/mock-api/server.js'], { stdio: 'ignore' });
const pause = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const browser = await playwright.chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
const errors = [];
page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
page.on('pageerror', error => errors.push(error.message));

try {
  if (mockProcess) await pause(250);
  await page.addInitScript(() => { window.MAHAPULSE_RUNTIME_CONFIG = { REQUEST_TIMEOUT_MS: 100 }; });
  await page.goto(staticUrl, { waitUntil: 'networkidle' });
  assert.equal(await page.locator('#analysis-text').count(), 1, 'analyzer page loads');
  const textarea = page.locator('#analysis-text');
  const analyze = page.locator('#analyze-button');
  const clear = page.locator('#clear-button');
  await textarea.fill('हे उत्पादन छान आहे');
  assert.match(await page.locator('#character-count').innerText(), /[1-9]/, 'character counter updates');
  await page.locator('[data-example]').last().click();
  assert.match(await textarea.inputValue(), /app/, 'example chip populates text');

  await textarea.fill('हे उत्पादन छान आहे');
  const pendingClick = analyze.click();
  assert.equal(await analyze.isDisabled(), true, 'analyze disables during request');
  await pendingClick;
  await page.waitForFunction(() => !document.querySelector('#analyze-button').disabled);
  assert.equal(await page.locator('#sentiment-label').innerText(), 'positive', 'positive response renders');
  assert.match(await page.locator('#prob-positive-value').innerText(), /%/, 'probabilities render');
  assert.match(await page.locator('#language-devanagari').innerText(), /%/, 'language stats render');

  async function analyzeText(value) {
    await textarea.fill(value);
    await analyze.click();
    await page.waitForFunction(() => !document.querySelector('#analyze-button').disabled);
  }
  await analyzeText('हा app मस्त आहे');
  assert.match(await page.locator('#language-mixed').innerText(), /Yes/, 'code-mixed indicator renders');
  await analyzeText('empty-keywords');
  assert.match(await page.locator('#keyword-list').innerText(), /No keywords/, 'empty keywords state renders');
  await analyzeText('topic-null');
  assert.equal(await page.locator('#topic-empty').isVisible(), true, 'null topic state renders');
  await analyzeText('summary-null');
  assert.equal(await page.locator('#summary-empty').isVisible(), true, 'null summary state renders');
  await analyzeText('partial');
  assert.match(await page.locator('#analysis-state').innerText(), /Partial enrichment/, 'partial state renders');
  await analyzeText('low-confidence');
  assert.equal(await page.locator('#confidence-warning').isVisible(), true, 'low confidence state renders');
  await textarea.fill('');
  await analyze.click();
  assert.match(await page.locator('#analysis-state').innerText(), /Validation error/, 'validation error renders');
  await analyzeText('trigger-422');
  assert.match(await page.locator('#analysis-state').innerText(), /Analysis failed safely/, 'HTTP 422 renders safely');
  await analyzeText('trigger-500');
  assert.match(await page.locator('#analysis-state').innerText(), /Backend unavailable/, 'HTTP 500 renders safely');
  await analyzeText('timeout');
  assert.match(await page.locator('#analysis-state').innerText(), /Request timed out/, 'timeout renders');
  await textarea.fill('हे उत्पादन छान आहे');
  await page.locator('#analysis-text').press('Control+Enter');
  await page.waitForFunction(() => !document.querySelector('#analyze-button').disabled);
  await clear.click();
  assert.equal(await textarea.inputValue(), '', 'Clear resets input');
  assert.equal(await page.locator('#results-panel').isHidden(), true, 'Clear resets results');
  const originalTheme = await page.locator('body').getAttribute('class');
  await page.locator('#theme-toggle').click();
  assert.notEqual(await page.locator('body').getAttribute('class'), originalTheme, 'theme toggle works');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true, 'mobile viewport has no horizontal overflow');
  assert.deepEqual(errors, [], 'no browser console errors');
  console.log('Browser smoke checks passed.');
} finally {
  await browser.close();
  if (mockProcess) mockProcess.kill();
}
