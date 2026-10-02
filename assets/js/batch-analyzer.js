/** CSV dashboard: the API owns analysis, persistence, aggregation and export. */
(function (global) {
  'use strict';
  const $ = id => document.getElementById(id);
  const api = global.MahaPulseAPI;
  const config = global.MAHAPULSE_CONFIG || {};
  const pageSize = Math.max(1, Math.min(100, Number(config.PAGE_SIZE) || 25));
  let selectedFile = null;
  let activeSession = null;
  let pageOffset = 0;
  let pageTotal = 0;
  let busy = false;
  let paging = false;
  let generation = 0;

  const write = (id, value) => { if ($(id)) $(id).textContent = value === null || value === undefined ? '—' : String(value); };
  const ratio = value => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'Unavailable';
  const integer = value => Number.isInteger(value) && value >= 0;
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const show = (id, visible) => { if ($(id)) $(id).hidden = !visible; };

  function state(kind, message) {
    const node = $('batch-state');
    if (node) { node.dataset.state = kind; node.textContent = message; }
  }

  function fileError(file) {
    if (!file) return 'Choose a CSV file before analyzing.';
    if (!/\.csv$/i.test(file.name || '')) return 'Choose a file with a .csv extension.';
    if (!file.size) return 'This CSV is empty. Choose a file containing a header and data rows.';
    if (file.size > config.MAX_UPLOAD_BYTES) return `The CSV exceeds the configured ${(config.MAX_UPLOAD_BYTES / 1000000).toFixed(1)} MB upload limit.`;
    return null;
  }

  function validBatch(result) {
    return !!result && typeof result.session_id === 'string' && result.session_id.length > 0
      && ['completed', 'partial', 'failed'].includes(result.status)
      && [result.total_documents, result.successful_documents, result.failed_documents, result.processing_ms].every(integer)
      && result.successful_documents + result.failed_documents === result.total_documents
      && typeof result.model_version === 'string';
  }

  function validSession(result) {
    return !!result && typeof result.session?.id === 'string'
      && ['processing', 'completed', 'partial', 'failed'].includes(result.session.status)
      && [result.total_documents, result.limit, result.offset, result.session.total_documents, result.session.successful_documents, result.session.failed_documents].every(integer)
      && result.limit > 0 && Array.isArray(result.documents) && result.documents.length <= result.limit
      && result.documents.every(row => integer(row.row_index) && ['success', 'failed'].includes(row.status));
  }

  function validAnalytics(result) {
    return !!result && [result.total, result.successful, result.failed, result.null_topic_count, result.confidence?.low_confidence_count, result.language?.code_mixed_count].every(integer)
      && result.total === result.successful + result.failed
      && ['positive', 'neutral', 'negative'].every(label => integer(result.sentiment?.[label]?.count) && number(result.sentiment[label].percentage))
      && [result.confidence.average, result.confidence.minimum, result.confidence.maximum].every(value => value === null || number(value))
      && number(result.language.code_mixed_percentage) && Array.isArray(result.keywords) && Array.isArray(result.topics);
  }

  function chooseFile(file) {
    if (busy) return;
    selectedFile = file || null;
    write('batch-file-info', file ? `${file.name} · ${(file.size / 1000).toFixed(1)} KB` : 'No file selected.');
    const error = file ? fileError(file) : null;
    $('batch-file')?.setAttribute('aria-invalid', error ? 'true' : 'false');
    if (error) state('validation', error);
    else state('idle', file ? 'File selected. The backend will validate its UTF-8 encoding, header and rows.' : 'Choose a CSV to start, or reopen a saved session below.');
  }

  function setBusy(value) {
    busy = value;
    ['analyze-csv-button', 'clear-csv-button', 'batch-file', 'text-column', 'load-session-button', 'session-input'].forEach(id => { if ($(id)) $(id).disabled = value; });
    $('batch-form')?.setAttribute('aria-busy', String(value));
    updatePagination();
  }

  function updatePagination() {
    if ($('previous-page-button')) $('previous-page-button').disabled = busy || paging || pageOffset === 0;
    if ($('next-page-button')) $('next-page-button').disabled = busy || paging || pageOffset + pageSize >= pageTotal;
  }

  function renderBatch(result) {
    activeSession = result.session_id || result.id;
    write('batch-session-id', activeSession);
    if ($('session-input')) $('session-input').value = activeSession;
    write('batch-status', result.status);
    if ($('batch-status')) $('batch-status').dataset.state = result.status;
    write('batch-total', result.total_documents);
    write('batch-successful', result.successful_documents);
    write('batch-failed', result.failed_documents);
    write('batch-processing', integer(result.processing_ms) ? `${result.processing_ms.toLocaleString()} ms` : 'Not recorded for reopened sessions');
    write('batch-model-version', result.model_version);
    show('batch-results', true);
  }

  function renderAnalytics(result) {
    write('batch-total', result.total);
    write('batch-successful', result.successful);
    write('batch-failed', result.failed);
    const sentimentList = $('batch-sentiment-list');
    sentimentList?.replaceChildren();
    ['positive', 'neutral', 'negative'].forEach(label => {
      const aggregate = result.sentiment[label];
      const row = document.createElement('div');
      row.className = 'batch-distribution-row';
      const copy = document.createElement('div');
      const name = document.createElement('span');
      name.textContent = label;
      const count = document.createElement('strong');
      count.textContent = `${aggregate.count} · ${aggregate.percentage.toFixed(1)}%`;
      copy.append(name, count);
      const track = document.createElement('div');
      track.className = 'probability-track';
      const bar = document.createElement('div');
      bar.className = `probability-fill ${label}`;
      bar.style.width = `${Math.max(0, Math.min(100, aggregate.percentage))}%`;
      track.append(bar); row.append(copy, track); sentimentList?.append(row);
    });
    write('confidence-average', ratio(result.confidence.average));
    write('confidence-low', result.confidence.low_confidence_count);
    write('confidence-min', ratio(result.confidence.minimum));
    write('confidence-max', ratio(result.confidence.maximum));
    write('batch-language', `${result.language.code_mixed_count} code-mixed documents · ${result.language.code_mixed_percentage.toFixed(1)}%`);
    renderAggregate('batch-keywords', result.keywords, keyword => [keyword.text, `${keyword.count} occurrences · ${ratio(keyword.average_score)} average score`], 'No aggregate keywords are available.');
    renderAggregate('batch-topics', result.topics, topic => [topic.label || `Topic ${topic.id}`, `${topic.count} documents · ${topic.percentage.toFixed(1)}%`], 'No topics were assigned.');
    write('batch-null-topics', `Unassigned / null topics: ${result.null_topic_count}`);
    write('analytics-state', 'Analytics loaded from the persisted session.');
    show('analytics-panel', true);
  }

  function renderAggregate(id, values, format, emptyMessage) {
    const list = $(id);
    if (!list) return;
    list.replaceChildren();
    if (!values.length) {
      const empty = document.createElement('p'); empty.className = 'analyzer-empty'; empty.textContent = emptyMessage; list.append(empty); return;
    }
    values.forEach(value => {
      const [label, detail] = format(value);
      const row = document.createElement('div'); row.className = 'batch-aggregate-row';
      const title = document.createElement('strong'); title.textContent = label;
      const copy = document.createElement('span'); copy.textContent = detail;
      row.append(title, copy); list.append(row);
    });
  }

  function renderDocuments(result) {
    const list = $('document-rows');
    list?.replaceChildren();
    let privateText = false;
    result.documents.forEach(row => {
      const tr = document.createElement('tr');
      tr.dataset.state = row.status;
      const omitted = row.original_text === null || row.original_text === undefined;
      privateText = privateText || omitted;
      const topic = row.topic?.id === null || row.topic?.id === undefined ? 'Unassigned / unavailable' : `${row.topic.label || 'Topic'} (${row.topic.id})`;
      const values = [
        String(row.row_index + 1),
        omitted ? 'Original text not stored (privacy policy)' : row.original_text || 'Empty input',
        row.sentiment ? `${row.sentiment.label} · ${ratio(row.sentiment.confidence)}${row.sentiment.low_confidence ? ' · Low confidence' : ''}` : 'No prediction',
        Array.isArray(row.keywords) && row.keywords.length ? row.keywords.map(keyword => `${keyword.text} (${ratio(keyword.score)})`).join(', ') : 'Unavailable / no keywords',
        topic,
        row.summary?.text || 'Unavailable',
        row.language ? `${row.language.primary} · ${row.language.is_code_mixed ? 'Code-mixed' : 'Not code-mixed'}` : 'Unavailable',
        integer(row.processing_ms) ? `${row.processing_ms} ms` : 'Unavailable',
        row.status === 'failed' ? `Failed · ${row.error_code || 'analysis_error'}: ${row.error_message || 'This row could not be analyzed.'}` : `Success${row.warnings?.length ? ` · ${row.warnings.join(' ')}` : ''}`,
      ];
      values.forEach(value => { const cell = document.createElement('td'); cell.textContent = value; tr.append(cell); });
      list?.append(tr);
    });
    show('raw-text-note', privateText);
    pageOffset = result.offset;
    pageTotal = result.total_documents;
    const start = result.documents.length ? pageOffset + 1 : 0;
    write('document-page-info', `${start}–${pageOffset + result.documents.length} of ${pageTotal} documents`);
    write('document-state', result.documents.length ? 'Results loaded. Row numbers are data rows, excluding the CSV header.' : 'No documents appear on this page.');
    updatePagination();
  }

  async function loadPage(offset) {
    if (!activeSession || paging) return;
    paging = true;
    updatePagination();
    write('document-state', 'Loading document results…');
    const session = activeSession;
    try {
      const result = await api.session(session, pageSize, offset);
      if (!validSession(result) || result.session.id !== session) throw new Error('The backend returned an invalid session page.');
      if (activeSession === session) renderDocuments(result);
    } catch (error) {
      write('document-state', `Document retrieval failed: ${error.message || 'Try again.'}`);
    } finally { paging = false; updatePagination(); }
  }

  async function loadDashboard(sessionId, batch = null) {
    const requestGeneration = ++generation;
    activeSession = sessionId;
    pageOffset = 0;
    show('analytics-panel', false);
    write('analytics-state', 'Loading analytics…');
    write('export-state', '');
    write('document-state', 'Loading documents…');
    $('document-rows')?.replaceChildren();
    write('document-page-info', 'Loading…');
    show('raw-text-note', false);
    const results = await Promise.allSettled([api.session(sessionId, pageSize, 0), api.analytics(sessionId)]);
    if (requestGeneration !== generation) return;
    const documents = results[0]; const analytics = results[1];
    if (documents.status === 'fulfilled' && validSession(documents.value) && documents.value.session.id === sessionId) {
      renderBatch(batch || documents.value.session);
      renderDocuments(documents.value);
      if (!batch) state(documents.value.session.status === 'completed' ? 'success' : documents.value.session.status === 'failed' ? 'error' : 'partial', `Saved session ${documents.value.session.status}. ${documents.value.session.failed_documents} failed rows.`);
    } else {
      write('document-state', `Document retrieval failed: ${documents.reason?.message || 'Invalid session response.'}`);
      if (!batch) { show('batch-results', false); throw new Error(documents.reason?.message || 'The session could not be loaded.'); }
    }
    if (analytics.status === 'fulfilled' && validAnalytics(analytics.value) && analytics.value.session_id === sessionId) renderAnalytics(analytics.value);
    else write('analytics-state', `Analytics unavailable: ${analytics.reason?.message || 'Invalid analytics response.'} You can retry by loading this session again.`);
  }

  async function analyzeCSV() {
    if (busy || paging) return;
    const error = fileError(selectedFile);
    if (error) { state('validation', error); $('batch-file')?.setAttribute('aria-invalid', 'true'); $('batch-file')?.focus(); return; }
    const column = $('text-column')?.value.trim();
    if (!column) { state('validation', 'Enter the CSV text-column header.'); $('text-column')?.focus(); return; }
    setBusy(true);
    show('batch-results', false);
    show('batch-progress-panel', true);
    $('batch-progress')?.removeAttribute('value');
    write('batch-progress-label', 'Uploading CSV…');
    state('loading', 'Uploading and analyzing the CSV. Large collections may take several minutes.');
    try {
      const result = await api.batch(selectedFile, column, value => {
        if (value === null || value >= 1) $('batch-progress')?.removeAttribute('value');
        else if ($('batch-progress')) $('batch-progress').value = value * 100;
        write('batch-progress-label', value >= 1 ? 'Upload complete. The backend is analyzing and storing rows…' : value === null ? 'Uploading CSV…' : `Uploading CSV · ${Math.round(value * 100)}%`);
      });
      if (!validBatch(result)) throw new Error('The backend returned an invalid batch summary.');
      renderBatch(result);
      state(result.status === 'completed' ? 'success' : result.status === 'partial' ? 'partial' : 'error', `Batch ${result.status}: ${result.successful_documents} successful, ${result.failed_documents} failed, ${result.total_documents} total documents.`);
      await loadDashboard(result.session_id, result);
    } catch (failure) { state('error', `Batch analysis failed: ${failure.message || 'Please try again.'}`); }
    finally { setBusy(false); show('batch-progress-panel', false); }
  }

  function clearCSV() {
    if (busy || paging) return;
    generation++;
    activeSession = null;
    pageTotal = 0;
    pageOffset = 0;
    if ($('batch-file')) $('batch-file').value = '';
    if ($('session-input')) $('session-input').value = '';
    chooseFile(null);
    show('batch-results', false);
    show('batch-progress-panel', false);
    updatePagination();
  }

  async function exportResults(format) {
    if (!activeSession) return;
    const session = activeSession;
    ['export-csv-button', 'export-json-button'].forEach(id => { if ($(id)) $(id).disabled = true; });
    write('export-state', `Downloading backend-generated ${format.toUpperCase()}…`);
    try {
      const result = await api.exportSession(session, format);
      const url = global.URL.createObjectURL(result.blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = result.filename || `mahapulse-${session.replace(/[^a-zA-Z0-9_-]/g, '_')}.${format}`;
      link.hidden = true;
      document.body.append(link); link.click(); link.remove();
      global.setTimeout(() => global.URL.revokeObjectURL(url), 60000);
      write('export-state', `${format.toUpperCase()} file download started.`);
    } catch (error) { write('export-state', `Export failed: ${error.message || 'Please try again.'}`); }
    finally { ['export-csv-button', 'export-json-button'].forEach(id => { if ($(id)) $(id).disabled = false; }); }
  }

  function switchMode(mode) {
    const single = mode === 'single';
    show('single-analysis-panel', single); show('batch-analysis-panel', !single);
    $('single-mode-button')?.setAttribute('aria-pressed', String(single));
    $('batch-mode-button')?.setAttribute('aria-pressed', String(!single));
  }

  function init() {
    if (!$('batch-form') || !api) return;
    $('single-mode-button')?.addEventListener('click', () => switchMode('single'));
    $('batch-mode-button')?.addEventListener('click', () => switchMode('batch'));
    $('batch-file')?.addEventListener('change', event => chooseFile(event.target.files?.[0]));
    $('batch-form')?.addEventListener('submit', event => { event.preventDefault(); analyzeCSV(); });
    $('clear-csv-button')?.addEventListener('click', clearCSV);
    $('session-form')?.addEventListener('submit', async event => {
      event.preventDefault();
      if (busy || paging) return;
      const sessionId = $('session-input')?.value.trim();
      if (!sessionId) { state('validation', 'Enter a saved session ID.'); return; }
      setBusy(true); state('loading', 'Loading saved session…');
      show('batch-results', false);
      try { await loadDashboard(sessionId); }
      catch (error) { state('error', `Session retrieval failed: ${error.message}`); }
      finally { setBusy(false); }
    });
    $('previous-page-button')?.addEventListener('click', () => loadPage(Math.max(0, pageOffset - pageSize)));
    $('next-page-button')?.addEventListener('click', () => loadPage(pageOffset + pageSize));
    $('export-csv-button')?.addEventListener('click', () => exportResults('csv'));
    $('export-json-button')?.addEventListener('click', () => exportResults('json'));
    const dropZone = $('csv-drop-zone');
    dropZone?.addEventListener('dragover', event => { event.preventDefault(); if (!busy) dropZone.dataset.dragging = 'true'; });
    dropZone?.addEventListener('dragleave', () => { dropZone.dataset.dragging = 'false'; });
    dropZone?.addEventListener('drop', event => { event.preventDefault(); dropZone.dataset.dragging = 'false'; if (!busy) chooseFile(event.dataTransfer.files?.[0]); });
    if ($('text-column')) $('text-column').value = config.DEFAULT_TEXT_COLUMN || 'text';
    write('batch-limits', `Configured upload limit: ${(config.MAX_UPLOAD_BYTES / 1000000).toFixed(1)} MB · up to ${config.MAX_BATCH_ROWS.toLocaleString()} rows. Server settings are authoritative.`);
  }

  global.MahaPulseBatch = { fileError, validBatch, validSession, validAnalytics, renderBatch, renderAnalytics, renderDocuments, loadDashboard, loadPage, analyzeCSV, chooseFile, clearCSV, switchMode, exportResults };
  if (global.NLP_COMPONENTS_READY) init();
  else document.addEventListener('nlp:ready', init, { once: true });
})(window);
