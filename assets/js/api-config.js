/**
 * MahaPulse frontend API configuration and fetch boundary.
 * Set MAHAPULSE_RUNTIME_CONFIG before this file for deployment-specific values.
 */
(function (global) {
  'use strict';

  const defaults = {
    API_BASE_URL: 'http://localhost:8000',
    USE_MOCK_API: false,
    REQUEST_TIMEOUT_MS: 12000,
    BATCH_TIMEOUT_MS: 600000,
    MAX_TEXT_LENGTH: 100000,
    MAX_UPLOAD_BYTES: 5000000,
    MAX_BATCH_ROWS: 1000,
    DEFAULT_TEXT_COLUMN: 'text',
    PAGE_SIZE: 25,
  };
  /* Runtime values win; a separately supplied static config may be used by a
     deployment; otherwise the safe localhost default remains in force. */
  const staticConfig = global.MAHAPULSE_STATIC_CONFIG || {};
  const legacyConfig = global.MAHAPULSE_CONFIG || {};
  const runtimeConfig = global.MAHAPULSE_RUNTIME_CONFIG || {};
  const config = Object.assign({}, defaults, legacyConfig, staticConfig, runtimeConfig);
  config.API_BASE_URL = String(config.API_BASE_URL || defaults.API_BASE_URL).replace(/\/+$/, '');
  global.MAHAPULSE_CONFIG = config;

  function apiError(code, message, details, status) {
    const error = new Error(message);
    error.code = code;
    error.details = details;
    error.status = status;
    return error;
  }

  function parsePayload(raw) {
    if (!raw) return null;
    try { return JSON.parse(raw); }
    catch { return null; }
  }

  function responseError(payload, status) {
    const detail = payload?.error?.message || payload?.detail;
    const message = typeof detail === 'string' ? detail
      : Array.isArray(detail) ? detail.map(item => item.msg || 'Invalid request').join('; ')
        : `The analyzer returned HTTP ${status}.`;
    return apiError(status >= 500 ? 'backend_unavailable' : 'api_error', message, payload, status);
  }

  async function request(path, options, download = false) {
    const controller = new AbortController();
    const timeout = global.setTimeout(() => controller.abort(), config.REQUEST_TIMEOUT_MS);
    try {
      const response = await global.fetch(config.API_BASE_URL + path, Object.assign({}, options, {
        signal: controller.signal,
        headers: Object.assign({ Accept: 'application/json' }, options?.headers || {}),
      }));
      if (download && response.ok) {
        const blob = await response.blob();
        const disposition = response.headers.get('Content-Disposition') || '';
        // Ignore any server-supplied path; provide a safe fallback when CORS hides this header.
        const match = /filename="?([^";]+)"?/.exec(disposition);
        return { blob, filename: match ? match[1].replace(/[\\/\r\n]/g, '_') : null };
      }
      const payload = parsePayload(await response.text());
      if (!response.ok) throw responseError(payload, response.status);
      if (payload === null) throw apiError('invalid_json', 'The analyzer returned an unreadable response.', null, response.status);
      return payload;
    } catch (error) {
      if (error?.code) throw error;
      if (error?.name === 'AbortError') {
        throw apiError('timeout', 'The analyzer request timed out. Please try again.', null);
      }
      throw apiError('network_error', 'The analyzer backend could not be reached.', null);
    } finally {
      global.clearTimeout(timeout);
    }

  }

  function uploadBatch(file, textColumn, onProgress) {
    // Batch NLP and persistence always run on the API, including frontend-dev/mock-api.
    return new Promise((resolve, reject) => {
      const xhr = new global.XMLHttpRequest();
      const body = new global.FormData();
      body.append('file', file);
      const query = textColumn ? `?text_column=${encodeURIComponent(textColumn)}` : '';
      xhr.open('POST', config.API_BASE_URL + '/v1/analyze/batch' + query);
      xhr.timeout = config.BATCH_TIMEOUT_MS;
      xhr.setRequestHeader('Accept', 'application/json');
      xhr.upload.onprogress = event => onProgress?.(event.lengthComputable ? event.loaded / event.total : null);
      xhr.upload.onload = () => onProgress?.(1);
      xhr.onerror = () => reject(apiError('network_error', 'The batch backend could not be reached.', null));
      xhr.ontimeout = () => reject(apiError('timeout', 'The batch request timed out. It may still be processing on the server; avoid immediately uploading it again.', null));
      xhr.onload = () => {
        const payload = parsePayload(xhr.responseText);
        if (xhr.status < 200 || xhr.status >= 300) reject(responseError(payload, xhr.status));
        else if (payload === null) reject(apiError('invalid_json', 'The batch backend returned an unreadable response.', null, xhr.status));
        else resolve(payload);
      };
      xhr.send(body);
    });
  }

  function mockAnalysis(text) {
    const lower = text.toLocaleLowerCase();
    const isNegative = /वाईट|नाही|निराश|bad|hate|खराब/.test(lower);
    const isPositive = /छान|चांगल|आवड|मस्त|उत्कृष्ट|good|love/.test(lower);
    const codeMixed = /[A-Za-z]/.test(text) && /[\u0900-\u097f]/.test(text);
    const sentiment = isNegative && !isPositive
      ? { label: 'negative', confidence: .91, probabilities: { positive: .03, negative: .91, neutral: .06 } }
      : isPositive && !isNegative
        ? { label: 'positive', confidence: .93, probabilities: { positive: .93, negative: .02, neutral: .05 } }
        : { label: 'neutral', confidence: .62, probabilities: { positive: .18, negative: .20, neutral: .62 } };
    const letters = [...text].filter(char => /[A-Za-z\u0900-\u097f]/.test(char));
    const devanagari = letters.filter(char => /[\u0900-\u097f]/.test(char)).length;
    const latin = letters.filter(char => /[A-Za-z]/.test(char)).length;
    const keywordError = /keyword[- ]?error|enrichment[- ]?warning|multi[- ]?enrichment[- ]?error/.test(lower);
    const topicError = /topic[- ]?error|enrichment[- ]?warning|multi[- ]?enrichment[- ]?error/.test(lower);
    const summaryError = /summary[- ]?error|enrichment[- ]?warning|multi[- ]?enrichment[- ]?error/.test(lower);
    const disabled = /services[- ]?disabled/.test(lower);
    const warnings = ['Mock API mode is enabled; no production model was called.'];
    if (keywordError) warnings.push('Keyword enrichment unavailable.');
    if (topicError) warnings.push('Topic enrichment unavailable.');
    if (summaryError) warnings.push('Summary enrichment unavailable.');
    return {
      request_id: 'mock-request-0001', original_text: text, model_text: text.replace(/\s+/g, ' ').trim(),
      analysis_text: text.replace(/[^A-Za-z\u0900-\u097f0-9]+/g, ' ').trim().toLocaleLowerCase(),
      language: { primary: 'mr', devanagari_ratio: letters.length ? devanagari / letters.length : 0, latin_ratio: letters.length ? latin / letters.length : 0, is_code_mixed: codeMixed },
      sentiment: Object.assign({}, sentiment, { low_confidence: sentiment.confidence < .6 }),
      keywords: disabled || keywordError ? [] : [{ text: 'अनुभव', score: 0.91 }, { text: codeMixed ? 'app' : 'उत्पादन', score: 0.73 }],
      topic: disabled || topicError || /topic[- ]?null|no[- ]?topic|topic[- ]?outlier/.test(lower)
        ? { id: null, label: null, probability: null }
        : /topic[- ]?assigned/.test(lower) ? { id: 7, label: 'उत्पादन अनुभव', probability: 0.82 } : { id: null, label: null, probability: null },
      summary: disabled || summaryError || !/summary[- ]?available/.test(lower)
        ? { text: null, provider: null }
        : { text: 'Mock extractive summary for frontend verification.', provider: 'extractive' },
      meta: { model_version: 'mock-v0', processing_ms: 1, warnings },
    };
  }

  function mockService(name, overrides) {
    return Object.assign({ name, version: 'mock-v0', device: 'not-loaded', state: 'mocked', backend: 'mock', provider: name, production_ready: null }, overrides || {});
  }

  function mockReadiness() {
    return {
      api: { state: 'ready', detail: 'API process is running' },
      preprocessing: { state: 'ready', detail: 'Deterministic preprocessing is available' },
      sentiment: { state: 'mocked', detail: 'Service is mocked and no model is loaded', production_ready: null },
      keywords: { state: 'mocked', detail: 'Development fixture' },
      topics: { state: 'mocked', detail: 'Development fixture' },
      summary: { state: 'mocked', detail: 'Development fixture' },
      database: { state: 'disabled', detail: 'Browser single-text fixtures have no persistence; batch requests need the HTTP API' },
    };
  }

  function mockModelInfo() {
    const readiness = mockReadiness();
    return {
      sentiment_model: mockService('MuRIL', { provider: 'MuRIL' }),
      labels: ['positive', 'negative', 'neutral'],
      keyword_service: mockService('KeyBERT'),
      topic_service: mockService('BERTopic'),
      summary_service: mockService('summarization', { provider: null }),
      readiness,
    };
  }

  const api = {
    config,
    health() { return config.USE_MOCK_API ? Promise.resolve({ status: 'ok', service: 'mahapulse-api', pipeline: 'phase-2' }) : request('/health'); },
    ready() { return config.USE_MOCK_API ? Promise.resolve({ status: 'degraded', services: mockReadiness() }) : request('/ready'); },
    modelInfo() { return config.USE_MOCK_API ? Promise.resolve(mockModelInfo()) : request('/v1/model-info'); },
    analyze(text) { return config.USE_MOCK_API ? Promise.resolve(mockAnalysis(text)) : request('/v1/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) }); },
    batch(file, textColumn, onProgress) { return uploadBatch(file, textColumn, onProgress); },
    session(id, limit = config.PAGE_SIZE, offset = 0) { return request(`/v1/analyses/${encodeURIComponent(id)}?limit=${limit}&offset=${offset}`); },
    analytics(id) { return request(`/v1/analyses/${encodeURIComponent(id)}/analytics`); },
    exportSession(id, format) {
      if (!['csv', 'json'].includes(format)) return Promise.reject(apiError('validation_error', 'Choose CSV or JSON export.', null));
      return request(`/v1/analyses/${encodeURIComponent(id)}/export?format=${format}`, { headers: { Accept: format === 'csv' ? 'text/csv' : 'application/json' } }, true);
    },
  };

  global.MahaPulseAPI = api;
})(window);
