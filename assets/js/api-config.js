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
    MAX_TEXT_LENGTH: 100000,
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

  async function request(path, options) {
    const controller = new AbortController();
    const timeout = global.setTimeout(() => controller.abort(), config.REQUEST_TIMEOUT_MS);
    let response;
    try {
      response = await global.fetch(config.API_BASE_URL + path, Object.assign({}, options, {
        signal: controller.signal,
        headers: Object.assign({ Accept: 'application/json' }, options?.headers || {}),
      }));
    } catch (error) {
      if (error?.name === 'AbortError') {
        throw apiError('timeout', 'The analyzer request timed out. Please try again.', null);
      }
      throw apiError('network_error', 'The analyzer backend could not be reached.', null);
    } finally {
      global.clearTimeout(timeout);
    }

    const raw = await response.text();
    const payload = parsePayload(raw);
    if (!response.ok) {
      const serverMessage = payload?.error?.message || payload?.detail;
      throw apiError(
        response.status >= 500 ? 'backend_unavailable' : 'api_error',
        serverMessage || `The analyzer returned HTTP ${response.status}.`,
        payload,
        response.status,
      );
    }
    if (payload === null) throw apiError('invalid_json', 'The analyzer returned an unreadable response.', null, response.status);
    return payload;
  }

  function mockAnalysis(text) {
    const lower = text.toLocaleLowerCase();
    const isNegative = /वाईट|नाही|निराश|bad|hate|खराब/.test(lower);
    const isPositive = /छान|चांगले|आवड|मस्त|उत्कृष्ट|good|love/.test(lower);
    const codeMixed = /[A-Za-z]/.test(text) && /[\u0900-\u097f]/.test(text);
    const sentiment = isNegative && !isPositive
      ? { label: 'negative', confidence: .91, probabilities: { positive: .03, negative: .91, neutral: .06 } }
      : isPositive && !isNegative
        ? { label: 'positive', confidence: .93, probabilities: { positive: .93, negative: .02, neutral: .05 } }
        : { label: 'neutral', confidence: .62, probabilities: { positive: .18, negative: .20, neutral: .62 } };
    const letters = [...text].filter(char => /[A-Za-z\u0900-\u097f]/.test(char));
    const devanagari = letters.filter(char => /[\u0900-\u097f]/.test(char)).length;
    const latin = letters.filter(char => /[A-Za-z]/.test(char)).length;
    return {
      request_id: 'mock-request-0001', original_text: text, model_text: text.replace(/\s+/g, ' ').trim(),
      analysis_text: text.replace(/[^A-Za-z\u0900-\u097f0-9]+/g, ' ').trim().toLocaleLowerCase(),
      language: { primary: 'mr', devanagari_ratio: letters.length ? devanagari / letters.length : 0, latin_ratio: letters.length ? latin / letters.length : 0, is_code_mixed: codeMixed },
      sentiment: Object.assign({}, sentiment, { low_confidence: sentiment.confidence < .6 }),
      keywords: [], topic: { id: null, label: null, probability: null }, summary: { text: null, provider: null },
      meta: { model_version: 'mock-v0', processing_ms: 1, warnings: ['Mock API mode is enabled; no production model was called.'] },
    };
  }

  const api = {
    config,
    health() { return config.USE_MOCK_API ? Promise.resolve({ status: 'ok', service: 'mock-api' }) : request('/health'); },
    ready() { return config.USE_MOCK_API ? Promise.resolve({ status: 'ready', services: {} }) : request('/ready'); },
    modelInfo() {
      return config.USE_MOCK_API
        ? Promise.resolve({ sentiment_model: { name: 'MuRIL', version: 'mock-v0', device: 'not-loaded', state: 'mocked' }, labels: ['positive', 'negative', 'neutral'], keyword_service: { name: 'KeyBERT', version: 'mock-v0', device: 'not-loaded', state: 'mocked' }, topic_service: { name: 'BERTopic', version: 'mock-v0', device: 'not-loaded', state: 'mocked' }, summary_service: { name: 'summarization', version: 'mock-v0', device: 'not-loaded', state: 'mocked' }, readiness: {} })
        : request('/v1/model-info');
    },
    analyze(text) { return config.USE_MOCK_API ? Promise.resolve(mockAnalysis(text)) : request('/v1/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) }); },
  };

  global.MahaPulseAPI = api;
})(window);
