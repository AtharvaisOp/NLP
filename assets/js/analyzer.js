/** MahaPulse Analyzer page controller. */
(function (global) {
  'use strict';

  const $ = id => document.getElementById(id);
  const config = global.MAHAPULSE_CONFIG;
  const api = global.MahaPulseAPI;
  let submitting = false;

  function text(id, value) {
    const node = $(id);
    if (node) node.textContent = value === null || value === undefined || value === '' ? '—' : String(value);
  }

  function hasValue(value) { return value !== null && value !== undefined && value !== ''; }

  function percent(value) {
    return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
  }

  function isFiniteRatio(value) {
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
  }

  function validAnalysisResponse(result) {
    if (!result || typeof result !== 'object') return false;
    if (![result.request_id, result.original_text, result.model_text, result.analysis_text].every(value => typeof value === 'string')) return false;
    if (!result.request_id.trim()) return false;
    const language = result.language;
    if (!language || typeof language.primary !== 'string' || typeof language.is_code_mixed !== 'boolean' || !isFiniteRatio(language.devanagari_ratio) || !isFiniteRatio(language.latin_ratio)) return false;
    const sentiment = result.sentiment;
    if (!sentiment || !['positive', 'negative', 'neutral'].includes(sentiment.label) || !isFiniteRatio(sentiment.confidence) || typeof sentiment.low_confidence !== 'boolean') return false;
    const probabilities = sentiment.probabilities;
    if (!probabilities || typeof probabilities !== 'object' || !['positive', 'negative', 'neutral'].every(label => isFiniteRatio(probabilities[label]))) return false;
    const total = probabilities.positive + probabilities.negative + probabilities.neutral;
    return Math.abs(total - 1) <= 0.02;
  }

  function setState(kind, title, message) {
    const panel = $('analysis-state');
    if (!panel) return;
    panel.dataset.state = kind;
    panel.replaceChildren();
    const icon = document.createElement('span');
    icon.className = 'analyzer-state-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = kind === 'loading' ? '◌' : ['error', 'validation', 'network', 'offline'].includes(kind) ? '!' : kind === 'partial' ? '◐' : kind === 'success' ? '✓' : 'i';
    const copy = document.createElement('div');
    const heading = document.createElement('strong');
    heading.textContent = title;
    const detail = document.createElement('span');
    detail.textContent = message;
    copy.append(heading, detail);
    panel.append(icon, copy);
    panel.hidden = false;
  }

  function setIndicator(id, state, label) {
    const node = $(id);
    if (!node) return;
    node.dataset.state = state;
    node.textContent = label;
  }

  function serviceLabel(service) {
    if (!service) return 'Unavailable';
    const state = service.state ? ` · ${service.state}` : '';
    const smoke = service.smoke_test === true || service.production_ready === false;
    const lifecycle = smoke ? ' · smoke · not production-ready' : service.state === 'mocked' ? ' · development mock · not production-ready' : '';
    return `${service.name || 'Service'}${service.version ? ` ${service.version}` : ''}${state}${lifecycle}`;
  }

  function compactServiceInfo(label, service) {
    if (!service) return `${label}: unavailable`;
    const parts = [service.name || label];
    if (service.version) parts.push(service.version);
    if (service.backend) parts.push(service.backend);
    if (service.device && service.device !== 'not-loaded') parts.push(service.device);
    if (service.embedding_model) parts.push(`embedding ${service.embedding_model}`);
    if (service.provider && service.provider !== service.name) parts.push(service.provider);
    if (service.smoke_test === true || service.production_ready === false) parts.push('smoke · not production-ready');
    return `${label}: ${parts.join(' · ')}`;
  }

  function enrichmentSummary(readiness, modelInfo) {
    const infoKeys = { keywords: 'keyword_service', topics: 'topic_service', summary: 'summary_service' };
    const services = ['keywords', 'topics', 'summary'].map(name => ({
      name,
      ...(readiness?.[name] || modelInfo?.readiness?.[name] || modelInfo?.[infoKeys[name]] || {}),
    }));
    const states = services.map(service => service.state || 'unavailable');
    if (states.every(state => state === 'disabled')) return { kind: 'mocked', label: 'Enrichment disabled' };
    if (states.every(state => state === 'ready')) return { kind: 'ready', label: 'Enrichment ready' };
    if (states.some(state => ['unavailable', 'not_ready', 'not_loaded'].includes(state))) {
      return { kind: 'mocked', label: 'Enrichment partial' };
    }
    if (states.some(state => state === 'mocked')) return { kind: 'mocked', label: 'Development mocks active' };
    if (states.some(state => state === 'disabled')) return { kind: 'mocked', label: 'Enrichment partial · some disabled' };
    return { kind: 'mocked', label: 'Enrichment status unavailable' };
  }

  function renderModelInfo(modelInfo) {
    const node = $('service-model-info');
    if (!node) return;
    node.textContent = [
      compactServiceInfo('Sentiment', modelInfo?.sentiment_model),
      compactServiceInfo('Keywords', modelInfo?.keyword_service),
      compactServiceInfo('Topics', modelInfo?.topic_service),
      compactServiceInfo('Summary', modelInfo?.summary_service),
    ].join('  |  ');
  }

  async function loadServiceStatus() {
    setIndicator('backend-status', 'checking', 'Checking…');
    setIndicator('model-status', 'checking', 'Checking…');
    const results = await Promise.allSettled([api.health(), api.ready(), api.modelInfo()]);
    const health = results[0];
    const ready = results[1];
    const model = results[2];
    const readiness = ready.status === 'fulfilled' ? ready.value : null;
    const modelInfo = model.status === 'fulfilled' ? model.value : null;
    const sentimentState = readiness?.services?.sentiment?.state || modelInfo?.sentiment_model?.state;
    const sentimentLifecycle = modelInfo?.sentiment_model || readiness?.services?.sentiment;
    const smoke = sentimentLifecycle?.smoke_test === true || sentimentLifecycle?.production_ready === false;
    const enrichment = enrichmentSummary(readiness?.services, modelInfo);
    if (health.status === 'fulfilled') {
      const coreReady = sentimentState === 'ready';
      const coreLabel = coreReady
        ? (smoke ? 'Core sentiment smoke' : 'Core sentiment ready')
        : sentimentState === 'mocked' ? 'Core sentiment mocked' : 'Core sentiment unavailable';
      const suffix = enrichment.label;
      setIndicator('backend-status', coreReady && !smoke && enrichment.kind === 'ready' ? 'ready' : 'mocked', `Online · ${coreLabel} · ${suffix}`);
    } else {
      setIndicator('backend-status', 'offline', 'Unavailable');
    }
    setIndicator('enrichment-status', enrichment.kind, enrichment.label);
    const optional = { keywords: ['keyword-status', 'keyword_service'], topics: ['topic-status', 'topic_service'], summary: ['summary-status', 'summary_service'] };
    Object.entries(optional).forEach(([key, [id, infoKey]]) => {
      const info = modelInfo?.[infoKey];
      const service = readiness?.services?.[key] || modelInfo?.readiness?.[key] || info;
      const state = service?.state || 'unavailable';
      const label = state === 'ready' ? `${info?.name || key} ready`
        : state === 'mocked' ? `${info?.name || key} · mock`
          : state === 'disabled' ? 'Disabled'
            : key === 'topics' && ['unavailable', 'not_loaded', 'not_ready'].includes(state) ? 'BERTopic artifact unavailable' : 'Unavailable';
      setIndicator(id, state === 'ready' ? 'ready' : ['disabled', 'mocked'].includes(state) ? 'mocked' : 'offline', label);
    });
    const database = readiness?.services?.database;
    setIndicator('database-status', database?.state === 'ready' ? 'ready' : database?.state === 'disabled' ? 'mocked' : 'offline', database?.state === 'ready' ? 'Ready · persistence available' : database?.state === 'disabled' ? 'Not required for stateless mode · batch unavailable' : 'Unavailable · batch needs persistence');
    if (model.status === 'fulfilled') {
      const info = model.value.sentiment_model;
      setIndicator('model-status', info?.state === 'ready' && !smoke ? 'ready' : ['ready', 'mocked'].includes(info?.state) ? 'mocked' : 'offline', serviceLabel(info));
      renderModelInfo(model.value);
    } else {
      setIndicator('model-status', 'offline', 'Not available');
      renderModelInfo(null);
    }
    const note = $('mock-mode-note');
    if (note) {
      const mockActive = config.USE_MOCK_API || sentimentState === 'mocked' || enrichment.label === 'Development mocks active';
      note.hidden = !mockActive && !smoke;
      note.textContent = smoke
        ? 'Development/Smoke Model · Not Production Ready. Predictions are for integration testing; smoke metrics are not project performance.'
        : 'Development Mock Services · Not Production Ready. Mock outputs are deterministic demonstrations, not model performance. Optional enrichment may be disabled or unavailable.';
    }
  }

  function renderProbabilities(probabilities) {
    const names = ['positive', 'neutral', 'negative'];
    names.forEach(name => {
      const value = probabilities && typeof probabilities[name] === 'number' ? probabilities[name] : null;
      text(`prob-${name}-value`, percent(value));
      const bar = $(`prob-${name}-bar`);
      if (bar) {
        bar.style.width = value === null ? '0%' : `${Math.max(0, Math.min(1, value)) * 100}%`;
        bar.setAttribute('aria-valuenow', value === null ? '0' : String(Math.round(value * 100)));
      }
    });
  }

  function renderKeywords(keywords) {
    const list = $('keyword-list');
    if (!list) return;
    list.replaceChildren();
    if (!Array.isArray(keywords) || keywords.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'analyzer-empty';
      empty.textContent = 'No keywords were returned for this analysis.';
      list.append(empty);
      return;
    }
    keywords.forEach(keyword => {
      const chip = document.createElement('span');
      chip.className = 'keyword-chip';
      chip.textContent = keyword?.text || 'Unnamed keyword';
      const score = document.createElement('span');
      score.className = 'keyword-score';
      score.textContent = typeof keyword?.score === 'number' ? keyword.score.toFixed(2) : '—';
      chip.append(score);
      list.append(chip);
    });
  }

  function renderTopic(topic) {
    const unavailable = !topic || (!hasValue(topic.id) && !hasValue(topic.label) && !hasValue(topic.probability));
    const empty = $('topic-empty');
    const data = $('topic-data');
    if (empty) empty.hidden = !unavailable;
    if (data) data.hidden = unavailable;
    if (!unavailable) {
      text('topic-id', topic.id);
      text('topic-label', topic.label);
      text('topic-probability', percent(topic.probability));
    }
    return unavailable;
  }

  function renderSummary(summary) {
    const unavailable = !summary || !hasValue(summary.text);
    const empty = $('summary-empty');
    const data = $('summary-data');
    if (empty) empty.hidden = !unavailable;
    if (data) data.hidden = unavailable;
    if (!unavailable) {
      text('summary-text', summary.text);
      text('summary-provider', summary.provider || 'Provider not specified');
    }
    return unavailable;
  }

  function renderWarnings(warnings) {
    const list = $('warning-list');
    if (!list) return;
    list.replaceChildren();
    if (!Array.isArray(warnings) || warnings.length === 0) {
      list.hidden = true;
      return;
    }
    warnings.forEach(warning => {
      const item = document.createElement('li');
      item.textContent = warning;
      list.append(item);
    });
    list.hidden = false;
  }

  function renderResponse(result) {
    const language = result.language || {};
    const sentiment = result.sentiment || {};
    const meta = result.meta || {};
    const warnings = Array.isArray(meta.warnings) ? meta.warnings : [];
    text('sentiment-label', sentiment.label);
    text('sentiment-confidence', percent(sentiment.confidence));
    renderProbabilities(sentiment.probabilities);
    const confidenceWarning = $('confidence-warning');
    if (confidenceWarning) confidenceWarning.hidden = !sentiment.low_confidence;
    text('language-primary', language.primary);
    text('language-devanagari', percent(language.devanagari_ratio));
    text('language-latin', percent(language.latin_ratio));
    text('language-mixed', language.is_code_mixed ? 'Yes · code-mixed' : 'No');
    renderKeywords(result.keywords);
    const topicUnavailable = renderTopic(result.topic);
    const summaryUnavailable = renderSummary(result.summary);
    text('pipeline-original', result.original_text);
    text('pipeline-model', result.model_text);
    text('pipeline-analysis', result.analysis_text);
    text('request-id', result.request_id);
    text('model-version', meta.model_version);
    text('processing-ms', typeof meta.processing_ms === 'number' ? `${meta.processing_ms} ms` : null);
    renderWarnings(warnings);
    const partial = warnings.length > 0 || topicUnavailable || summaryUnavailable || !Array.isArray(result.keywords) || result.keywords.length === 0;
    setState(partial ? 'partial' : 'success', partial ? 'Partial enrichment result' : 'Analysis complete', partial ? 'Sentiment and language results are available; optional enrichment may not be assigned yet.' : 'The response was returned by the MahaPulse analysis contract.');
    $('results-panel')?.removeAttribute('hidden');
  }

  function classifyError(error) {
    if (error?.code === 'timeout') return ['network', 'Request timed out', error.message];
    if (error?.code === 'backend_unavailable' || error?.code === 'network_error') return ['offline', 'Backend unavailable', error.message];
    if (error?.code === 'invalid_backend_response') return ['error', 'Invalid backend response', error.message];
    return ['error', 'Analysis failed safely', error?.message || 'The backend returned an unexpected response.'];
  }

  async function analyze() {
    if (submitting) return;
    const input = $('analysis-text');
    const raw = input?.value || '';
    const trimmed = raw.trim();
    if (!trimmed) {
      setState('validation', 'Validation error', 'Enter Marathi or Marathi-English text before analyzing.');
      input?.focus();
      return;
    }
    if (raw.length > config.MAX_TEXT_LENGTH) {
      setState('validation', 'Validation error', `Keep the input below ${config.MAX_TEXT_LENGTH.toLocaleString()} characters.`);
      input?.focus();
      return;
    }
    submitting = true;
    $('analyze-button').disabled = true;
    $('clear-button').disabled = true;
    setState('loading', 'Analyzing text', 'Preparing the two text paths and requesting sentiment results…');
    try {
      const result = await api.analyze(raw);
      if (!validAnalysisResponse(result)) {
        const invalid = new Error('The backend response is missing required analysis fields.');
        invalid.code = 'invalid_backend_response';
        throw invalid;
      }
      renderResponse(result);
    } catch (error) {
      const [kind, title, message] = classifyError(error);
      setState(kind, title, message);
    } finally {
      submitting = false;
      $('analyze-button').disabled = false;
      $('clear-button').disabled = false;
    }
  }

  function clearInput() {
    const input = $('analysis-text');
    if (input) input.value = '';
    updateCounter();
    setState('idle', 'Ready to analyze', 'Add a Marathi or Marathi-English example to inspect the pipeline.');
    $('results-panel')?.setAttribute('hidden', '');
    input?.focus();
  }

  function updateCounter() {
    const input = $('analysis-text');
    text('character-count', `${(input?.value.length || 0).toLocaleString()} / ${config.MAX_TEXT_LENGTH.toLocaleString()}`);
  }

  function init() {
    if (!$('analysis-text') || !api) return;
    $('analyzer-form').addEventListener('submit', event => {
      event.preventDefault();
      analyze();
    });
    $('analyze-button').addEventListener('click', analyze);
    $('clear-button').addEventListener('click', clearInput);
    $('analysis-text').addEventListener('input', updateCounter);
    $('analysis-text').addEventListener('keydown', event => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        analyze();
      }
    });
    document.querySelectorAll('[data-example]').forEach(button => {
      button.addEventListener('click', () => {
        $('analysis-text').value = button.dataset.example || '';
        updateCounter();
        $('analysis-text').focus();
      });
    });
    updateCounter();
    setState('idle', 'Ready to analyze', 'Add a Marathi or Marathi-English example to inspect the pipeline.');
    loadServiceStatus();
  }

  global.MahaPulseAnalyzer = {
    analyze,
    renderProbabilities,
    renderKeywords,
    renderTopic,
    renderSummary,
    renderWarnings,
    renderResponse,
    validateAnalysisResponse: validAnalysisResponse,
    serviceLabel,
    enrichmentSummary,
    loadServiceStatus,
  };
  if (global.NLP_COMPONENTS_READY) init();
  else document.addEventListener('nlp:ready', init, { once: true });
})(window);
