/**
 * NLP Hub — Search Module
 * ─────────────────────────────────────────────────────────────
 * Provides local-first, keyboard-navigable search over a
 * predefined index of NLP concepts and pages.
 *
 * Deps: main.js (NLP.Search.open/close must be available)
 */
(function (global) {
  'use strict';

  /* ── Search Data Index ──────────────────────────────────── */
  /* Each item: { type, title, category, difficulty?, href, tags? } */
  const INDEX = [
    /* ── Concepts ── */
    { type:'concept', title:'Sentence Segmentation',    category:'Text Preprocessing',   difficulty:'Beginner',     href:'concept/index.html?id=sentence-segmentation', tags:['sentence','split','segmentation','boundary','disambiguation'] },
    { type:'concept', title:'Tokenization',             category:'Text Preprocessing',   difficulty:'Beginner',     href:'concept/index.html?id=tokenization',          tags:['tokens','split','words','subwords','characters'] },
    { type:'concept', title:'Case Normalization',       category:'Text Preprocessing',   difficulty:'Beginner',     href:'concept/index.html?id=case-normalization',    tags:['casing','lowercase','uppercase','normalization'] },
    { type:'concept', title:'Stop-word Removal',        category:'Text Preprocessing',   difficulty:'Beginner',     href:'concept/index.html?id=stop-word-removal',     tags:['filtering','stopwords','noise'] },
    { type:'concept', title:'Stemming',                 category:'Text Preprocessing',   difficulty:'Beginner',     href:'concept/index.html?id=stemming',              tags:['normalization','suffix','porter','stem'] },
    { type:'concept', title:'Lemmatization',            category:'Text Preprocessing',   difficulty:'Beginner',     href:'concept/index.html?id=lemmatization',         tags:['normalization','lemma','dictionary','pos'] },
    { type:'concept', title:'Noise Removal',            category:'Text Preprocessing',   difficulty:'Beginner',     href:'concept/index.html?id=noise-removal',         tags:['cleaning','html','xml','tags'] },
    { type:'concept', title:'Text Cleaning',            category:'Text Preprocessing',   difficulty:'Beginner',     href:'concept/index.html?id=text-cleaning',         tags:['emojis','spelling','contractions','cleaning'] },

    { type:'concept', title:'Bag of Words',             category:'Feature Engineering',  difficulty:'Beginner',     href:'concept/index.html?id=bag-of-words',          tags:['bow','features','vector','frequency'] },
    { type:'concept', title:'N-Grams',                  category:'Feature Engineering',  difficulty:'Beginner',     href:'concept/index.html?id=n-grams',               tags:['bigrams','trigrams','context','window'] },
    { type:'concept', title:'TF (Term Frequency)',      category:'Feature Engineering',  difficulty:'Beginner',     href:'concept/index.html?id=tf',                    tags:['term','frequency','weight'] },
    { type:'concept', title:'IDF (Inverse Doc Freq)',   category:'Feature Engineering',  difficulty:'Intermediate', href:'concept/index.html?id=idf',                   tags:['inverse','document','rarity','weight'] },
    { type:'concept', title:'TF-IDF',                   category:'Feature Engineering',  difficulty:'Intermediate', href:'concept/index.html?id=tf-idf',                tags:['tfidf','frequency','weight','features'] },

    { type:'concept', title:'One-Hot Encoding',         category:'Language Representation', difficulty:'Beginner',  href:'concept/index.html?id=one-hot-encoding',      tags:['categorical','vector','sparse','onehot'] },
    { type:'concept', title:'Word Embeddings',          category:'Language Representation', difficulty:'Intermediate',href:'concept/index.html?id=word-embeddings',       tags:['dense','vector','continuous','similarity'] },
    { type:'concept', title:'Word2Vec',                 category:'Language Representation', difficulty:'Intermediate',href:'concept/index.html?id=word2vec',              tags:['embeddings','skip-gram','cbow','negative-sampling'] },
    { type:'concept', title:'FastText',                 category:'Language Representation', difficulty:'Intermediate',href:'concept/index.html?id=fasttext',              tags:['subword','character','ngrams','facebook'] },
    { type:'concept', title:'Contextual Embeddings',     category:'Language Representation', difficulty:'Advanced',    href:'concept/index.html?id=contextual-embeddings',  tags:['dynamic','bert','elmo','polysemy'] },

    { type:'concept', title:'Statistical Language Models', category:'Language Models',    difficulty:'Intermediate', href:'concept/index.html?id=statistical-language-models', tags:['markov','smoothing','unseen','ngrams'] },
    { type:'concept', title:'Neural Language Models',      category:'Language Models',    difficulty:'Intermediate', href:'concept/index.html?id=neural-language-models',      tags:['lstm','rnn','softmax','dense'] },
    { type:'concept', title:'Transformer Representations', category:'Language Models',    difficulty:'Advanced',     href:'concept/index.html?id=transformer-representations', tags:['attention','self-attention','bert','gpt'] },

    /* ── Comparisons ── */
    { type:'compare', title:'Stemming vs Lemmatization',                  category:'Text Preprocessing',     href:'compare/index.html?id=stemming-vs-lemmatization' },
    { type:'compare', title:'TF vs TF-IDF',                               category:'Feature Engineering',    href:'compare/index.html?id=tf-vs-tfidf' },
    { type:'compare', title:'Word2Vec vs FastText',                       category:'Language Representation',href:'compare/index.html?id=word2vec-vs-fasttext' },
    { type:'compare', title:'Statistical vs Neural Language Models',      category:'Language Models',        href:'compare/index.html?id=statistical-vs-neural' },
    { type:'compare', title:'One-Hot Encoding vs Word Embeddings',        category:'Language Representation',href:'compare/index.html?id=onehot-vs-embeddings' },

    /* ── Pages ── */
    { type:'page', title:'Repository Dashboard', category:'Browse all NLP concepts',        href:'repository/index.html' },
    { type:'page', title:'NLP Pipeline Workflow',category:'Interactive step-by-step visual', href:'workflows/index.html' },
    { type:'page', title:'Applications',         category:'Real-world NLP use cases',        href:'applications/index.html' },
    { type:'page', title:'Research & Insights',  category:'History & landmark papers',        href:'research/index.html' },
    { type:'page', title:'Sustainability',        category:'Green AI & carbon footprint',      href:'sustainability/index.html' },
    { type:'page', title:'Reflection',           category:'Learning journey',                 href:'reflection/index.html' },
    { type:'page', title:'References',           category:'Bibliography & citations',          href:'references/index.html' },
    { type:'page', title:'Contribution Matrix',  category:'Team contributions',               href:'team/index.html' },
    { type:'page', title:'AI Usage Declaration',  category:'Academic AI disclosure',           href:'ai-usage/index.html' },
  ];

  /* ── Icons per type ─────────────────────────────────────── */
  const TYPE_META = {
    concept: { icon:'📄', bg:'rgba(99,102,241,.14)' },
    compare: { icon:'⚖️', bg:'rgba(16,185,129,.14)' },
    page:    { icon:'🔖', bg:'rgba(245,158,11,.14)'  },
  };

  const DIFF_COLOURS = {
    Beginner:     'var(--success)',
    Intermediate: 'var(--warning)',
    Advanced:     'var(--danger)',
  };

  /* ── State ──────────────────────────────────────────────── */
  let query        = '';
  let results      = [];
  let selectedIdx  = -1;
  let debounceTimer = null;

  /* ── Helpers ────────────────────────────────────────────── */
  const $ = id => document.getElementById(id);

  /** Compute a relevance score for an item against the query. */
  function score(item, q) {
    const needle = q.toLowerCase();
    const title  = item.title.toLowerCase();
    const cat    = item.category.toLowerCase();
    const tags   = (item.tags || []).join(' ').toLowerCase();

    if (title === needle)             return 100;
    if (title.startsWith(needle))     return 80;
    if (title.includes(needle))       return 60;
    if (cat.includes(needle))         return 40;
    if (tags.includes(needle))        return 20;
    return 0;
  }

  /** Highlight the matched portion of text. */
  function highlight(text, q) {
    if (!q) return text;
    const idx = text.toLowerCase().indexOf(q.toLowerCase());
    if (idx === -1) return text;
    return (
      text.slice(0, idx) +
      `<mark style="background:rgba(99,102,241,.25);color:var(--primary);border-radius:2px;">${text.slice(idx, idx + q.length)}</mark>` +
      text.slice(idx + q.length)
    );
  }

  /** Build ROOT prefix from page depth. */
  function getRoot() {
    const cfg = global.NLP_CONFIG || {};
    return '../'.repeat(cfg.depth || 0);
  }

  /* ── Render results into #search-results ─────────────────── */
  function renderResults() {
    const container = $('search-results');
    if (!container) return;

    /* Empty state */
    if (query && results.length === 0) {
      container.innerHTML = `
        <div style="padding:2rem;text-align:center;color:var(--text-muted);">
          <div style="font-size:2rem;margin-bottom:.75rem;">🔍</div>
          <p style="font-size:.875rem;margin:0;">No results for <strong style="color:var(--text);">"${query}"</strong></p>
          <p style="font-size:.75rem;margin:.5rem 0 0;color:var(--text-muted);">Try different keywords</p>
        </div>`;
      return;
    }

    /* Group results */
    const byType = {};
    results.forEach(item => {
      (byType[item.type] = byType[item.type] || []).push(item);
    });

    const groupLabels = { concept:'Concepts', compare:'Comparisons', page:'Pages' };
    const ROOT = getRoot();

    let html = '';

    /* If no query → show suggestions */
    if (!query) {
      const suggested = INDEX.filter(i => i.type === 'concept').slice(0, 4);
      const pages     = INDEX.filter(i => i.type === 'page').slice(0, 3);

      html += `<div class="results-group-label">Suggested Concepts</div>`;
      suggested.forEach(item => { html += itemHTML(item, '', ROOT); });
      html += `<div class="results-group-label" style="margin-top:.5rem;">Pages</div>`;
      pages.forEach(item => { html += itemHTML(item, '', ROOT); });
    } else {
      Object.entries(byType).forEach(([type, items]) => {
        html += `<div class="results-group-label">${groupLabels[type] || type}</div>`;
        items.slice(0, 6).forEach(item => { html += itemHTML(item, query, ROOT); });
      });
    }

    container.innerHTML = html;
    selectedIdx = -1;
    syncSelection();
  }

  function itemHTML(item, q, ROOT) {
    const meta    = TYPE_META[item.type] || TYPE_META.concept;
    const diffCol = item.difficulty ? `<span style="font-size:.6875rem;color:${DIFF_COLOURS[item.difficulty] || 'var(--text-muted)'};">● ${item.difficulty}</span>` : '';

    return `
    <a href="${ROOT}${item.href}" class="result-item" role="option" tabindex="-1">
      <div class="result-icon" style="background:${meta.bg};">${meta.icon}</div>
      <div style="min-width:0;flex:1;">
        <p class="result-title">${highlight(item.title, q)}</p>
        <p class="result-meta">${item.category}${diffCol ? ' &nbsp;·&nbsp; ' + diffCol : ''}</p>
      </div>
    </a>`;
  }

  /* ── Keyboard navigation within results ──────────────────── */
  function syncSelection() {
    const items = [...($('search-results')?.querySelectorAll('.result-item') || [])];
    items.forEach((el, i) => {
      el.classList.toggle('selected', i === selectedIdx);
      if (i === selectedIdx) el.scrollIntoView({ block:'nearest' });
    });
  }

  function moveSelection(dir) {
    const items = [...($('search-results')?.querySelectorAll('.result-item') || [])];
    if (!items.length) return;
    selectedIdx = (selectedIdx + dir + items.length) % items.length;
    syncSelection();
  }

  function selectCurrent() {
    const items = [...($('search-results')?.querySelectorAll('.result-item') || [])];
    if (selectedIdx >= 0 && items[selectedIdx]) items[selectedIdx].click();
  }

  /* ── Search function ─────────────────────────────────────── */
  function runSearch(q) {
    query   = q.trim();
    results = [];

    if (query.length >= 1) {
      results = INDEX
        .map(item => ({ item, s: score(item, query) }))
        .filter(x => x.s > 0)
        .sort((a, b) => b.s - a.s)
        .map(x => x.item);
    }

    renderResults();
  }

  /* ── Init ────────────────────────────────────────────────── */
  function init() {
    const input = $('search-input');
    if (!input) return;

    /* Render default suggestions immediately */
    renderResults();

    /* Input with debounce */
    input.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => runSearch(input.value), 150);
    });

    /* Keyboard navigation */
    input.addEventListener('keydown', e => {
      switch (e.key) {
        case 'ArrowDown': e.preventDefault(); moveSelection(+1); break;
        case 'ArrowUp':   e.preventDefault(); moveSelection(-1); break;
        case 'Enter':     e.preventDefault(); selectCurrent();   break;
      }
    });

    /* Close when a result is clicked */
    $('search-results')?.addEventListener('click', () => {
      global.NLP?.Search?.close?.();
    });

    /* Clear input when modal opens */
    const overlay = $('search-overlay');
    if (overlay) {
      const observer = new MutationObserver(() => {
        if (overlay.classList.contains('active')) {
          input.value = '';
          query       = '';
          selectedIdx = -1;
          renderResults();
          input.focus();
        }
      });
      observer.observe(overlay, { attributes: true, attributeFilter: ['class'] });
    }
  }

  /* ── Auto-init after DOM ready ───────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 0);
  }

})(window);
