/**
 * NLP Hub — Component Injection
 * ─────────────────────────────────────────────────────────────
 * Generates and injects the Navbar, Sidebar, Footer, Search
 * Modal, and Breadcrumb into placeholder <div> containers.
 *
 * Each page defines window.NLP_CONFIG before this script loads:
 *   window.NLP_CONFIG = {
 *     page:        'repository',   // active nav/sidebar key
 *     depth:       1,              // directory depth from root
 *     breadcrumbs: [               // [{label, href}, ...]
 *       { label: 'Repository', href: '../repository/' },
 *       { label: 'Current Page' }  // last item = current
 *     ]
 *   };
 */
(function (global) {
  'use strict';

  /* ── Config ─────────────────────────────────────────────── */
  const CFG   = global.NLP_CONFIG || { page: 'home', depth: 0, breadcrumbs: [] };
  const ROOT  = '../'.repeat(CFG.depth || 0);   // path back to site root

  /* ── Top navigation links ───────────────────────────────── */
  const TOP_NAV = [
    { id: 'repository',   label: 'Repository',   href: 'repository/index.html' },
    { id: 'compare',      label: 'Compare',       href: 'compare/index.html'    },
    { id: 'workflows',    label: 'Workflows',     href: 'workflows/index.html'  },
    { id: 'applications', label: 'Applications',  href: 'applications/index.html' },
    { id: 'research',     label: 'Research',      href: 'research/index.html'   },
  ];

  /* ── Sidebar navigation tree ────────────────────────────── */
  const SIDEBAR = [
    {
      label: 'Getting Started',
      items: [
        { id: 'repository',    label: 'Repository Home',    href: 'repository/index.html',    icon: '⊞' },
        { id: 'design-system', label: 'Design System',      href: 'design-system/index.html', icon: '◈' },
      ],
    },
    {
      label: 'Text Preprocessing',
      items: [
        { id: 'concept-sentence-segmentation', label: 'Sentence Segmentation', href: 'concept/index.html?id=sentence-segmentation', icon: '◉' },
        { id: 'concept-tokenization',          label: 'Tokenization',          href: 'concept/index.html?id=tokenization',          icon: '◉' },
        { id: 'concept-case-normalization',    label: 'Case Normalization',    href: 'concept/index.html?id=case-normalization',    icon: '◉' },
        { id: 'concept-stop-word-removal',     label: 'Stop-word Removal',     href: 'concept/index.html?id=stop-word-removal',     icon: '◉' },
        { id: 'concept-stemming',              label: 'Stemming',              href: 'concept/index.html?id=stemming',              icon: '◉' },
        { id: 'concept-lemmatization',          label: 'Lemmatization',          href: 'concept/index.html?id=lemmatization',          icon: '◉' },
        { id: 'concept-noise-removal',          label: 'Noise Removal',          href: 'concept/index.html?id=noise-removal',          icon: '◉' },
        { id: 'concept-text-cleaning',          label: 'Text Cleaning',          href: 'concept/index.html?id=text-cleaning',          icon: '◉' },
      ],
    },
    {
      label: 'Feature Engineering',
      items: [
        { id: 'concept-bag-of-words', label: 'Bag of Words', href: 'concept/index.html?id=bag-of-words', icon: '◉' },
        { id: 'concept-n-grams',      label: 'N-Grams',      href: 'concept/index.html?id=n-grams',      icon: '◉' },
        { id: 'concept-tf',           label: 'TF',           href: 'concept/index.html?id=tf',           icon: '◉' },
        { id: 'concept-idf',          label: 'IDF',          href: 'concept/index.html?id=idf',          icon: '◉' },
        { id: 'concept-tf-idf',       label: 'TF-IDF',       href: 'concept/index.html?id=tf-idf',       icon: '◉' },
      ],
    },
    {
      label: 'Language Representation',
      items: [
        { id: 'concept-one-hot-encoding', label: 'One-Hot Encoding', href: 'concept/index.html?id=one-hot-encoding', icon: '◉' },
        { id: 'concept-word-embeddings',  label: 'Word Embeddings',  href: 'concept/index.html?id=word-embeddings',  icon: '◉' },
        { id: 'concept-word2vec',         label: 'Word2Vec',         href: 'concept/index.html?id=word2vec',         icon: '◉' },
        { id: 'concept-fasttext',         label: 'FastText',         href: 'concept/index.html?id=fasttext',         icon: '◉' },
        { id: 'concept-contextual-embeddings', label: 'Contextual Embeddings', href: 'concept/index.html?id=contextual-embeddings', icon: '◉' },
      ],
    },
    {
      label: 'Language Models',
      items: [
        { id: 'concept-statistical-language-models', label: 'Statistical Models', href: 'concept/index.html?id=statistical-language-models', icon: '◉' },
        { id: 'concept-neural-language-models',      label: 'Neural Models',      href: 'concept/index.html?id=neural-language-models',      icon: '◉' },
        { id: 'concept-transformer-representations', label: 'Transformer Reps',   href: 'concept/index.html?id=transformer-representations', icon: '◉' },
      ],
    },
    {
      label: 'Compare & Visualize',
      items: [
        { id: 'compare',   label: 'Model Comparisons', href: 'compare/index.html',   icon: '⇌' },
        { id: 'workflows', label: 'NLP Workflows',     href: 'workflows/index.html', icon: '⊳' },
      ],
    },
    {
      label: 'Applications',
      items: [
        { id: 'applications', label: 'Real-World Uses', href: 'applications/index.html', icon: '◈' },
      ],
    },
    {
      label: 'More',
      items: [
        { id: 'research',      label: 'Research & Insights',  href: 'research/index.html',      icon: '⬡' },
        { id: 'sustainability',label: 'Sustainability',        href: 'sustainability/index.html',icon: '⬡' },
        { id: 'reflection',    label: 'Reflection',            href: 'reflection/index.html',    icon: '⬡' },
        { id: 'references',    label: 'References',            href: 'references/index.html',    icon: '⬡' },
        { id: 'team',          label: 'Contribution Matrix',   href: 'team/index.html',          icon: '⬡' },
        { id: 'ai-usage',      label: 'AI Usage Declaration',  href: 'ai-usage/index.html',      icon: '⬡' },
      ],
    },
  ];

  /* ── SVG Icons (inline) ─────────────────────────────────── */
  const ICONS = {
    search: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>`,
    moon:   `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`,
    sun:    `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`,
    menu:   `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>`,
    chevron:`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M9 18l6-6-6-6"/></svg>`,
    github: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>`,
    up:     `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M18 15l-6-6-6 6"/></svg>`,
  };

  /* ── Logo HTML ───────────────────────────────────────────── */
  function logoMark(size = 28, fontSize = 13) {
    return `<img src="${ROOT}favicon-32x32.png" alt="" style="width:${size}px;height:${size}px;border-radius:6px;flex-shrink:0;object-fit:contain;" aria-hidden="true">`;
  }

  /* ═══════════════════════════════════════════════════════════
     NAVBAR
  ═══════════════════════════════════════════════════════════ */
  function navbarHTML() {
    const links = TOP_NAV.map(n => {
      const active = CFG.page === n.id;
      return `<a href="${ROOT}${n.href}" class="nav-link${active ? ' active' : ''}" data-nav="${n.id}" aria-current="${active ? 'page' : 'false'}">${n.label}</a>`;
    }).join('');

    const isLight = document.body.classList.contains('light');

    return `
    <div id="reading-progress" role="progressbar" aria-label="Reading progress" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>

    <nav id="navbar" role="navigation" aria-label="Main navigation">
      <!-- Hamburger -->
      <button id="menu-toggle" class="nav-btn" aria-label="Toggle sidebar menu" aria-expanded="false" aria-controls="sidebar">
        ${ICONS.menu}
      </button>

      <!-- Logo -->
      <a href="${ROOT}index.html" class="nav-logo" aria-label="NLP Hub home">
        ${logoMark(28, 13)}
        <span class="nav-logo-text" aria-hidden="false">NLP Hub</span>
      </a>

      <!-- Center links (desktop) -->
      <div class="nav-center" role="menubar" aria-label="Site sections">
        ${links}
      </div>

      <!-- Right actions -->
      <div class="nav-actions">
        <!-- Search chip -->
        <button id="search-btn" class="search-kbd-chip" aria-label="Open search" aria-keyshortcuts="Control+k">
          ${ICONS.search}
          <span>Search</span>
          <span class="kbd" aria-hidden="true">Ctrl K</span>
        </button>

        <!-- Search icon (mobile) -->
        <button id="search-btn-mobile" class="nav-btn" aria-label="Open search" style="display:flex;" aria-keyshortcuts="Control+k">
          ${ICONS.search}
        </button>

        <!-- Theme toggle -->
        <button id="theme-toggle" class="nav-btn" aria-label="Toggle dark/light mode">
          <span id="theme-icon-dark"  aria-hidden="true">${ICONS.moon}</span>
          <span id="theme-icon-light" aria-hidden="true" style="display:none;">${ICONS.sun}</span>
        </button>

        <!-- GitHub -->
        <a href="https://github.com/AtharvaisOp/NLP" target="_blank" rel="noopener noreferrer" class="nav-github-btn" aria-label="View on GitHub (opens in new tab)">
          ${ICONS.github}
          <span>GitHub</span>
        </a>
      </div>
    </nav>`;
  }

  /* ═══════════════════════════════════════════════════════════
     SIDEBAR
  ═══════════════════════════════════════════════════════════ */
  function sidebarHTML() {
    const sections = SIDEBAR.map(section => {
      const items = section.items.map(item => {
        // Active if exact match OR page starts with item ID prefix (for sub-pages)
        const active = CFG.page === item.id || String(CFG.page).startsWith(item.id + '-');
        return `<a href="${ROOT}${item.href}" class="sidebar-item${active ? ' active' : ''}" data-page="${item.id}">
          <span class="sidebar-item-icon" aria-hidden="true">${item.icon}</span>
          <span>${item.label}</span>
        </a>`;
      }).join('');

      return `<div class="sidebar-section">
        <span class="sidebar-label">${section.label}</span>
        ${items}
      </div>`;
    }).join('');

    return `
    <aside id="sidebar" role="complementary" aria-label="Documentation navigation">
      <a href="${ROOT}index.html" class="sidebar-brand" aria-label="NLP Hub home">
        ${logoMark(22, 11)}
        <span class="sidebar-brand-text">NLP Hub</span>
      </a>

      ${sections}

      <div class="sidebar-footer">
        <span>Academic Year 2025–26</span>
      </div>
    </aside>

    <div id="sidebar-overlay" aria-hidden="true" tabindex="-1"></div>`;
  }

  /* ═══════════════════════════════════════════════════════════
     FOOTER
  ═══════════════════════════════════════════════════════════ */
  function footerHTML() {
    return `
    <footer id="footer" role="contentinfo">
      <div class="footer-grid">
        <!-- Brand -->
        <div>
          <a href="${ROOT}index.html" class="nav-logo" style="margin-bottom:0.875rem;text-decoration:none;" aria-label="NLP Hub home">
            ${logoMark(28, 13)}
            <span class="nav-logo-text">NLP Hub</span>
          </a>
          <p style="font-size:.875rem;color:var(--text-sec);line-height:1.6;margin:0 0 1rem;max-width:200px;">
            An interactive NLP knowledge repository for students and beginners.
          </p>
          <a href="https://github.com/AtharvaisOp/NLP" target="_blank" rel="noopener noreferrer"
             style="display:inline-flex;align-items:center;gap:.375rem;font-size:.75rem;padding:.25rem .75rem;border:1px solid var(--border-hover);border-radius:9999px;color:var(--text-sec);text-decoration:none;transition:all .15s;"
             aria-label="GitHub (opens in new tab)">
            ${ICONS.github} GitHub
          </a>
        </div>

        <!-- Learn -->
        <div>
          <h3 class="footer-col-title">Learn</h3>
          <nav aria-label="Learning resources">
            <a href="${ROOT}repository/index.html"    class="footer-link">Repository</a>
            <a href="${ROOT}category/index.html"      class="footer-link">Categories</a>
            <a href="${ROOT}concept/index.html"       class="footer-link">Concepts</a>
            <a href="${ROOT}workflows/index.html"     class="footer-link">Workflows</a>
          </nav>
        </div>

        <!-- Explore -->
        <div>
          <h3 class="footer-col-title">Explore</h3>
          <nav aria-label="Exploration resources">
            <a href="${ROOT}compare/index.html"       class="footer-link">Compare</a>
            <a href="${ROOT}applications/index.html"  class="footer-link">Applications</a>
            <a href="${ROOT}research/index.html"      class="footer-link">Research</a>
            <a href="${ROOT}sustainability/index.html" class="footer-link">Sustainability</a>
          </nav>
        </div>

        <!-- About -->
        <div>
          <h3 class="footer-col-title">About</h3>
          <nav aria-label="About resources">
            <a href="${ROOT}reflection/index.html"    class="footer-link">Reflection</a>
            <a href="${ROOT}references/index.html"    class="footer-link">References</a>
            <a href="${ROOT}team/index.html"          class="footer-link">Team</a>
            <a href="${ROOT}design-system/index.html" class="footer-link">Design System</a>
          </nav>
        </div>
      </div>

      <div class="footer-bottom">
        <span>© 2026 NLP Hub · Academic Project</span>
        <div style="display:flex;gap:1rem;align-items:center;">
          <span>Open Source</span>
          <span aria-hidden="true">·</span>
          <span>MIT License</span>
        </div>
      </div>
    </footer>

    <!-- Scroll to top -->
    <button id="scroll-top" aria-label="Scroll back to top">
      ${ICONS.up}
    </button>`;
  }

  /* ═══════════════════════════════════════════════════════════
     SEARCH MODAL
  ═══════════════════════════════════════════════════════════ */
  function searchModalHTML() {
    return `
    <div id="search-overlay" role="dialog" aria-modal="true" aria-label="Search NLP Hub">
      <div id="search-modal">
        <div class="search-input-row">
          <span style="color:var(--text-muted);flex-shrink:0;" aria-hidden="true">${ICONS.search}</span>
          <input
            type="search"
            id="search-input"
            placeholder="Search NLP concepts, algorithms, techniques…"
            autocomplete="off"
            spellcheck="false"
            aria-label="Search"
            aria-autocomplete="list"
            aria-controls="search-results"
          />
          <kbd class="kbd" aria-label="Press Escape to close">Esc</kbd>
        </div>

        <div id="search-results" role="listbox" aria-label="Search results">
          <!-- Populated by search.js -->
        </div>

        <div class="search-footer" aria-hidden="true">
          <span class="search-footer-hint"><kbd class="kbd">↑↓</kbd> navigate</span>
          <span class="search-footer-hint"><kbd class="kbd">↵</kbd> select</span>
          <span class="search-footer-hint"><kbd class="kbd">Esc</kbd> close</span>
          <span style="margin-left:auto;display:flex;align-items:center;gap:.25rem;">
            <kbd class="kbd">Ctrl</kbd><span>+</span><kbd class="kbd">K</kbd>
          </span>
        </div>
      </div>
    </div>`;
  }

  /* ═══════════════════════════════════════════════════════════
     BREADCRUMB
  ═══════════════════════════════════════════════════════════ */
  function breadcrumbHTML() {
    if (!CFG.breadcrumbs || CFG.breadcrumbs.length === 0) return '';

    const crumbs = [{ label: 'Home', href: `${ROOT}index.html` }, ...CFG.breadcrumbs];

    const items = crumbs.map((crumb, i) => {
      const isLast = i === crumbs.length - 1;
      const sep = !isLast
        ? `<span class="breadcrumb-sep" aria-hidden="true">${ICONS.chevron}</span>`
        : '';
      const link = isLast
        ? `<span class="breadcrumb-current" aria-current="page">${crumb.label}</span>`
        : `<a href="${crumb.href || '#'}" class="breadcrumb-link">${crumb.label}</a>`;
      return link + sep;
    }).join('');

    return `<nav class="breadcrumb" aria-label="Breadcrumb">${items}</nav>`;
  }

  /* ═══════════════════════════════════════════════════════════
     INJECT — Called on DOMContentLoaded
  ═══════════════════════════════════════════════════════════ */
  function inject() {
    const get = id => document.getElementById(id);

    const navbar       = get('navbar-container');
    const sidebar      = get('sidebar-container');
    const footer       = get('footer-container');
    const searchModal  = get('search-container');
    const breadcrumb   = get('breadcrumb-container');

    if (navbar)      navbar.innerHTML      = navbarHTML();
    if (sidebar)     sidebar.innerHTML     = sidebarHTML();
    if (footer)      footer.innerHTML      = footerHTML();
    if (searchModal) searchModal.innerHTML = searchModalHTML();
    if (breadcrumb)  breadcrumb.innerHTML  = breadcrumbHTML();

    // Update theme icon state after injection
    if (document.body.classList.contains('light')) {
      const d = get('theme-icon-dark');
      const l = get('theme-icon-light');
      if (d) d.style.display = 'none';
      if (l) l.style.display = '';
    }
  }

  /* ── Auto-init ──────────────────────────────────────────── */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', inject);
  } else {
    inject();
  }

  /* ── Expose ─────────────────────────────────────────────── */
  global.NLP = global.NLP || {};
  global.NLP.Components = { inject, breadcrumbHTML, ROOT, CFG };

})(window);
