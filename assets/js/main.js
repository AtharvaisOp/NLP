/**
 * NLP Hub — Core Application Logic
 * ─────────────────────────────────────────────────────────────
 * Handles: Theme, Mobile Menu, Scroll Progress, Scroll-to-Top,
 *          Keyboard Shortcuts, TOC Active Tracking, Copy Code,
 *          Active Navigation, External Link Handling.
 *
 * NOTE: Applies theme class BEFORE DOMContentLoaded to prevent
 * a flash of wrong theme. All other init happens after components
 * are injected by components.js.
 */
(function (global) {
  'use strict';

  /* ══════════════════════════════════════════════════════════
     THEME
     Manages dark / light mode. Default: dark.
  ══════════════════════════════════════════════════════════ */
  const Theme = {
    STORAGE_KEY: 'nlp-theme',

    /** Read saved preference, default to 'dark'. */
    getSaved() {
      try { return localStorage.getItem(this.STORAGE_KEY) || 'dark'; }
      catch { return 'dark'; }
    },

    /** Apply theme to document body (called immediately on load). */
    apply(mode) {
      document.body.classList.toggle('light', mode === 'light');
    },

    /** Update the toggle button icons after components are injected. */
    syncIcons(mode) {
      const dark  = document.getElementById('theme-icon-dark');
      const light = document.getElementById('theme-icon-light');
      if (!dark || !light) return;
      if (mode === 'light') {
        dark.style.display  = 'none';
        light.style.display = '';
      } else {
        dark.style.display  = '';
        light.style.display = 'none';
      }
    },

    /** Persist and apply a new mode. */
    set(mode) {
      this.apply(mode);
      this.syncIcons(mode);
      try { localStorage.setItem(this.STORAGE_KEY, mode); } catch {}
    },

    toggle() {
      const current = this.getSaved();
      this.set(current === 'dark' ? 'light' : 'dark');
    },

    init() {
      const saved = this.getSaved();
      this.apply(saved);    // immediate — prevents flash
    },
  };

  /* ══════════════════════════════════════════════════════════
     MOBILE MENU
     Toggles sidebar visibility on small screens.
  ══════════════════════════════════════════════════════════ */
  const MobileMenu = {
    open: false,

    toggle() { this.open ? this.close() : this.show(); },

    show() {
      this.open = true;
      document.getElementById('sidebar')?.classList.add('open');
      document.getElementById('sidebar-overlay')?.classList.add('active');
      document.getElementById('menu-toggle')?.setAttribute('aria-expanded', 'true');
      document.body.style.overflow = 'hidden';
    },

    close() {
      this.open = false;
      document.getElementById('sidebar')?.classList.remove('open');
      document.getElementById('sidebar-overlay')?.classList.remove('active');
      document.getElementById('menu-toggle')?.setAttribute('aria-expanded', 'false');
      document.body.style.overflow = '';
    },
  };

  /* ══════════════════════════════════════════════════════════
     SCROLL PROGRESS BAR
     Fills the 2px bar at the top proportional to scroll depth.
  ══════════════════════════════════════════════════════════ */
  const ScrollProgress = {
    bar: null,

    init() {
      this.bar = document.getElementById('reading-progress');
      if (!this.bar) return;
      window.addEventListener('scroll', () => this.update(), { passive: true });
      this.update();
    },

    update() {
      if (!this.bar) return;
      const doc      = document.documentElement;
      const scrolled = doc.scrollTop / (doc.scrollHeight - doc.clientHeight) || 0;
      const pct      = Math.min(100, Math.round(scrolled * 1000) / 10);
      this.bar.style.width = pct + '%';
      this.bar.setAttribute('aria-valuenow', Math.round(pct));
    },
  };

  /* ══════════════════════════════════════════════════════════
     SCROLL-TO-TOP BUTTON
  ══════════════════════════════════════════════════════════ */
  const ScrollTop = {
    btn: null,

    init() {
      this.btn = document.getElementById('scroll-top');
      if (!this.btn) return;

      window.addEventListener('scroll', () => {
        this.btn.classList.toggle('visible', window.scrollY > 400);
      }, { passive: true });

      this.btn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    },
  };

  /* ══════════════════════════════════════════════════════════
     TABLE OF CONTENTS — Active Section Tracking
     Uses IntersectionObserver to highlight the current heading.
  ══════════════════════════════════════════════════════════ */
  const TOC = {
    init() {
      const tocLinks = [...document.querySelectorAll('.toc-link[href^="#"]')];
      if (!tocLinks.length) return;

      const pairs = tocLinks.map(link => ({
        link,
        target: document.getElementById(link.getAttribute('href').slice(1)),
      })).filter(p => p.target);

      if (!pairs.length) return;

      const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            tocLinks.forEach(l => l.classList.remove('active'));
            const match = pairs.find(p => p.target === entry.target);
            if (match) match.link.classList.add('active');
          }
        });
      }, { rootMargin: '-10% 0px -78% 0px', threshold: 0 });

      pairs.forEach(({ target }) => observer.observe(target));
    },
  };

  /* ══════════════════════════════════════════════════════════
     COPY CODE BUTTONS
     Attaches click handlers to all .code-copy buttons.
  ══════════════════════════════════════════════════════════ */
  const CopyCode = {
    init() {
      document.addEventListener('click', e => {
        const btn = e.target.closest('.code-copy');
        if (!btn) return;

        const block = btn.closest('.code-block');
        const code  = block?.querySelector('code');
        if (!code) return;

        navigator.clipboard?.writeText(code.textContent).then(() => {
          const orig = btn.textContent;
          btn.textContent = 'Copied!';
          btn.style.color = 'var(--success)';
          setTimeout(() => {
            btn.textContent = orig;
            btn.style.color = '';
          }, 1800);
        }).catch(() => {});
      });
    },
  };

  /* ══════════════════════════════════════════════════════════
     KEYBOARD SHORTCUTS
  ══════════════════════════════════════════════════════════ */
  const Keyboard = {
    init() {
      document.addEventListener('keydown', e => {
        /* Ctrl/Cmd + K → open search */
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
          e.preventDefault();
          Search.open();
          return;
        }

        /* Escape → close modals / sidebar */
        if (e.key === 'Escape') {
          Search.close();
          MobileMenu.close();
        }
      });
    },
  };

  /* ══════════════════════════════════════════════════════════
     SEARCH MODAL — open / close (full logic in search.js)
  ══════════════════════════════════════════════════════════ */
  const Search = {
    open() {
      const overlay = document.getElementById('search-overlay');
      if (overlay) {
        overlay.classList.add('active');
        document.getElementById('search-input')?.focus();
        document.body.style.overflow = 'hidden';
      }
    },
    close() {
      const overlay = document.getElementById('search-overlay');
      if (overlay) {
        overlay.classList.remove('active');
        document.body.style.overflow = '';
      }
    },
    toggle() {
      const overlay = document.getElementById('search-overlay');
      overlay?.classList.contains('active') ? this.close() : this.open();
    },
  };

  /* ══════════════════════════════════════════════════════════
     ACTIVE NAV — ensure current page link is highlighted
  ══════════════════════════════════════════════════════════ */
  const ActiveNav = {
    init() {
      /* Sidebar items: match data-page to config */
      const cfg = global.NLP_CONFIG || {};
      document.querySelectorAll('.sidebar-item').forEach(el => {
        const pg = el.dataset.page;
        if (pg && (cfg.page === pg || String(cfg.page).startsWith(pg + '-'))) {
          el.classList.add('active');
        }
      });

      /* Top nav: match data-nav to config */
      document.querySelectorAll('.nav-link').forEach(el => {
        const nav = el.dataset.nav;
        if (nav && cfg.page === nav) el.classList.add('active');
      });
    },
  };

  /* ══════════════════════════════════════════════════════════
     FOOTER LINK HOVER EFFECT
  ══════════════════════════════════════════════════════════ */
  function initFooterHover() {
    document.querySelectorAll('.footer-link, .footer-col-title + nav a').forEach(a => {
      a.addEventListener('mouseenter', () => { a.style.color = 'var(--primary)'; });
      a.addEventListener('mouseleave', () => { a.style.color = ''; });
    });
  }

  /* ══════════════════════════════════════════════════════════
     MAIN INIT — runs after components.js has injected HTML
  ══════════════════════════════════════════════════════════ */
  function init() {
    /* Theme icon sync (component just injected) */
    Theme.syncIcons(Theme.getSaved());

    /* Theme toggle button */
    document.getElementById('theme-toggle')?.addEventListener('click', () => Theme.toggle());

    /* Mobile menu */
    document.getElementById('menu-toggle')?.addEventListener('click', () => MobileMenu.toggle());
    document.getElementById('sidebar-overlay')?.addEventListener('click', () => MobileMenu.close());
    window.addEventListener('resize', () => {
      if (window.innerWidth >= 1024) MobileMenu.close();
    });

    /* Search open buttons */
    document.getElementById('search-btn')?.addEventListener('click', () => Search.open());
    document.getElementById('search-btn-mobile')?.addEventListener('click', () => Search.open());

    /* Close search overlay on backdrop click */
    document.getElementById('search-overlay')?.addEventListener('click', e => {
      if (e.target.id === 'search-overlay') Search.close();
    });

    /* Scroll features */
    ScrollProgress.init();
    ScrollTop.init();

    /* TOC (only meaningful on concept/doc pages) */
    TOC.init();

    /* Copy code buttons */
    CopyCode.init();

    /* Keyboard shortcuts */
    Keyboard.init();

    /* Active navigation state */
    ActiveNav.init();

    /* Footer hover */
    initFooterHover();
  }

  /* ── Apply theme immediately (before DOMContentLoaded) ── */
  Theme.init();

  /* ── Wait for DOMContentLoaded (components injected first) */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // DOM already ready — run after components.js DOMContentLoaded
    // handlers complete (same event loop tick is guaranteed ordered)
    setTimeout(init, 0);
  }

  /* ── Expose globals for page-level scripts ─────────────── */
  global.NLP = global.NLP || {};
  Object.assign(global.NLP, { Theme, MobileMenu, Search, ScrollProgress });

})(window);
