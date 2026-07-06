/**
 * महापल्स — Tailwind CSS Configuration
 * Load this BEFORE cdn.tailwindcss.com
 * Sets custom design tokens as Tailwind utilities
 */
window.tailwind = {
  config: {
    darkMode: 'class',
    theme: {
      extend: {
        colors: {
          'nlp-bg':        '#0A0F1E',
          'nlp-surface':   '#0F172A',
          'nlp-card':      '#1E293B',
          'nlp-elevated':  '#263148',
          'nlp-primary':   '#6366F1',
          'nlp-secondary': '#22D3EE',
          'nlp-tertiary':  '#A78BFA',
          'nlp-success':   '#10B981',
          'nlp-warning':   '#F59E0B',
          'nlp-danger':    '#EF4444',
          'nlp-text':      '#F1F5F9',
          'nlp-muted':     '#94A3B8',
          'nlp-border':    '#1E293B',
        },
        fontFamily: {
          'sans': ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
          'mono': ['JetBrains Mono', 'Fira Code', 'Courier New', 'monospace'],
        },
        maxWidth: {
          'content': '1180px',
        },
        backdropBlur: {
          'nav': '40px',
        },
        animation: {
          'fade-up': 'fadeUp 0.5s ease forwards',
          'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        },
        keyframes: {
          fadeUp: {
            'from': { opacity: '0', transform: 'translateY(16px)' },
            'to':   { opacity: '1', transform: 'translateY(0)' },
          },
          pulseGlow: {
            '0%, 100%': { boxShadow: '0 0 10px rgba(99,102,241,0.3)' },
            '50%':       { boxShadow: '0 0 25px rgba(99,102,241,0.6)' },
          },
        },
      },
    },
  },
};
