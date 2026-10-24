# All app CSS in one place. Injected once via a single st.markdown block.
# Follows .kiro/steering/frontend-design.md (Emil Kowalski style).
import streamlit as st

_CSS = """
<style>
:root {
  --ease: cubic-bezier(0.23, 1, 0.32, 1);
  --accent: #e5484d;
  --ink: #1a1a1a;
  --muted: #6b7280;
  --surface: #ffffff;
  --bg: #fafafa;
}

html, body, [class*="css"] {
  font-variant-numeric: tabular-nums;
}

.stApp { background: var(--bg); }

/* numbers never jitter */
.metric-number, .headline, [data-testid="stMetricValue"] {
  font-variant-numeric: tabular-nums;
}

/* soft layered shadow cards, no hard borders */
.card {
  background: var(--surface);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(16,24,40,0.06),
              0 4px 12px rgba(16,24,40,0.06);
  opacity: 0;
  transform: scale(0.95);
  animation: enter 260ms var(--ease) forwards;
}

/* staggered entrances, capped at 8 */
.card:nth-child(1) { animation-delay: 0ms; }
.card:nth-child(2) { animation-delay: 40ms; }
.card:nth-child(3) { animation-delay: 80ms; }
.card:nth-child(4) { animation-delay: 120ms; }
.card:nth-child(5) { animation-delay: 160ms; }
.card:nth-child(6) { animation-delay: 200ms; }
.card:nth-child(7) { animation-delay: 240ms; }
.card:nth-child(8) { animation-delay: 280ms; }

@keyframes enter {
  from { opacity: 0; transform: scale(0.95); }
  to   { opacity: 1; transform: scale(1); }
}

/* headline counts up visually with a quick fade and rise */
.headline {
  font-size: 56px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: -0.02em;
  line-height: 1.05;
  opacity: 0;
  transform: scale(0.95);
  animation: enter 560ms var(--ease) forwards;
}
.headline .accent { color: var(--accent); }
.headline-sub { color: var(--muted); font-size: 15px; margin-top: 4px; }

/* press feedback on buttons and clickable cards */
.stButton > button, .stDownloadButton > button {
  transition: transform 150ms var(--ease), box-shadow 150ms var(--ease);
  border-radius: 10px;
}
.stButton > button:active, .stDownloadButton > button:active {
  transform: scale(0.97);
}

/* inner highlight on dark surfaces */
.dark-surface {
  background: #16181d;
  color: #f5f5f5;
  border-radius: 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08),
              0 6px 18px rgba(0,0,0,0.25);
  padding: 18px 20px;
}

.pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  background: rgba(229,72,77,0.12);
  color: var(--accent);
}

/* reduced motion kills all animation */
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
  .card, .headline { opacity: 1 !important; transform: none !important; }
}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)