"""
CARN-X  --  presentation layer (visual only)
===========================================
Pure cosmetics for ``mstr_app.py``: a neon-aurora dark skin injected as global
CSS, a hero-banner helper, and a matching Altair chart theme.

**Nothing here computes, fetches, caches or decides anything.**  It only changes
how the existing widgets look.  Import ``inject_theme`` once (right after
``st.set_page_config``) and use ``hero(...)`` in place of ``st.header(...)``.
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------- #
# palette (single source of truth for the skin)
# --------------------------------------------------------------------------- #
CYAN = "#38E8FF"
VIOLET = "#A855F7"
MAGENTA = "#F472B6"
LIME = "#A3E635"
AMBER = "#FBBF24"
INK = "#070B1E"

CATEGORICAL = [CYAN, VIOLET, MAGENTA, LIME, AMBER, "#22D3EE", "#818CF8", "#FB7185"]


# --------------------------------------------------------------------------- #
# global stylesheet
# --------------------------------------------------------------------------- #
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
  --cx-cyan:{CYAN};   --cx-violet:{VIOLET};  --cx-magenta:{MAGENTA};
  --cx-lime:{LIME};   --cx-amber:{AMBER};
  --cx-glass: rgba(20, 28, 60, 0.55);
  --cx-border: rgba(150, 170, 255, 0.16);
  --cx-border-hi: rgba(120, 235, 255, 0.55);
  --cx-grad: linear-gradient(120deg, var(--cx-cyan) 0%, var(--cx-violet) 52%, var(--cx-magenta) 100%);
}}

/* ---------- base canvas + drifting aurora ---------- */
html, body, [class*="stApp"] {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
.stApp {{
  background:
    radial-gradient(1100px 720px at 12% -8%, rgba(168, 85, 247, 0.20), transparent 60%),
    radial-gradient(1000px 680px at 92% 4%, rgba(56, 232, 255, 0.16), transparent 58%),
    radial-gradient(1200px 900px at 60% 118%, rgba(244, 114, 182, 0.14), transparent 60%),
    {INK};
  background-attachment: fixed;
}}
.stApp::before {{
  content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(680px 520px at 20% 30%, rgba(56, 232, 255, 0.08), transparent 70%),
    radial-gradient(720px 560px at 80% 70%, rgba(168, 85, 247, 0.09), transparent 70%);
  animation: cx-aurora 24s ease-in-out infinite alternate;
}}
@keyframes cx-aurora {{
  0%   {{ transform: translate3d(-3%, -2%, 0) scale(1.02); opacity: .85; }}
  100% {{ transform: translate3d(3%, 3%, 0) scale(1.08);  opacity: 1; }}
}}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ position: relative; z-index: 1; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMain"] .block-container {{ padding-top: 2.4rem; max-width: 1400px; }}

/* ---------- typography ---------- */
h1, h2, h3 {{ font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -0.01em; }}
h1, h2 {{
  background: var(--cx-grad);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 700;
}}
h3 {{ color: #EAF2FF; font-weight: 600; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] {{ color: #C4CBEF; }}
a, a:visited {{ color: var(--cx-cyan); }}

/* ---------- hero banner ---------- */
.cx-hero {{
  position: relative; margin: 0 0 1.4rem 0; padding: 1.5rem 1.7rem 1.6rem;
  border-radius: 22px; overflow: hidden;
  background: linear-gradient(135deg, rgba(56,232,255,0.14), rgba(168,85,247,0.12) 55%, rgba(244,114,182,0.12));
  border: 1px solid var(--cx-border);
  box-shadow: 0 18px 48px -18px rgba(0,0,0,0.7), inset 0 1px 0 rgba(255,255,255,0.08);
  backdrop-filter: blur(10px);
}}
.cx-hero::after {{
  content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 3px;
  background: var(--cx-grad); background-size: 220% 100%;
  animation: cx-slide 6s linear infinite;
}}
@keyframes cx-slide {{ to {{ background-position: 220% 0; }} }}
.cx-hero-title {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-size: clamp(1.5rem, 2.4vw, 2.1rem); line-height: 1.15;
  background: var(--cx-grad); -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.cx-hero-sub {{ margin-top: .35rem; color: #B9C2EA; font-size: .95rem; font-weight: 400; }}
.cx-hero-eyebrow {{
  display: inline-block; margin-bottom: .55rem; padding: .18rem .7rem;
  font-size: .68rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;
  color: #061225; background: var(--cx-grad); border-radius: 999px;
}}

/* ---------- metric cards (glass) ---------- */
[data-testid="stMetric"] {{
  background: linear-gradient(150deg, rgba(56,232,255,0.10), rgba(168,85,247,0.07) 60%, rgba(10,16,40,0.35));
  border: 1px solid var(--cx-border);
  border-radius: 18px; padding: 1rem 1.1rem;
  box-shadow: 0 10px 30px -14px rgba(0,0,0,0.65), inset 0 1px 0 rgba(255,255,255,0.06);
  backdrop-filter: blur(8px);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}}
[data-testid="stMetric"]:hover {{
  transform: translateY(-3px);
  border-color: var(--cx-border-hi);
  box-shadow: 0 18px 40px -16px rgba(56,232,255,0.35), inset 0 1px 0 rgba(255,255,255,0.10);
}}
[data-testid="stMetricLabel"] {{
  text-transform: uppercase; letter-spacing: .09em; font-size: .70rem !important;
  color: #97A2D4 !important; font-weight: 600;
}}
[data-testid="stMetricValue"] {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 700;
  font-variant-numeric: tabular-nums; font-size: 1.9rem !important;
  background: linear-gradient(100deg, #EAF6FF, var(--cx-cyan) 55%, var(--cx-violet));
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}}
[data-testid="stMetricDelta"] {{ font-weight: 600; font-variant-numeric: tabular-nums; }}

/* ---------- tabs -> neon pills ---------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: .4rem; padding: .35rem; border-radius: 16px;
  background: rgba(12, 18, 44, 0.6); border: 1px solid var(--cx-border);
  backdrop-filter: blur(6px);
}}
.stTabs [data-baseweb="tab"] {{
  height: auto; padding: .5rem 1rem; border-radius: 12px;
  color: #AEB7E0; font-weight: 600; border: none; background: transparent;
  transition: color .15s ease, background .15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: #EAF2FF; background: rgba(120,235,255,0.08); }}
.stTabs [aria-selected="true"] {{
  color: #061024 !important; background: var(--cx-grad) !important;
  box-shadow: 0 8px 22px -8px rgba(168,85,247,0.6);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ background: transparent !important; }}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border: 1px solid rgba(255,255,255,0.14); border-radius: 13px;
  background: linear-gradient(135deg, var(--cx-violet), var(--cx-magenta));
  color: #fff; font-weight: 700; letter-spacing: .01em;
  box-shadow: 0 10px 26px -12px rgba(168,85,247,0.7);
  transition: transform .16s ease, box-shadow .16s ease, filter .16s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  transform: translateY(-2px); filter: brightness(1.08);
  box-shadow: 0 16px 34px -12px rgba(244,114,182,0.75);
}}
.stButton > button:active {{ transform: translateY(0); }}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(10,16,44,0.92), rgba(8,12,32,0.96));
  border-right: 1px solid var(--cx-border);
  backdrop-filter: blur(12px);
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] .stCaption {{ text-align: left; }}
[data-testid="stSidebar"] [data-testid="stMetric"] {{ padding: .8rem .9rem; border-radius: 14px; }}

/* radio (screen switcher) as a nav list */
[data-testid="stSidebar"] [role="radiogroup"] > label {{
  padding: .5rem .7rem; margin-bottom: .18rem; border-radius: 11px;
  border: 1px solid transparent; transition: background .14s ease, border-color .14s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
  background: rgba(120,235,255,0.08); border-color: var(--cx-border);
}}

/* ---------- inputs / sliders / toggles ---------- */
[data-baseweb="slider"] [data-testid="stSliderTrack"] > div {{ background: var(--cx-grad) !important; }}
[data-baseweb="slider"] [role="slider"] {{
  background: #fff !important;
  box-shadow: 0 0 0 4px rgba(120,235,255,0.30), 0 2px 8px rgba(0,0,0,0.4) !important;
}}
[data-testid="stWidgetLabel"] p {{ color: #B7C0E8 !important; font-weight: 600; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
  border-radius: 12px !important; border-color: var(--cx-border) !important;
  background: rgba(12,18,44,0.7) !important;
}}
[data-testid="stToggle"] [data-baseweb="toggle"][aria-checked="true"] > div {{
  background: var(--cx-grad) !important;
}}

/* ---------- progress bar (shimmer) ---------- */
[data-testid="stProgressBar"] > div > div, .stProgress > div > div > div {{
  background: linear-gradient(90deg, var(--cx-cyan), var(--cx-violet), var(--cx-magenta), var(--cx-cyan));
  background-size: 300% 100%; animation: cx-shimmer 2.6s linear infinite;
}}
@keyframes cx-shimmer {{ to {{ background-position: 300% 0; }} }}

/* ---------- alerts ---------- */
[data-testid="stAlert"], [data-testid="stNotification"] {{
  border-radius: 15px; border: 1px solid var(--cx-border);
  background: var(--cx-glass); backdrop-filter: blur(8px);
  border-left-width: 4px;
}}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{ color: #E7ECFF; }}

/* ---------- expander / containers / dataframe ---------- */
[data-testid="stExpander"] {{
  border-radius: 16px; border: 1px solid var(--cx-border);
  background: rgba(12,18,44,0.5); backdrop-filter: blur(6px); overflow: hidden;
}}
[data-testid="stExpander"] summary:hover {{ background: rgba(120,235,255,0.06); }}
[data-testid="stForm"] {{
  border-radius: 18px; border: 1px solid var(--cx-border);
  background: rgba(12,18,44,0.45); backdrop-filter: blur(6px);
}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border-radius: 14px; overflow: hidden; border: 1px solid var(--cx-border);
}}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"],
.stPlotlyChart, [data-testid="stVegaLiteChart"] > div {{
  border-radius: 16px;
}}
hr {{ border-color: var(--cx-border); }}

/* ---------- scrollbars ---------- */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
  background: linear-gradient(180deg, var(--cx-violet), var(--cx-cyan));
  border-radius: 999px; border: 2px solid {INK};
}}

/* nothing may spill sideways -- wide content scrolls inside its own box */
[data-testid="stAppViewContainer"] {{ overflow-x: hidden; }}
[data-testid="stMain"] {{ overflow-x: clip; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ overflow-x: auto; }}
pre, code {{ white-space: pre-wrap; word-break: break-word; }}
[data-testid="stMetricValue"] {{ overflow-wrap: anywhere; }}
img, iframe, canvas, svg {{ max-width: 100%; }}

/* progress bar: the RTL page must not shove the track off the left edge */
[data-testid="stProgress"], .stProgress, [data-testid="stProgressBar"],
[data-testid="stProgressBar"] > div {{ direction: ltr !important; }}
[data-testid="stProgressBar"] > div {{
  margin: 0 !important; left: auto !important; right: auto !important;
  transform: none !important; width: 100% !important;
}}

/* inline KaTeX formulas scroll inside their own box, never past the screen edge */
[data-testid="stMain"] .katex-display {{ overflow-x: auto; overflow-y: hidden; }}
[data-testid="stMain"] .katex {{ max-width: 100%; overflow-x: auto; overflow-y: hidden; }}

/* tab strips scroll horizontally instead of pushing the layout wide */
.stTabs [data-baseweb="tab-list"] {{ overflow-x: auto; overflow-y: hidden; flex-wrap: nowrap; }}
.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"] {{ flex: 0 0 auto; white-space: nowrap; }}

/* ==================== tablet  (<= 900px) ==================== */
@media (max-width: 900px) {{
  [data-testid="stMainBlockContainer"], .block-container {{
    padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important;
  }}
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; row-gap: .6rem; }}
  [data-testid="stColumn"] {{ flex: 1 1 calc(50% - .6rem) !important; min-width: calc(50% - .6rem) !important; }}
  .stTabs [data-baseweb="tab-list"] {{ overflow-x: auto !important; flex-wrap: nowrap !important; }}
  .stTabs [data-baseweb="tab"] {{ flex: 0 0 auto; }}
  [data-testid="stSegmentedControl"] [role="group"],
  [data-testid="stSegmentedControl"] > div {{ flex-wrap: wrap !important; }}
}}

/* ==================== phone  (<= 560px -- iPhone) ==================== */
@media (max-width: 560px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding: .75rem .6rem 3rem !important; }}
  [data-testid="stHeader"] {{ height: 2.6rem; }}
  [data-testid="stColumn"] {{ flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }}
  .cx-hero {{ padding: .9rem 1rem 1rem; border-radius: 15px; margin-bottom: 1rem; }}
  .cx-hero-title {{ font-size: 1.15rem !important; line-height: 1.2; }}
  .cx-hero-sub {{ font-size: .82rem; }}
  .cx-hero-eyebrow {{ font-size: .6rem; }}
  h1, h2 {{ font-size: 1.3rem !important; line-height: 1.25; }}
  h3 {{ font-size: 1.05rem !important; }}
  [data-testid="stMetric"] {{ padding: .65rem .75rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
  [data-testid="stMetricLabel"] {{ font-size: .62rem !important; }}
  [data-testid="stMetricLabel"] p {{ white-space: normal; }}
  /* open sidebar floats over the content as an overlay -- never let its
     layout box push or squeeze stMain (that caused the 40px content strip) */
  [data-testid="stSidebar"][aria-expanded="true"] {{
    position: fixed !important; top: 0; bottom: 0; left: 0; z-index: 9999;
    min-width: min(86vw, 340px) !important; width: min(86vw, 340px) !important;
    box-shadow: 0 0 40px rgba(0,0,0,.6);
  }}
  [data-testid="stSidebar"][aria-expanded="false"] {{ min-width: 0 !important; width: 0 !important; }}
  iframe {{ width: 100% !important; }}
  .stTabs [data-baseweb="tab"] {{ padding: .45rem .7rem; font-size: .8rem; }}
  .stProgress p {{ white-space: normal; }}
  [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] p {{ font-size: .86rem; }}
  /* KaTeX / MathML formulas scroll inside their own box instead of spilling */
  [data-testid="stMain"] .katex-display,
  [data-testid="stMain"] mjx-container,
  [data-testid="stMain"] math {{ overflow-x: auto; overflow-y: hidden; max-width: 100%; }}
}}
</style>
"""


def inject_theme() -> None:
    """Inject the global skin.  Call once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)
    _register_altair_theme()


def hero(title: str, subtitle: str = "", eyebrow: str = "CARN-X") -> None:
    """Drop-in replacement for ``st.header`` with a gradient hero banner."""
    parts = ['<div class="cx-hero">']
    if eyebrow:
        parts.append(f'<span class="cx-hero-eyebrow">{eyebrow}</span>')
    parts.append(f'<div class="cx-hero-title">{title}</div>')
    if subtitle:
        parts.append(f'<div class="cx-hero-sub">{subtitle}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Altair theme  (custom charts on the Monte-Carlo / BTC screens)
# --------------------------------------------------------------------------- #
_ALTAIR_DONE = False


def _register_altair_theme() -> None:
    global _ALTAIR_DONE
    if _ALTAIR_DONE:
        return
    try:
        import altair as alt
    except Exception:  # pragma: no cover
        return

    def _carnx():
        return {
            "config": {
                "background": "transparent",
                "view": {"stroke": "transparent", "continuousHeight": 300},
                "font": "Inter, sans-serif",
                "title": {"color": "#E9ECFF", "fontSize": 15, "font": "Space Grotesk, sans-serif"},
                "axis": {
                    "domainColor": "rgba(150,170,255,0.25)",
                    "gridColor": "rgba(150,170,255,0.10)",
                    "tickColor": "rgba(150,170,255,0.25)",
                    "labelColor": "#AEB7E0",
                    "titleColor": "#C4CBEF",
                    "labelFont": "Inter, sans-serif",
                    "titleFont": "Inter, sans-serif",
                },
                "legend": {
                    "labelColor": "#C4CBEF",
                    "titleColor": "#C4CBEF",
                    "labelFont": "Inter, sans-serif",
                    "titleFont": "Inter, sans-serif",
                },
                "range": {
                    "category": CATEGORICAL,
                    "heatmap": ["#0B1233", "#1E3A8A", "#38E8FF"],
                    "ramp": ["#0B1233", "#3B5BDB", "#38E8FF"],
                },
            }
        }

    try:  # Altair >= 5.5: theme.register is a decorator (name, *, enable)
        alt.theme.register("carnx", enable=True)(_carnx)  # type: ignore[attr-defined]
        _ALTAIR_DONE = True
    except Exception:
        try:  # Altair < 5.5
            alt.themes.register("carnx", _carnx)  # type: ignore[attr-defined]
            alt.themes.enable("carnx")  # type: ignore[attr-defined]
            _ALTAIR_DONE = True
        except Exception:
            pass
