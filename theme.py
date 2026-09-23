"""
CARN-X  --  presentation layer  (visual only; no logic, no data, no decisions)
=============================================================================
A calm, normal white dashboard: a flat white ground with a family of blue /
turquoise accents (cards and lines carry a faint blue wash rather than one
single flat tint, so the surface reads as "white, varied" rather than
"white, blank"), plain cards with a hairline border, ordinary sans
typography. No animated backdrop, no glow, no chromatic effects, no
per-page "design languages" -- one consistent look everywhere, on purpose
(the app was repeatedly too loud; this is the deliberate correction).

Public surface (unchanged):
    inject_theme()               once, right after st.set_page_config
    header_nav(sections, default) top identity + centred menu; returns the pick
    hero(title, subtitle, eyebrow)  a plain section heading
    refresh_button()             reload control (drawn by header_nav)
    brand()                      back-compat -> no-op

Nothing here computes, fetches, caches or decides.
"""

from __future__ import annotations

import base64
import os

import streamlit as st

# --------------------------------------------------------------------------- #
# palette  --  flat white ground, a family of blue/turquoise accents, plain
# surfaces tinted with a faint blue wash instead of grey (that wash is what
# reads as "varied" against a flat white, per the user's request).
# (var NAMES kept as GOLD*/INK* for back-compat with the rest of this file --
#  the values are just a normal, calm colour set.)
# --------------------------------------------------------------------------- #
INK = "#FFFFFF"           # the ground -- white
INK_EDGE = "#EAF2FF"      # a pale blue surface (sidebar) -- the "variety" against flat white
PANEL = "rgba(31,79,189,0.045)"   # a plain card fill -- a faint blue wash over white
PANEL_HI = "rgba(31,79,189,0.08)"  # hover state
SUNK = "rgba(31,79,189,0.06)"
LINE = "rgba(31,79,189,0.16)"      # the hairline every card/input uses
LINE_SOFT = "rgba(31,79,189,0.08)"

GOLD = "#2F6FED"          # the one accent colour (a plain, strong blue)
GOLD_BRIGHT = "#4C8DFF"
GOLD_DEEP = "#1D4FBD"
GOLD_TEXT = "#1D4FBD"

TEXT = "#0B1220"
TEXT_DIM = "#4C5B75"
TEXT_FAINT = "#7C8AA3"

UP = "#1E8E3E"
DOWN = "#D33B2C"

CYAN = "#00AEEF"          # turquoise, for the requested blue/cyan variety
VIOLET = "#7B68F0"
MAGENTA = "#D0499D"
LIME = "#5FAE3A"
AMBER = "#D68A1D"
CATEGORICAL = [GOLD, UP, AMBER, DOWN, CYAN, VIOLET, "#3D7FE0", MAGENTA]

# one plain sans family throughout -- no display/serif split, no dramatic
# glitch type.  numerals use the mono face (normal in a finance app).
_DISPLAY = "'Inter', 'Assistant', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
_SANS = _DISPLAY
_MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700;800"
    "&family=Assistant:wght@400;500;600;700"
    "&family=IBM+Plex+Mono:wght@400;500;600"
    "&display=swap"
)

_SECTIONS: list[str] = [
    "סקירה", "טרמינל מסחר", "מחשבון הסתברויות", "אבחון סטטיסטי",
    "תחזית הסתברותית", "Monte Carlo", "מבנים מתמטיים",
    "מחזור ביטקוין → MSTR", "ראיות Walk-Forward", "סיכון ומינוף", "הגדרות",
]


def _css() -> str:
    return f"""
<style>
@import url('{_FONTS}');

@property --cx-sec-i {{ syntax: "<number>"; inherits: true; initial-value: 0; }}
@property --spin {{ syntax: "<angle>"; inherits: true; initial-value: 0deg; }}

:root {{
  --ink:{INK}; --ink-edge:{INK_EDGE}; --panel:{PANEL}; --panel-hi:{PANEL_HI};
  --sunk:{SUNK}; --line:{LINE}; --line-soft:{LINE_SOFT};
  --gold:{GOLD}; --gold-bright:{GOLD_BRIGHT}; --gold-deep:{GOLD_DEEP}; --gold-text:{GOLD_TEXT};
  --gold-wash:rgba(47,111,237,0.10); --gold-line:rgba(47,111,237,0.28);
  --glow:0 0 0 1px rgba(47,111,237,.28);
  --text:{TEXT}; --text-dim:{TEXT_DIM}; --text-faint:{TEXT_FAINT};
  --up:{UP}; --down:{DOWN};
  --display:{_DISPLAY}; --sans:{_SANS}; --mono:{_MONO};
  --cx-sec-i: 0;
}}

/* ---------- ground: flat, no gradient, no animation ---------- */
html, body, [class*="stApp"] {{ font-family: var(--sans); }}
html, body {{ background: var(--ink); }}
.stApp {{ background: transparent; color: var(--text); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], [data-testid="stSidebarContent"] {{ background: transparent; }}
[data-testid="stAppViewContainer"] > .main, .block-container {{ position: relative; z-index: 1; }}
[data-testid="stToolbar"] {{ right: .5rem; }}
[data-testid="stAppDeployButton"], [data-testid="stDeployButton"] {{ display: none !important; }}
[data-testid="stMain"] .block-container {{ padding-top: 1.1rem; max-width: 1180px; }}
[data-testid="stSidebarCollapsedControl"] {{ opacity: .5; }}
[data-testid="stSidebar"] {{ background: var(--ink-edge); border-right: 1px solid var(--line); }}

/* section-change cue: a quick, barely-there fade -- not a "sweep", not a flash */
.cx-wipe, .cx-flash {{
  position: fixed; inset: 0; z-index: 5; pointer-events: none; opacity: 0;
  background: rgba(31,79,189,0.05);
}}
.cx-wipe.run, .cx-flash.run {{ animation: cx-fade .3s ease-out; }}
@keyframes cx-fade {{ 0% {{ opacity: 0; }} 30% {{ opacity: 1; }} 100% {{ opacity: 0; }} }}

/* ---------- masthead ---------- */
.cx-mast {{
  display: flex; align-items: center; gap: .7rem;
  padding: .1rem 0 .5rem; flex-wrap: wrap;
}}
.cx-mast svg {{ height: 40px; width: auto; display: block; }}

/* the data-refresh control -- a plain button, not a glowing pill */
.st-key-cx_refresh button, .cx-reload button {{
  font-family: var(--sans) !important; font-size: .78rem !important; font-weight: 600 !important;
  letter-spacing: 0 !important; text-transform: none !important;
  padding: .38rem .95rem !important; border-radius: 8px !important;
  color: #fff !important; -webkit-text-fill-color: #fff !important;
  background: #2E9F5C !important;
  border: 1px solid rgba(31,79,189,0.14) !important;
  transition: background .12s ease !important;
}}
.st-key-cx_refresh button:hover, .cx-reload button:hover {{ background: #35B267 !important; }}
.st-key-cx_refresh button:active, .cx-reload button:active {{ background: #278750 !important; }}
.st-key-cx_refresh button:focus-visible, .cx-reload button:focus-visible {{
  outline: 2px solid rgba(63,185,80,.6) !important; outline-offset: 2px;
}}
.st-key-cx_refresh button p, .cx-reload button p {{ color: #fff !important; }}

/* ---------- nav ---------- *
 * A plain, always-visible horizontal tab list is the REAL navigation -- it
 * needs no JavaScript at all, so it works even where scroll_boot's page-patch
 * can't run (some managed hosts block writing into Streamlit's own static
 * files). Where the patch DOES run, scroll_boot additionally builds a quiet
 * ring of pills around the mark (.cx-orbit) and -- only once that actually
 * exists in the page -- CSS hides this plain list in its favour. Either way
 * there is always a working way to switch pages. */
.st-key-cx_nav [role="radiogroup"] {{
  display: flex; flex-wrap: wrap; gap: .4rem; justify-content: center;
}}
.st-key-cx_nav [role="radiogroup"] > label {{
  border: 1px solid var(--line); border-radius: 999px; background: var(--panel);
  padding: .4rem .9rem; cursor: pointer; transition: background .12s ease, border-color .12s ease;
}}
.st-key-cx_nav [role="radiogroup"] > label:hover {{ background: var(--panel-hi); }}
.st-key-cx_nav [role="radiogroup"] > label:has(input:checked) {{
  background: var(--gold-deep); border-color: var(--gold);
}}
.st-key-cx_nav [role="radiogroup"] > label:has(input:checked) p {{ color: #fff !important; }}
body:has(.cx-orbit) .st-key-cx_nav {{
  position: absolute !important; left: -9999px !important; top: 0 !important;
  width: 1px !important; height: 0 !important; overflow: hidden !important;
  opacity: 0 !important; pointer-events: none !important; margin: 0 !important; border: 0 !important;
}}

.cx-orbit {{
  --r: 168px; --size: 372px; --orbit-rot: 0deg; --spin: 0deg;
  position: relative; width: var(--size); height: var(--size);
  margin: .6rem auto 2.8rem; z-index: 20;
  touch-action: none; -webkit-user-select: none; user-select: none;
}}
.cx-orbit::before {{
  content: ""; position: absolute; inset: 26px; border-radius: 50%;
  border: 1px solid var(--line);
  -webkit-mask: radial-gradient(circle, transparent calc(50% - 26px), #000 calc(50% - 25px));
          mask: radial-gradient(circle, transparent calc(50% - 26px), #000 calc(50% - 25px));
}}
.cx-orbit-ring {{
  position: absolute; inset: 0; border-radius: 50%;
  transform: rotate(calc(var(--orbit-rot) + var(--spin)));
  transition: transform .5s cubic-bezier(.2,.85,.25,1);
}}
.cx-orbit.dragging .cx-orbit-ring {{ transition: none; }}

.cx-orbit-item {{
  position: absolute; left: 50%; top: 50%; width: 0; height: 0;
  transform: rotate(var(--a)) translateY(calc(-1 * var(--r)));
}}
.cx-orbit-item > button {{
  position: absolute; left: 50%; top: 50%;
  transform: translate(-50%,-50%) rotate(calc(-1 * var(--a) - var(--orbit-rot) - var(--spin)));
  white-space: nowrap; cursor: pointer; line-height: 1;
  font-family: var(--sans); font-weight: 600; font-size: .74rem;
  color: var(--text-dim);
  background: var(--panel);
  border: 1px solid var(--line); border-radius: 999px; padding: .4rem .78rem;
  transition: color .15s, background .15s, border-color .15s;
}}
.cx-orbit-item > button:hover {{
  color: var(--text); border-color: rgba(31,79,189,.22); background: var(--panel-hi);
}}
.cx-orbit-item.on > button {{
  color: #fff; -webkit-text-fill-color: #fff; font-weight: 700;
  background: var(--gold-deep); border-color: var(--gold);
  z-index: 4;
}}

.cx-orbit-hub {{
  position: absolute; left: 50%; top: 50%; width: 84px; height: 84px;
  transform: translate(-50%,-50%); border-radius: 50%; cursor: grab; z-index: 5;
}}
.cx-orbit-hub:active {{ cursor: grabbing; }}
.cx-orbit-hub svg, .cx-orbit-hub img {{
  width: 100%; height: 100%; display: block; border-radius: 50%;
  border: 1px solid var(--line);
}}

/* ---------- typography: one plain sans face throughout ---------- */
h1, h2, h3, h4, h5 {{
  font-family: var(--sans); letter-spacing: -0.01em; text-wrap: balance; font-weight: 700;
  color: var(--text);
}}
h1 {{ font-size: clamp(1.8rem, 2.6vw, 2.4rem); line-height: 1.1; font-weight: 800; }}
h2 {{ font-size: clamp(1.4rem, 2.1vw, 1.9rem); line-height: 1.15; font-weight: 700; }}
h3 {{ font-size: 1.18rem; }}  h4 {{ font-size: 1.02rem; }}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {{
  color: var(--text-dim); line-height: 1.65; text-wrap: pretty; font-family: var(--sans);
}}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong {{
  color: var(--text); font-weight: 600;
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  color: var(--text-faint); font-size: .82rem;
}}
a, a:visited {{ color: var(--gold-text); text-underline-offset: 2px; }}
code, pre, kbd, [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
.cx-num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
:not(pre) > code {{
  background: var(--gold-wash); color: var(--gold-text);
  border: 1px solid var(--gold-line); border-radius: 6px; padding: .05em .38em; font-size: .84em;
}}
pre, [data-testid="stCode"] {{
  background: var(--sunk) !important;
  border: 1px solid var(--line); border-radius: 10px;
}}
.katex {{ color: var(--text); }}

/* ---------- section heading: a plain title, nothing else ---------- */
.cx-hero {{ margin: 1rem 0 1.8rem; padding: .2rem 0; }}
.cx-hero-title {{
  display: block; font-family: var(--sans); font-weight: 800; color: var(--text);
  font-size: clamp(1.7rem, 3vw, 2.3rem); line-height: 1.15; letter-spacing: -0.01em;
}}
.cx-hero-sub {{
  margin-top: .5rem; color: var(--text-dim);
  font-size: .95rem; max-width: 64ch; text-wrap: pretty; font-family: var(--sans);
}}

/* ---------- metric: a plain card, one hairline, no glow ---------- */
[data-testid="stMetric"] {{
  background: var(--panel);
  border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.15rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.18);
  contain: layout style;
  transition: background .15s ease, border-color .15s ease;
}}
[data-testid="stMetric"]:hover {{ background: var(--panel-hi); border-color: rgba(31,79,189,.20); }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
  text-transform: uppercase !important; letter-spacing: .07em !important;
  font-size: .66rem !important; color: var(--text-faint) !important;
  font-weight: 600; font-family: var(--mono);
}}
[data-testid="stMetricValue"] {{
  font-weight: 600; color: var(--text); -webkit-text-fill-color: var(--text);
  font-family: var(--mono);
  font-size: 1.6rem !important; letter-spacing: -0.01em; overflow-wrap: anywhere;
}}
[data-testid="stMetricDelta"] {{ font-weight: 500; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: 1.2rem; padding: 0; background: transparent; border: none;
  border-bottom: 1px solid var(--line); border-radius: 0;
}}
.stTabs [data-baseweb="tab"] {{
  height: auto; padding: .5rem .1rem; border-radius: 0; border: none;
  background: transparent; color: var(--text-dim); font-weight: 500;
  font-family: var(--sans); font-size: .94rem;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--text); background: transparent; }}
.stTabs [aria-selected="true"] {{
  color: var(--text) !important; background: transparent !important;
  box-shadow: inset 0 -2px 0 var(--gold);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
  background: transparent !important;
}}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 8px; font-weight: 600; font-family: var(--sans);
  border: 1px solid var(--line); background: var(--panel); color: var(--text);
  transition: border-color .12s ease, background .12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
  border-color: rgba(31,79,189,.20); background: var(--panel-hi);
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{
  background: var(--gold); color: #fff; -webkit-text-fill-color: #fff;
  border-color: var(--gold-deep);
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  background: var(--gold-bright); color: #fff;
}}

/* ---------- sidebar radios (utility drawer) ---------- */
[data-testid="stSidebar"] [role="radiogroup"] > label {{ padding: .4rem .55rem; border-radius: 4px; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
  background: var(--gold-wash); box-shadow: inset 2px 0 0 var(--gold);
}}

/* ---------- sliders ---------- */
[data-baseweb="slider"] [data-testid="stSliderTrack"] {{ background: var(--line) !important; }}
[data-baseweb="slider"] [data-testid="stSliderTrack"] > div {{ background: var(--gold) !important; }}
[data-baseweb="slider"] [role="slider"] {{
  background: var(--gold-bright) !important; border: 2px solid var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-wash) !important;
}}
[data-testid="stSliderThumbValue"] {{
  color: var(--gold-text) !important; font-family: var(--mono); font-variant-numeric: tabular-nums;
}}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] *,
[data-testid="stSliderTickBar"] *, [data-testid="stSliderTickBarMin"] *,
[data-testid="stSliderTickBarMax"] * {{
  white-space: nowrap !important; word-break: keep-all !important; overflow-wrap: normal !important;
}}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] > div {{
  width: auto !important; min-width: max-content !important;
}}
[data-testid="stSliderTickBar"], [data-testid="stTickBar"],
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{
  color: var(--text-faint) !important; font-family: var(--mono);
}}

/* ---------- inputs ---------- */
[data-testid="stWidgetLabel"] p, label p {{ color: var(--text-dim) !important; font-weight: 600; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stDateInput"] input {{
  border-radius: 8px !important; border-color: var(--line) !important;
  background: var(--panel) !important; color: var(--text) !important; font-family: var(--mono);
}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 2px var(--gold-wash) !important;
}}
[data-testid="stToggle"] [data-baseweb="toggle"][aria-checked="true"] > div {{ background: var(--gold) !important; }}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] [aria-selected="true"] {{
  color: var(--gold-text) !important; box-shadow: inset 0 -2px 0 var(--gold);
}}

/* ---------- progress (LTR on an RTL page) ---------- */
[data-testid="stProgressBar"] > div > div, .stProgress > div > div > div {{ background: var(--gold); }}
[data-testid="stProgress"], .stProgress, [data-testid="stProgressBar"],
[data-testid="stProgressBar"] > div {{ direction: ltr !important; }}
[data-testid="stProgressBar"] > div {{
  margin: 0 !important; left: auto !important; right: auto !important;
  transform: none !important; width: 100% !important;
}}

/* ---------- alerts / panels ---------- */
[data-testid="stAlert"], [data-testid="stNotification"],
[data-testid="stAlert"] [data-baseweb="notification"], [data-testid="stAlertContainer"],
[data-testid="stNotificationContentInfo"], [data-testid="stNotificationContentWarning"],
[data-testid="stNotificationContentError"], [data-testid="stNotificationContentSuccess"] {{
  border-radius: 10px; background: var(--panel) !important;
}}
[data-testid="stAlert"] {{ border: 1px solid var(--line); border-inline-start: 3px solid var(--gold); }}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p, [data-testid="stAlertContainer"] p {{
  color: var(--text) !important;
}}
[data-testid="stExpander"] {{
  border-radius: 12px; border: 1px solid var(--line); overflow: hidden;
  background: var(--panel);
  contain: layout style;
}}
[data-testid="stExpander"] summary:hover {{ background: var(--gold-wash); }}
[data-testid="stForm"] {{
  border-radius: 12px; border: 1px solid var(--line);
  background: var(--panel);
  contain: layout style;
}}
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 12px;
  background: var(--panel-hi);
  box-shadow: inset 0 0 0 1px var(--line);
}}
[data-testid="stMetricChart"], [data-testid="stMetricChart"] canvas,
[data-testid="stMetricChart"] svg {{ background: transparent !important; }}
[data-testid="stMetricChart"] {{ opacity: .9; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border-radius: 10px; overflow: hidden; border: 1px solid var(--line);
}}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], .stPlotlyChart {{ border-radius: 10px; }}
hr {{ border-color: var(--line); }}

/* ---------- scrollbars ---------- */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--line); border-radius: 999px; border: 2px solid var(--ink-edge); }}
::-webkit-scrollbar-thumb:hover {{ background: var(--gold-deep); }}

/* ---------- guards (functional -- keep) ---------- */
[data-testid="stAppViewContainer"] {{ overflow-x: hidden; }}
[data-testid="stMain"] {{ overflow-x: clip; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ overflow-x: auto; }}
pre, code {{ white-space: pre-wrap; word-break: break-word; }}
[data-testid="stMetricValue"] {{ overflow-wrap: anywhere; }}
img, iframe, canvas, svg {{ max-width: 100%; }}
[data-testid="stMain"] .katex-display {{ overflow-x: auto; overflow-y: hidden; }}
[data-testid="stMain"] .katex {{ max-width: 100%; overflow-x: auto; overflow-y: hidden; }}
.stTabs [data-baseweb="tab-list"] {{ overflow-x: auto; overflow-y: hidden; flex-wrap: nowrap; }}
.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"] {{ flex: 0 0 auto; white-space: nowrap; }}

@media (max-width: 900px) {{
  [data-testid="stMainBlockContainer"], .block-container {{
    padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important;
  }}
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; row-gap: .6rem; }}
  [data-testid="stColumn"] {{ flex: 1 1 calc(50% - .6rem) !important; min-width: calc(50% - .6rem) !important; }}
  .st-key-cx_nav [role="radiogroup"] {{ justify-content: flex-start; overflow-x: auto; flex-wrap: nowrap; }}
  .st-key-cx_nav [role="radiogroup"] > label {{ flex: 0 0 auto; }}
  .cx-orbit {{ --r: 128px; --size: 288px; margin: .3rem auto 2.2rem; }}
  .cx-orbit-item > button {{ font-size: .62rem; padding: .3rem .58rem; }}
  .cx-orbit-hub {{ width: 64px; height: 64px; }}
}}
@media (max-width: 560px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding: .7rem .6rem 3rem !important; }}
  [data-testid="stColumn"] {{ flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }}
  .cx-hero {{ margin: .7rem 0 1.2rem; }}
  .cx-hero-title {{ font-size: 1.5rem; }}
  h1 {{ font-size: 1.5rem !important; }}  h2 {{ font-size: 1.28rem !important; }}  h3 {{ font-size: 1.08rem !important; }}
  [data-testid="stMetric"] {{ padding: .65rem .75rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.28rem !important; }}
  [data-testid="stMetricLabel"] p {{ white-space: normal; }}
  iframe {{ width: 100% !important; }}
  .stTabs [data-baseweb="tab"] {{ padding: .45rem .7rem; font-size: .86rem; }}
  .cx-orbit {{ --r: 108px; --size: 244px; }}
  .cx-orbit-hub {{ width: 56px; height: 56px; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: .001ms !important; transition-duration: .001ms !important; }}
}}

/* ======================================================================= *
 *  first-load reveal: a one-time, quiet fade-and-rise (compositor-only).
 *  Default state is the FINISHED state, so if the script never runs (or
 *  reduced-motion), everything shows normally -- nothing depends on it.
 * ======================================================================= */
body::after {{
  content: ""; position: fixed; top: 0; left: 0; right: 0; height: 2px;
  z-index: 2147483000; pointer-events: none; opacity: .85;
  background: var(--gold);
  transform-origin: 0 50%; transform: scaleX(var(--cx-progress, 0));
  transition: transform .12s linear;
}}
@media (prefers-reduced-motion: no-preference) {{
  html.cx-js [data-testid="stMain"] .block-container [data-testid="stElementContainer"].cx-hide,
  html.cx-js [data-testid="stMain"] .block-container [data-testid="stHorizontalBlock"].cx-hide {{
    opacity: 0; transform: translateY(14px);
  }}
  html.cx-js .cx-hero.cx-hide {{ opacity: 0; transform: translateY(14px); }}
  html.cx-js .cx-hide {{
    transition: opacity .4s ease, transform .4s ease;
    will-change: opacity, transform;
  }}
  html.cx-js .cx-rev {{ opacity: 1 !important; transform: none !important; will-change: auto; }}
}}

/* ====================== per-section variation (none -- one consistent
   look everywhere, on purpose) ====================== */
{_section_themes()}

/* ============== CARN agent -- a plain, quiet chat panel ============== *
 * Same dark surface as the rest of the app, one accent for your own
 * messages.  No gradients, no glow, no "carved" anything.                */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] .stMarkdown {{ color: var(--text-dim) !important; }}

.cx-agent-head {{
  display: flex; align-items: center; gap: .6rem; padding: .2rem 0 .7rem;
  margin-bottom: .5rem; border-bottom: 1px solid var(--line);
}}
.cx-agent-head svg, .cx-agent-head img {{ flex: 0 0 auto; border-radius: 50%; }}
.cx-agent-id {{ display: flex; flex-direction: column; line-height: 1.2; }}
.cx-agent-id b {{ font-family: var(--sans); font-size: 1.02rem; font-weight: 700; color: var(--text); }}
.cx-agent-id span {{
  font-family: var(--mono); font-size: .62rem; letter-spacing: .06em;
  text-transform: uppercase; color: var(--text-faint);
}}

.cx-a-msg {{
  font-size: .85rem; line-height: 1.55; border-radius: 10px;
  padding: .55rem .72rem; margin: .35rem 0; max-width: 94%; width: fit-content;
  border: 1px solid var(--line);
}}
.cx-a-msg b {{ color: var(--text); }}
.cx-a-msg i {{ color: var(--text-faint); font-style: italic; }}
.cx-a-bot {{ background: var(--panel); color: var(--text-dim); margin-inline-end: auto; }}
.cx-a-me {{
  background: var(--gold-wash); color: var(--text);
  margin-inline-start: auto; border-color: var(--gold-line);
}}

[data-testid="stSidebar"] [data-testid="stTextInput"] input {{
  background: var(--sunk) !important; border: 1px solid var(--line) !important;
  color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
}}
[data-testid="stSidebar"] .stFormSubmitButton button {{
  background: var(--gold) !important; color: #fff !important; -webkit-text-fill-color: #fff !important;
  border: 1px solid var(--gold-deep) !important; font-weight: 600 !important;
}}

.cx-agent-rule {{ height: 1px; margin: .6rem 0 .3rem; background: var(--line); }}

[data-testid="stSidebar"] [data-testid="stMetricValue"] {{ color: var(--text) !important; -webkit-text-fill-color: var(--text) !important; }}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {{ color: var(--text-faint) !important; }}
[data-testid="stSidebar"] hr {{ border-color: var(--line) !important; }}
[data-testid="stSidebar"] [data-testid="stAlert"] {{
  background: var(--panel) !important; border: 1px solid var(--line) !important;
}}
</style>
"""


def _section_themes() -> str:
    # No per-page "design languages" -- one consistent look everywhere.
    return ""


# --------------------------------------------------------------------------- #
# force LTR widget geometry  (react-aria reads navigator.language)
# --------------------------------------------------------------------------- #
_LTR_SHIM = (
    "<script>(function(){try{"
    "var w='en-US';"
    "if((navigator.language||'').slice(0,2).toLowerCase()!=='en'"
    "   || (navigator.languages&&navigator.languages[0]||'').slice(0,2).toLowerCase()!=='en'){"
    "Object.defineProperty(navigator,'language',{get:function(){return w;},configurable:true});"
    "Object.defineProperty(navigator,'languages',{get:function(){return [w];},configurable:true});"
    "window.dispatchEvent(new Event('languagechange'));"
    "requestAnimationFrame(function(){window.dispatchEvent(new Event('languagechange'));});"
    "}"
    "}catch(e){}})();</script>"
)


def inject_theme() -> None:
    """Inject the global skin.  Call once, immediately after set_page_config.
    The scroll engine is installed separately by ``scroll_boot`` (index.html),
    because Streamlit's ``st.html`` strips a raw <script> even with the flag."""
    st.markdown(_css(), unsafe_allow_html=True)
    try:
        st.html(_LTR_SHIM, unsafe_allow_javascript=True)
    except TypeError:
        pass
    _register_altair_theme()


def _asset_data_uri(filename: str) -> str:
    """Inline a PNG from assets/ as a data: URI.

    brand_boot's ``./carnx/*.png`` paths depend on copying files into
    Streamlit's own static directory, which a managed host (e.g. Streamlit
    Community Cloud) can silently refuse to let it write into -- that's
    exactly what left the header logo (and the orbit hub icon) blank on the
    live deploy. A data: URI has no such dependency: the bytes live directly
    in the HTML, so there is nothing left for a restrictive host to block.
    """
    path = os.path.join(os.path.dirname(__file__), "assets", filename)
    try:
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def _wordmark_svg() -> str:
    src = _asset_data_uri("mark.png")
    img = (
        f'<img src="{src}" alt="" '
        'style="height:38px;width:38px;border-radius:9px;display:block;'
        'border:1px solid rgba(31,79,189,.16)">'
        if src
        else '<div style="height:38px;width:38px;border-radius:9px;'
        'border:1px solid rgba(31,79,189,.16)"></div>'
    )
    return (
        img
        + f'<span style="font-family:{_SANS};font-weight:700;font-size:1.14rem;'
        f'color:{TEXT};letter-spacing:-.01em">CARN-X</span>'
    )


def refresh_button() -> None:
    """A small green reload control (clears the data caches and reruns)."""
    if st.button(":material/refresh: רענן נתונים", key="cx_refresh"):
        for clr in (getattr(st, "cache_data", None), getattr(st, "cache_resource", None)):
            try:
                clr.clear()
            except Exception:
                pass
        st.rerun()


def header_nav(sections: list[str] | None = None, default: str = "סקירה") -> str:
    """Masthead + centred menu.  Returns the chosen section (routing unchanged)."""
    sections = sections or _SECTIONS
    ss = st.session_state
    ss.setdefault("cx_section", default)

    top = st.columns([5, 1], vertical_alignment="center")
    with top[0]:
        st.markdown(
            f'<div class="cx-mast">{_wordmark_svg()}</div>',
            unsafe_allow_html=True,
        )
    with top[1]:
        refresh_button()

    with st.container(key="cx_nav"):
        picked = st.radio(
            "ניווט",
            sections,
            index=sections.index(ss["cx_section"]) if ss["cx_section"] in sections else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="cx_nav_radio",
        )
    ss["cx_section"] = picked or default
    return ss["cx_section"]


def hero(title: str, subtitle: str = "", eyebrow: str = "CARN-X") -> None:
    """A plain section heading: a title, an optional sub-line.  Cosmetic only."""
    import html as _h
    t = _h.escape(title)
    parts = ['<div class="cx-hero">', f'<h2 class="cx-hero-title">{t}</h2>']
    if subtitle:
        parts.append(f'<div class="cx-hero-sub">{_h.escape(subtitle)}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def brand() -> None:
    """Back-compat: identity now lives in header_nav()."""
    return None


# --------------------------------------------------------------------------- #
# Altair theme
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
                "font": _SANS,
                "title": {"color": GOLD_TEXT, "fontSize": 14, "font": _SANS, "fontWeight": 600},
                "axis": {
                    "domainColor": "rgba(11,18,32,0.18)",
                    "gridColor": "rgba(11,18,32,0.07)",
                    "tickColor": "rgba(11,18,32,0.18)",
                    "labelColor": TEXT_DIM,
                    "titleColor": TEXT_DIM,
                    "labelFont": _MONO,
                    "titleFont": _SANS,
                },
                "legend": {
                    "labelColor": TEXT_DIM,
                    "titleColor": TEXT_DIM,
                    "labelFont": _SANS,
                    "titleFont": _SANS,
                },
                "range": {
                    "category": CATEGORICAL,
                    "heatmap": ["#EAF2FF", "#8FB8FF", GOLD, GOLD_DEEP],
                    "ramp": ["#EAF2FF", "#8FB8FF", GOLD, GOLD_DEEP],
                },
            }
        }

    try:
        alt.theme.register("carnx", enable=True)(_carnx)  # type: ignore[attr-defined]
        _ALTAIR_DONE = True
    except Exception:
        try:
            alt.themes.register("carnx", _carnx)  # type: ignore[attr-defined]
            alt.themes.enable("carnx")  # type: ignore[attr-defined]
            _ALTAIR_DONE = True
        except Exception:
            pass
