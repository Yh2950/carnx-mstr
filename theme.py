"""
CARN-X  --  presentation layer  (visual only, no logic / no data / no decisions)
==============================================================================
Deep indigo ground, a restrained gilt accent, a serif display face, and a
top navigation bar (no left rail).  A faint mathematical watermark, nothing
loud.

Public surface:
    inject_theme()              call once, right after st.set_page_config
    header_nav(sections)        the top identity + navigation; returns the choice
    hero(title, sub, eyebrow)   drop-in for st.header
    refresh_button()            reload control (rendered by header_nav)
    brand()                     back-compat alias -> no-op

Everything is cosmetic.  Nothing here computes, fetches, caches or decides.
"""

from __future__ import annotations

import base64

import streamlit as st

# --------------------------------------------------------------------------- #
# palette  --  deep indigo, quiet gilt
# --------------------------------------------------------------------------- #
INK = "#0A1024"
INK_2 = "#0B1330"
PANEL = "#101A3E"
SUNK = "#070C1E"
LINE = "#243158"
GOLD = "#D8B45A"
GOLD_BRIGHT = "#F0D48A"
GOLD_DEEP = "#9E7C34"
GOLD_INK = "#D9C08A"
TEXT = "#E9ECF8"
TEXT_DIM = "#98A3C6"
TEXT_FAINT = "#697498"
UP = "#54C08A"
DOWN = "#E06A4E"

CYAN = "#84BEE0"
VIOLET = "#9A93D8"
MAGENTA = "#D291B0"
LIME = "#95C46F"
AMBER = GOLD
CATEGORICAL = [GOLD, CYAN, UP, MAGENTA, VIOLET, "#C7A24E", "#7A88C0", DOWN]

_SERIF = "'Spectral', 'EB Garamond', Georgia, 'Times New Roman', serif"
_SANS = "'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif"

_SECTIONS: list[str] = [
    "סקירה", "טרמינל מסחר", "מחשבון הסתברויות", "אבחון סטטיסטי",
    "תחזית הסתברותית", "Monte Carlo", "מבנים מתמטיים",
    "מחזור ביטקוין → MSTR", "ראיות Walk-Forward", "סיכון ומינוף", "הגדרות",
]


# --------------------------------------------------------------------------- #
# a faint mathematical watermark  --  one large glyph per big tile, low key
# --------------------------------------------------------------------------- #
def _watermark_svg() -> str:
    u = 460
    g = GOLD
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{u}" height="{u}" viewBox="0 0 {u} {u}">'
        f'<text x="70" y="330" font-family="{_SERIF}" font-style="italic" font-size="300" '
        f'fill="{g}" fill-opacity="0.028">&#8747;</text>'
        f'<text x="300" y="120" font-family="{_SERIF}" font-size="54" fill="{g}" '
        f'fill-opacity="0.030">&#931;</text>'
        f'<text x="330" y="410" font-family="{_SERIF}" font-style="italic" font-size="44" '
        f'fill="{g}" fill-opacity="0.030">&#8706;</text>'
        f'<text x="40" y="90" font-family="{_SERIF}" font-style="italic" font-size="34" '
        f'fill="{g}" fill-opacity="0.028">&#960;</text>'
        f"</svg>"
    )


def _watermark_uri() -> str:
    b64 = base64.b64encode(_watermark_svg().encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _css() -> str:
    wm = _watermark_uri()
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Spectral:ital,wght@0,400;0,500;0,600;1,400;1,500&display=swap');

:root {{
  --ink:{INK}; --ink-2:{INK_2}; --panel:{PANEL}; --sunk:{SUNK}; --line:{LINE};
  --gold:{GOLD}; --gold-bright:{GOLD_BRIGHT}; --gold-deep:{GOLD_DEEP}; --gold-ink:{GOLD_INK};
  --gold-wash:rgba(216,180,90,0.10); --gold-line:rgba(216,180,90,0.30);
  --text:{TEXT}; --text-dim:{TEXT_DIM}; --text-faint:{TEXT_FAINT};
  --up:{UP}; --down:{DOWN};
}}

/* ---------- the ground ---------- */
html, body, [class*="stApp"] {{ font-family: {_SANS}; }}
.stApp {{
  background:
    url("{wm}") 0 0 / 460px 460px repeat,
    linear-gradient(180deg, {INK_2} 0%, {INK} 55%, {SUNK} 100%);
  background-attachment: fixed, fixed;
  color: var(--text);
}}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: transparent; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: .5rem; }}
[data-testid="stAppDeployButton"], [data-testid="stDeployButton"] {{ display: none !important; }}
[data-testid="stMain"] .block-container {{ padding-top: 1.2rem; max-width: 1200px; }}

[data-testid="stSidebarCollapsedControl"] {{ opacity: .55; }}
[data-testid="stSidebar"] {{ background: var(--sunk); border-right: 1px solid var(--line); }}

/* ---------- top bar: wordmark + reload ---------- */
.cx-topbar {{
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: .1rem 0 .55rem; margin-bottom: .5rem;
}}
.cx-topbar .cx-word {{ display: flex; align-items: center; gap: .6rem; }}
.cx-topbar svg {{ height: 42px; width: auto; display: block; }}
.cx-topbar .cx-tag {{ color: var(--text-faint); font-size: .78rem; }}
.cx-reload button {{ font-size: .8rem !important; padding: .32rem .8rem !important; }}

/* ---------- top navigation (a keyed radio -> a menu bar) ---------- */
.st-key-cx_nav [role="radiogroup"] {{
  flex-direction: row; flex-wrap: wrap; gap: .05rem .2rem; align-items: stretch;
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
  padding: .12rem 0; margin-bottom: 1.5rem;
}}
.st-key-cx_nav [role="radiogroup"] > label {{
  margin: 0; padding: .42rem .8rem; border-radius: 0; cursor: pointer;
  border-bottom: 2px solid transparent; transition: border-color .13s ease;
}}
.st-key-cx_nav [data-testid="stRadioOption"] > div > div > div:first-child {{
  display: none !important;
}}
.st-key-cx_nav [role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {{
  color: var(--text-dim) !important; font-weight: 500; font-size: .92rem; white-space: nowrap;
}}
.st-key-cx_nav [role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {{ color: var(--text) !important; }}
.st-key-cx_nav [role="radiogroup"] > label:has(input:checked) {{ border-bottom-color: var(--gold); }}
.st-key-cx_nav [role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {{
  color: var(--gold-bright) !important; font-weight: 600;
}}

/* ---------- typography ---------- */
h1, h2, h3, h4 {{ font-family: {_SERIF}; letter-spacing: -0.002em; text-wrap: balance; }}
h1, h2 {{ color: var(--gold-bright); font-weight: 600; }}
h3, h4 {{ color: var(--text); font-weight: 600; }}
h1 {{ font-size: 1.85rem; }}  h2 {{ font-size: 1.45rem; }}  h3 {{ font-size: 1.16rem; }}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {{
  color: #CDD4EC; text-wrap: pretty; line-height: 1.6;
}}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong {{ color: var(--gold-ink); font-weight: 600; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--text-dim); font-size: .84rem; }}
a, a:visited {{ color: var(--gold-bright); text-underline-offset: 2px; }}
code, pre, kbd {{ font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace; }}
:not(pre) > code {{
  background: var(--gold-wash); color: var(--gold-bright);
  border: 1px solid var(--gold-line); border-radius: 4px; padding: .05em .4em; font-size: .86em;
}}
pre, [data-testid="stCode"] {{ background: var(--sunk) !important; border: 1px solid var(--line); border-radius: 6px; }}
.katex {{ color: var(--text); }}

/* ---------- section head ---------- */
.cx-hero {{ position: relative; margin: .1rem 0 1.5rem; padding: 0 0 .85rem; }}
.cx-hero::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 1px;
  background: linear-gradient(90deg, var(--gold) 0 42px, var(--line) 42px 100%); }}
.cx-hero-eyebrow {{ display: block; margin: 0 0 .28rem; font-family: {_SERIF};
  font-style: italic; font-size: .84rem; color: var(--gold); letter-spacing: .02em; }}
.cx-hero-title {{ font-family: {_SERIF}; font-weight: 600; color: var(--gold-bright);
  font-size: clamp(1.5rem, 2.3vw, 2.05rem); line-height: 1.16; letter-spacing: -0.004em; }}
.cx-hero-sub {{ margin-top: .4rem; color: var(--text-dim); font-size: .93rem; max-width: 72ch; text-wrap: pretty; }}

/* ---------- metric ---------- */
[data-testid="stMetric"] {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 7px;
  padding: .85rem 1rem .8rem; transition: border-color .14s ease;
}}
[data-testid="stMetric"]:hover {{ border-color: var(--gold-line); }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
  text-transform: none !important; letter-spacing: 0 !important;
  font-size: .8rem !important; color: var(--gold-ink) !important; font-weight: 500;
  font-family: {_SERIF}; font-style: italic;
}}
[data-testid="stMetricValue"] {{
  font-family: {_SERIF}; font-weight: 600; color: var(--text);
  -webkit-text-fill-color: var(--text);
  font-variant-numeric: tabular-nums; font-size: 1.62rem !important; overflow-wrap: anywhere;
}}
[data-testid="stMetricDelta"] {{ font-weight: 600; font-variant-numeric: tabular-nums; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: 1.2rem; padding: 0; background: transparent; border: none;
  border-bottom: 1px solid var(--line); border-radius: 0;
}}
.stTabs [data-baseweb="tab"] {{
  height: auto; padding: .5rem .1rem; border-radius: 0; border: none;
  background: transparent; color: var(--text-dim); font-weight: 500; font-family: {_SERIF};
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--gold-ink); background: transparent; }}
.stTabs [aria-selected="true"] {{
  color: var(--gold-bright) !important; background: transparent !important;
  box-shadow: inset 0 -2px 0 var(--gold);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ background: transparent !important; }}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 6px; font-weight: 600; letter-spacing: 0;
  border: 1px solid var(--gold-line); background: var(--panel); color: var(--gold-ink);
  transition: border-color .13s ease, color .13s ease, background .13s ease; box-shadow: none;
}}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  border-color: var(--gold); color: var(--gold-bright); background: var(--gold-wash); transform: none;
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{
  background: var(--gold); color: #1A1606; border-color: var(--gold-deep);
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  filter: brightness(1.05); color: #1A1606;
}}

/* ---------- sidebar (a utility drawer; collapsed by default) ---------- */
[data-testid="stSidebar"] [role="radiogroup"] > label {{ padding: .4rem .55rem; border-radius: 5px; }}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
  background: var(--gold-wash); box-shadow: inset 2px 0 0 var(--gold);
}}

/* ---------- sliders ---------- */
[data-baseweb="slider"] [data-testid="stSliderTrack"] {{ background: var(--line) !important; }}
[data-baseweb="slider"] [data-testid="stSliderTrack"] > div {{ background: var(--gold) !important; }}
[data-baseweb="slider"] [role="slider"] {{
  background: var(--gold-bright) !important; border: 2px solid var(--gold) !important;
  box-shadow: 0 0 0 4px var(--gold-wash) !important;
}}
[data-testid="stSliderThumbValue"] {{ color: var(--gold-bright) !important; font-variant-numeric: tabular-nums; }}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] *,
[data-testid="stSliderTickBar"] *, [data-testid="stSliderTickBarMin"] *,
[data-testid="stSliderTickBarMax"] * {{
  white-space: nowrap !important; word-break: keep-all !important; overflow-wrap: normal !important;
}}
[data-testid="stSliderThumbValue"], [data-testid="stSliderThumbValue"] > div {{
  width: auto !important; min-width: max-content !important;
}}
[data-testid="stSliderTickBar"], [data-testid="stTickBar"],
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color: var(--text-faint) !important; }}

/* ---------- inputs ---------- */
[data-testid="stWidgetLabel"] p, label p {{ color: var(--gold-ink) !important; font-weight: 500; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stDateInput"] input {{
  border-radius: 6px !important; border-color: var(--line) !important;
  background: var(--sunk) !important; color: var(--text) !important;
}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{ border-color: var(--gold) !important; }}
[data-testid="stToggle"] [data-baseweb="toggle"][aria-checked="true"] > div {{ background: var(--gold) !important; }}
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] [aria-selected="true"] {{
  color: var(--gold-bright) !important; box-shadow: inset 0 -2px 0 var(--gold);
}}

/* ---------- progress (LTR on an RTL page) ---------- */
[data-testid="stProgressBar"] > div > div, .stProgress > div > div > div {{ background: var(--gold); }}
[data-testid="stProgress"], .stProgress, [data-testid="stProgressBar"],
[data-testid="stProgressBar"] > div {{ direction: ltr !important; }}
[data-testid="stProgressBar"] > div {{
  margin: 0 !important; left: auto !important; right: auto !important; transform: none !important; width: 100% !important;
}}

/* ---------- alerts / panels ---------- */
[data-testid="stAlert"], [data-testid="stNotification"],
[data-testid="stAlert"] [data-baseweb="notification"], [data-testid="stAlertContainer"],
[data-testid="stNotificationContentInfo"], [data-testid="stNotificationContentWarning"],
[data-testid="stNotificationContentError"], [data-testid="stNotificationContentSuccess"] {{
  border-radius: 7px; background: var(--panel) !important;
}}
[data-testid="stAlert"] {{ border: 1px solid var(--line); border-left: 3px solid var(--gold); }}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p, [data-testid="stAlertContainer"] p {{ color: var(--text) !important; }}
[data-testid="stExpander"] {{ border-radius: 7px; border: 1px solid var(--line); background: var(--panel); overflow: hidden; }}
[data-testid="stExpander"] summary:hover {{ background: var(--gold-wash); }}
[data-testid="stForm"] {{ border-radius: 8px; border: 1px solid var(--line); background: var(--panel); }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 8px; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border-radius: 7px; overflow: hidden; border: 1px solid var(--line); }}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], .stPlotlyChart {{ border-radius: 7px; }}
hr {{ border-color: var(--line); }}

/* ---------- scrollbars ---------- */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--line); border-radius: 999px; border: 2px solid var(--ink); }}
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
  .st-key-cx_nav [role="radiogroup"] {{ overflow-x: auto; flex-wrap: nowrap; }}
  .st-key-cx_nav [role="radiogroup"] > label {{ flex: 0 0 auto; }}
}}
@media (max-width: 560px) {{
  .stApp {{ background-size: 300px 300px, auto; }}
  [data-testid="stMainBlockContainer"], .block-container {{ padding: .7rem .6rem 3rem !important; }}
  [data-testid="stColumn"] {{ flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }}
  .cx-topbar {{ flex-direction: column; align-items: flex-start; gap: .4rem; }}
  .cx-hero-title {{ font-size: 1.3rem !important; line-height: 1.22; }}
  h1 {{ font-size: 1.36rem !important; }}  h2 {{ font-size: 1.2rem !important; }}  h3 {{ font-size: 1.04rem !important; }}
  [data-testid="stMetric"] {{ padding: .65rem .75rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
  [data-testid="stMetricLabel"] p {{ white-space: normal; }}
  iframe {{ width: 100% !important; }}
  .stTabs [data-baseweb="tab"] {{ padding: .45rem .7rem; font-size: .82rem; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: .001ms !important; transition-duration: .001ms !important; }}
}}
</style>
"""


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
    """Inject the global skin.  Call once, immediately after set_page_config."""
    st.markdown(_css(), unsafe_allow_html=True)
    try:
        st.html(_LTR_SHIM, unsafe_allow_javascript=True)
    except TypeError:
        pass
    _register_altair_theme()


def _wordmark_svg() -> str:
    try:
        from assets.brandmark import svg_wordmark

        return svg_wordmark(430, 96)
    except Exception:  # pragma: no cover
        return (
            f'<span style="font-family:{_SERIF};font-weight:600;font-size:1.2rem;'
            f'color:{GOLD_BRIGHT};letter-spacing:1px">CARN-X</span>'
        )


def refresh_button() -> None:
    """A small reload control (clears the data caches and reruns)."""
    st.markdown('<div class="cx-reload">', unsafe_allow_html=True)
    if st.button("↻  רענן נתונים", key="cx_refresh"):
        for clr in (getattr(st, "cache_data", None), getattr(st, "cache_resource", None)):
            try:
                clr.clear()
            except Exception:
                pass
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def header_nav(sections: list[str] | None = None, default: str = "סקירה") -> str:
    """Top identity bar + horizontal navigation, all in the centre column.
    Returns the chosen section name (routing is unchanged downstream)."""
    sections = sections or _SECTIONS
    ss = st.session_state
    ss.setdefault("cx_section", default)

    top = st.columns([5, 1], vertical_alignment="center")
    with top[0]:
        st.markdown(
            f'<div class="cx-topbar"><div class="cx-word">{_wordmark_svg()}</div>'
            f'<span class="cx-tag">מודל הסתברותי — MSTR · Bitcoin</span></div>',
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
    """Drop-in for ``st.header``: a ruled chapter opening in gilt serif."""
    parts = ['<div class="cx-hero">']
    if eyebrow:
        parts.append(f'<span class="cx-hero-eyebrow">{eyebrow}</span>')
    parts.append(f'<div class="cx-hero-title">{title}</div>')
    if subtitle:
        parts.append(f'<div class="cx-hero-sub">{subtitle}</div>')
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
                "title": {"color": GOLD_INK, "fontSize": 14, "font": _SERIF, "fontWeight": 600},
                "axis": {
                    "domainColor": "rgba(160,175,220,0.22)",
                    "gridColor": "rgba(160,175,220,0.08)",
                    "tickColor": "rgba(160,175,220,0.22)",
                    "labelColor": TEXT_DIM,
                    "titleColor": TEXT_DIM,
                    "labelFont": _SANS,
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
                    "heatmap": ["#0C1435", "#3C3E7E", GOLD],
                    "ramp": ["#0C1435", "#5E5A93", GOLD],
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
