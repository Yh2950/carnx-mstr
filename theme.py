"""
CARN-X  --  presentation layer  (visual only; no logic, no data, no decisions)
=============================================================================
A lacquered midnight-blue instrument face.  Behind everything sits ONE large,
hand-composed engraving in gilt -- a Gaussian sweep with its Riemann sum, an
integral, a fragment of Pascal's triangle, a Fibonacci run, a small graph,
matrix rules, a few notes of music -- placed once and bled off every edge, so it
reads as an engraved plate rather than a tiled pattern.  Display type is a
cut-serif (Fraunces); the working type is IBM Plex Sans / Plex Mono.

Public surface (unchanged):
    inject_theme()               once, right after st.set_page_config
    header_nav(sections, default) top identity + centred menu; returns the pick
    hero(title, subtitle, eyebrow)  a ruled chapter opening
    refresh_button()             reload control (drawn by header_nav)
    brand()                      back-compat -> no-op

Nothing here computes, fetches, caches or decides.
"""

from __future__ import annotations

import base64
import math

import streamlit as st

# --------------------------------------------------------------------------- #
# palette  --  near-black navy lacquer, one metallic gilt
# --------------------------------------------------------------------------- #
INK = "#0A0E1C"          # plate centre
INK_EDGE = "#05070F"     # plate rim / deepest well
PANEL = "#0F1528"        # a raised card
PANEL_HI = "#141D33"     # raised + hover
SUNK = "#080B15"         # sunk well / code
LINE = "#1F2A47"         # hairline
LINE_SOFT = "#141C31"    # faint hairline

GOLD = "#D8B25C"         # the accent
GOLD_BRIGHT = "#F4E1AC"  # struck highlight
GOLD_DEEP = "#8A6A2E"    # engraved shadow
GOLD_TEXT = "#E6CF98"    # gilt used as text on dark (a shade lighter, for legibility)

TEXT = "#E7EAF5"
TEXT_DIM = "#98A2C0"
TEXT_FAINT = "#5F6A8C"

UP = "#58BE8B"
DOWN = "#DE6B54"

CYAN = "#83BDE0"
VIOLET = "#9A93D8"
MAGENTA = "#D291B0"
LIME = "#95C46F"
AMBER = GOLD
CATEGORICAL = [GOLD, CYAN, UP, MAGENTA, VIOLET, "#C7A24E", "#7A88C0", DOWN]

_DISPLAY = "'Fraunces', 'Frank Ruhl Libre', 'Spectral', 'EB Garamond', Georgia, serif"
_SANS = "'IBM Plex Sans', 'Heebo', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
_MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400"
    "&family=Frank+Ruhl+Libre:wght@500;600;700"
    "&family=Heebo:wght@400;500;600;700"
    "&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400"
    "&family=IBM+Plex+Mono:wght@400;500;600"
    "&display=swap"
)

_SECTIONS: list[str] = [
    "סקירה", "טרמינל מסחר", "מחשבון הסתברויות", "אבחון סטטיסטי",
    "תחזית הסתברותית", "Monte Carlo", "מבנים מתמטיים",
    "מחזור ביטקוין → MSTR", "ראיות Walk-Forward", "סיכון ומינוף", "הגדרות",
]


# --------------------------------------------------------------------------- #
# the engraving  --  one composed plate, 1600x1000, bled off every edge
# --------------------------------------------------------------------------- #
_ENGRAVE_GOLD = "#EAD08C"   # a brighter gilt for the plate art
_ENGRAVE_MUL = 2.8         # global presence of the engraving


def _engraving_svg() -> str:
    W, H = 1600, 1000
    g = _ENGRAVE_GOLD
    m = _ENGRAVE_MUL
    out: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid slice">'
    ]

    def glyph(ch, x, y, size, op, *, italic=False, mono=False, weight=400,
              anchor="start", rot=0.0):
        fam = _MONO if mono else _DISPLAY
        sty = ' font-style="italic"' if italic else ""
        tr = f' transform="rotate({rot} {x} {y})"' if rot else ""
        return (
            f'<text x="{x}" y="{y}" font-family="{fam}"{sty} font-weight="{weight}" '
            f'font-size="{size}" fill="{g}" fill-opacity="{min(op * m, 0.5):.4f}" '
            f'text-anchor="{anchor}"{tr}>{ch}</text>'
        )

    def sop(v: float) -> str:
        return f"{min(v * m, 0.5):.4f}"

    # -- 1. the great integral, bleeding off the left edge -------------------
    out.append(glyph("&#8747;", -70, 760, 1180, 0.030, italic=True, weight=500))

    # -- 2. a Gaussian sweep with its Riemann sum, low on the plate ---------
    base = 690.0
    amp = 300.0
    mu, sig = 880.0, 250.0
    xs = [120 + i * (1360 / 200) for i in range(201)]

    def bell(x):
        return base - amp * math.exp(-((x - mu) ** 2) / (2 * sig * sig))

    d = "M" + " L".join(f"{x:.1f} {bell(x):.1f}" for x in xs)
    out.append(
        f'<path d="{d}" fill="none" stroke="{g}" stroke-opacity="{sop(0.075)}" '
        f'stroke-width="2.6" stroke-linecap="round"/>'
    )
    # Riemann columns under the peak
    cols = []
    for k in range(-6, 7):
        cx = mu + k * 92
        top = bell(cx)
        cols.append(
            f'<rect x="{cx - 44:.1f}" y="{top:.1f}" width="88" height="{base - top:.1f}" '
            f'fill="{g}" fill-opacity="{sop(0.022)}" stroke="{g}" stroke-opacity="{sop(0.05)}" stroke-width="1"/>'
        )
    out.append("".join(cols))
    out.append(
        f'<line x1="90" y1="{base}" x2="1520" y2="{base}" stroke="{g}" '
        f'stroke-opacity="{sop(0.09)}" stroke-width="1.4"/>'
    )
    out.append(glyph("dx", mu + 190, base + 46, 40, 0.06, italic=True))
    out.append(glyph("&#956;", mu - 6, base + 52, 34, 0.06, italic=True, anchor="middle"))

    # -- 3. a scatter of notation, hand-placed (not gridded) ---------------
    marks = [
        ("&#931;", 1170, 430, 360, 0.028, {"weight": 500}),           # Sigma
        ("&#8706;", 470, 980, 250, 0.028, {"italic": True}),          # partial
        ("&#8711;", 1000, 205, 120, 0.04, {}),                        # nabla
        ("&#8734;", 240, 250, 130, 0.038, {}),                        # infinity
        ("&#960;", 1430, 930, 150, 0.036, {"italic": True}),          # pi
        ("&#955;", 96, 470, 92, 0.045, {"italic": True}),             # lambda
        ("&#950;", 1330, 150, 104, 0.038, {"italic": True}),          # zeta
        ("&#8501;", 690, 940, 96, 0.032, {}),                         # aleph
        ("&#8730;", 560, 210, 128, 0.036, {}),                        # radical
        ("&#916;", 350, 590, 88, 0.042, {}),                          # Delta
        ("&#8747;&#8869;", 1250, 640, 150, 0.03, {"italic": True}),   # contour-ish
        ("&#8721;", 150, 830, 92, 0.04, {}),                          # sum (small)
        ("&#8869;", 940, 470, 70, 0.045, {}),                         # perp
        ("&#8744;&#8743;", 780, 150, 60, 0.04, {}),                   # and/or
    ]
    for ch, x, y, s, op, kw in marks:
        out.append(glyph(ch, x, y, s, op, **kw))

    # short set lines, in mono, like margin notes
    notes = [
        ("e^{i&#960;} + 1 = 0", 430, 430, 46, 0.055),
        ("&#119978;(n log n)", 1010, 880, 44, 0.055),
        ("P(X &#8804; x)", 175, 930, 42, 0.055),
        ("f&#8242;(x) = lim h&#8594;0", 250, 300, 30, 0.05),
        ("&#8721; 1/n&#178; = &#960;&#178;/6", 1230, 300, 30, 0.05),
    ]
    for txt, x, y, s, op in notes:
        out.append(glyph(txt, x, y, s, op, mono=True))

    # -- 4. a fragment of Pascal's triangle -------------------------------
    tri = [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
    px, py, step = 600, 470, 34
    prows = []
    for r, row in enumerate(tri):
        for c, v in enumerate(row):
            gx = px + (c - r / 2) * step
            gy = py + r * step
            prows.append(glyph(str(v), gx, gy, 22, 0.05, mono=True, anchor="middle"))
    out.append("".join(prows))

    # -- 5. a Fibonacci run along a gentle arc ----------------------------
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    fx0, fy0 = 700, 300
    frow = []
    for i, v in enumerate(fib):
        gx = fx0 + i * 62
        gy = fy0 - 26 * math.sin(i / 3.2)
        frow.append(glyph(str(v), gx, gy, 30, 0.05, mono=True, anchor="middle"))
    out.append("".join(frow))

    # -- 6. matrix rules with entries -----------------------------------
    mx, my = 1245, 560
    out.append(
        f'<path d="M{mx} {my} h-16 v150 h16 M{mx+120} {my} h16 v150 h-16" '
        f'fill="none" stroke="{g}" stroke-opacity="{sop(0.05)}" stroke-width="2"/>'
    )
    for i in range(2):
        for j in range(2):
            out.append(glyph("a", mx + 20 + j * 64, my + 46 + i * 62, 26,
                             0.045, italic=True))

    # -- 7. a small graph (nodes + edges) ------------------------------
    nodes = [(150, 170), (250, 120), (330, 210), (210, 260), (120, 300)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 3), (3, 4)]
    gp = []
    for a, b in edges:
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        gp.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{g}" '
            f'stroke-opacity="{sop(0.06)}" stroke-width="1.3"/>'
        )
    for x, y in nodes:
        gp.append(
            f'<circle cx="{x}" cy="{y}" r="4.5" fill="{INK}" stroke="{g}" '
            f'stroke-opacity="{sop(0.10)}" stroke-width="1.4"/>'
        )
    out.append("".join(gp))

    # -- 8. a few notes of music, on a short stave --------------------
    sx, sy = 830, 110
    for k in range(4):
        out.append(
            f'<line x1="{sx}" y1="{sy + k*9}" x2="{sx+220}" y2="{sy + k*9}" '
            f'stroke="{g}" stroke-opacity="{sop(0.05)}" stroke-width="1"/>'
        )
    for ch, dx, dy, s in (("&#9834;", 20, 34, 44), ("&#9835;", 90, 20, 40),
                          ("&#9833;", 150, 30, 34), ("&#9834;", 190, 24, 40)):
        out.append(glyph(ch, sx + dx, sy + dy, s, 0.06))

    out.append("</svg>")
    return "".join(out)


def _engraving_uri() -> str:
    b64 = base64.b64encode(_engraving_svg().encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _css() -> str:
    eng = _engraving_uri()
    return f"""
<style>
@import url('{_FONTS}');

@property --cx-sec-i {{ syntax: "<number>"; inherits: true; initial-value: 0; }}

:root {{
  --ink:{INK}; --ink-edge:{INK_EDGE}; --panel:{PANEL}; --panel-hi:{PANEL_HI};
  --sunk:{SUNK}; --line:{LINE}; --line-soft:{LINE_SOFT};
  --gold:{GOLD}; --gold-bright:{GOLD_BRIGHT}; --gold-deep:{GOLD_DEEP}; --gold-text:{GOLD_TEXT};
  --gold-wash:rgba(216,178,92,0.09); --gold-line:rgba(216,178,92,0.28);
  --text:{TEXT}; --text-dim:{TEXT_DIM}; --text-faint:{TEXT_FAINT};
  --up:{UP}; --down:{DOWN};
  --display:{_DISPLAY}; --sans:{_SANS}; --mono:{_MONO};
  --cx-sec-i: 0;
  transition: --cx-sec-i .9s cubic-bezier(.16,.72,.24,1);
}}

/* ---------- the living plate ---------- *
 * back -> front:  aurora mesh (body::before, always drifting)
 *                 engraved plate (.stApp::before, scroll + pointer parallax,
 *                                 hue/scale shift per section)
 *                 vignette (.stApp::after)
 *                 transition wipe (.cx-wipe, one sweep on section change)      */
html, body, [class*="stApp"] {{ font-family: var(--sans); }}
.stApp {{ background: {INK}; color: var(--text); }}

body::before {{
  content: ""; position: fixed; inset: -12vmax; z-index: 0; pointer-events: none;
  --a1x: 22%; --a1y: 16%; --a2x: 82%; --a2y: 30%; --a3x: 52%; --a3y: 88%;
  background:
    radial-gradient(38vmax 34vmax at calc(var(--a1x) + var(--cx-amx,0px)) var(--a1y),
      rgba(216,178,92,0.10) 0%, rgba(216,178,92,0) 62%),
    radial-gradient(44vmax 40vmax at calc(var(--a2x) - var(--cx-amx,0px)) var(--a2y),
      rgba(58,74,132,0.20) 0%, rgba(58,74,132,0) 60%),
    radial-gradient(50vmax 46vmax at var(--a3x) calc(var(--a3y) + var(--cx-amy,0px)),
      rgba(120,96,168,0.12) 0%, rgba(120,96,168,0) 64%);
  filter: hue-rotate(var(--cx-hue,0deg)) saturate(var(--cx-sat,1));
  transition: filter 1.1s ease;
  animation: cx-aurora 46s ease-in-out infinite alternate;
  will-change: transform;
}}
.stApp::before {{
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    linear-gradient(178deg, rgba(20,29,54,0.42) 0%, rgba(10,14,28,0) 34%),
    url("{eng}") center top / cover no-repeat;
  transform: translate3d(var(--cx-mx,0px), calc(var(--cx-plate,0px) + var(--cx-my,0px)), 0)
             rotate(var(--cx-rot,0deg)) scale(var(--cx-scale,1.08));
  filter: hue-rotate(calc(var(--cx-hue,0deg) * .55)) saturate(var(--cx-sat,1));
  transition: transform .16s linear, filter .8s ease;
  will-change: transform;
}}
.stApp::after {{
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(160% 130% at 50% 4%, rgba(6,8,15,0) 55%, rgba(5,7,14,0.5) 100%);
}}

/* one gilt sweep across the plate whenever the section changes */
.cx-wipe {{
  position: fixed; inset: 0 -40vw; z-index: 4; pointer-events: none;
  opacity: 0; transform: translateX(-120%) skewX(-14deg);
  background: linear-gradient(90deg,
    rgba(216,178,92,0) 0%, rgba(216,178,92,0.05) 38%,
    rgba(244,225,172,0.16) 50%, rgba(216,178,92,0.05) 62%, rgba(216,178,92,0) 100%);
}}
.cx-wipe.run {{ animation: cx-wipe .66s cubic-bezier(.66,0,.30,1); }}

@keyframes cx-aurora {{
  0%   {{ transform: translate3d(0,0,0) scale(1); }}
  50%  {{ transform: translate3d(2.4vmax,-1.8vmax,0) scale(1.05) rotate(1.4deg); }}
  100% {{ transform: translate3d(-2vmax,2vmax,0) scale(1.08) rotate(-1.2deg); }}
}}
@keyframes cx-wipe {{
  0%   {{ opacity: 0; transform: translateX(-120%) skewX(-14deg); }}
  22%  {{ opacity: 1; }}
  100% {{ opacity: 0; transform: translateX(120%) skewX(-14deg); }}
}}

/* per-section: shift the whole plate's hue + the engraving's rest pose.
   --cx-sec-i (0..10) is set on <html> by scroll_boot from the active menu item. */
:root {{
  --cx-hue: calc((var(--cx-sec-i, 0) - 5) * 5deg);
  --cx-sat: calc(1 + (var(--cx-sec-i, 0) - 5) * 0.015);
  --cx-rot: calc((var(--cx-sec-i, 0) - 5) * 0.5deg);
  --cx-scale: calc(1.065 + var(--cx-sec-i, 0) * 0.006);
}}
body::before {{
  --a1x: calc(20% + var(--cx-sec-i, 0) * 3.4%);
  --a2x: calc(84% - var(--cx-sec-i, 0) * 2.6%);
  --a3y: calc(90% - var(--cx-sec-i, 0) * 2.4%);
}}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], [data-testid="stSidebarContent"] {{ background: transparent; }}
[data-testid="stAppViewContainer"] > .main, .block-container {{ position: relative; z-index: 1; }}
[data-testid="stToolbar"] {{ right: .5rem; }}
[data-testid="stAppDeployButton"], [data-testid="stDeployButton"] {{ display: none !important; }}
[data-testid="stMain"] .block-container {{ padding-top: 1.1rem; max-width: 1180px; }}
[data-testid="stSidebarCollapsedControl"] {{ opacity: .5; }}
[data-testid="stSidebar"] {{ background: var(--ink-edge); border-right: 1px solid var(--line-soft); }}

/* ---------- masthead ---------- */
.cx-mast {{
  display: flex; align-items: baseline; gap: .8rem;
  padding: .1rem 0 .5rem; flex-wrap: wrap;
}}
.cx-mast svg {{ height: 44px; width: auto; display: block; }}
.cx-mast .cx-sub {{
  color: var(--text-faint); font-size: .74rem; letter-spacing: .14em;
  text-transform: uppercase; font-weight: 500;
}}
/* the data-refresh control -- a small GitHub-green pill that reads as native */
.st-key-cx_refresh button, .cx-reload button {{
  font-family: var(--sans) !important; font-size: .78rem !important; font-weight: 600 !important;
  letter-spacing: .01em !important; text-transform: none !important;
  padding: .32rem .8rem !important; border-radius: 6px !important;
  color: #fff !important; -webkit-text-fill-color: #fff !important;
  background: #238636 !important;
  border: 1px solid rgba(240,246,252,0.10) !important;
  box-shadow: 0 1px 0 rgba(27,31,36,0.10), inset 0 1px 0 rgba(255,255,255,0.06) !important;
  transition: background .12s ease, transform .06s ease !important;
}}
.st-key-cx_refresh button:hover, .cx-reload button:hover {{
  background: #2ea043 !important; border-color: rgba(240,246,252,0.13) !important;
  color: #fff !important; -webkit-text-fill-color: #fff !important;
}}
.st-key-cx_refresh button:active, .cx-reload button:active {{
  background: #2c974b !important; transform: translateY(.5px);
}}
.st-key-cx_refresh button:focus-visible, .cx-reload button:focus-visible {{
  outline: 2px solid rgba(46,160,67,.55) !important; outline-offset: 2px;
}}
.st-key-cx_refresh button p, .cx-reload button p {{ color: #fff !important; }}

/* ---------- the menu (a keyed radio, drawn as a centred bar) ---------- */
.st-key-cx_nav {{ margin: .1rem 0 1.7rem; border-bottom: 1px solid var(--line); }}
.st-key-cx_nav [role="radiogroup"] {{
  flex-direction: row; flex-wrap: wrap; justify-content: center;
  gap: 0 .1rem; padding: .2rem 0 0;
}}
.st-key-cx_nav [role="radiogroup"] > label {{
  margin: 0; padding: .5rem .7rem .58rem; border-radius: 0; cursor: pointer;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
  transition: color .12s ease, border-color .12s ease;
}}
.st-key-cx_nav [data-testid="stRadioOption"] > div > div > div:first-child {{
  display: none !important;
}}
.st-key-cx_nav [role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {{
  color: var(--text-dim) !important; font-weight: 500; font-size: .86rem;
  white-space: nowrap; letter-spacing: .004em;
}}
.st-key-cx_nav [role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {{
  color: var(--text) !important;
}}
.st-key-cx_nav [role="radiogroup"] > label:has(input:checked) {{ border-bottom-color: var(--gold); }}
.st-key-cx_nav [role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {{
  color: var(--gold-bright) !important; font-weight: 600;
}}

/* ---------- typography ---------- */
h1, h2, h3, h4, h5 {{
  font-family: var(--display); letter-spacing: -0.006em; text-wrap: balance;
  font-optical-sizing: auto;
}}
h1, h2 {{ color: var(--gold-bright); font-weight: 600; }}
h3, h4, h5 {{ color: var(--text); font-weight: 600; }}
h1 {{ font-size: 1.95rem; }}  h2 {{ font-size: 1.5rem; }}
h3 {{ font-size: 1.18rem; }}  h4 {{ font-size: 1.02rem; }}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {{
  color: #CCD3EC; line-height: 1.62; text-wrap: pretty;
}}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong {{
  color: var(--gold-text); font-weight: 600;
}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
  color: var(--text-dim); font-size: .83rem; letter-spacing: .006em;
}}
a, a:visited {{ color: var(--gold-bright); text-underline-offset: 2px; }}
code, pre, kbd, [data-testid="stMetricValue"], [data-testid="stMetricDelta"],
.cx-num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
:not(pre) > code {{
  background: var(--gold-wash); color: var(--gold-bright);
  border: 1px solid var(--gold-line); border-radius: 4px; padding: .05em .38em; font-size: .84em;
}}
pre, [data-testid="stCode"] {{
  background: var(--sunk) !important; border: 1px solid var(--line); border-radius: 6px;
}}
.katex {{ color: var(--text); }}

/* ---------- chapter opening ---------- */
.cx-hero {{ position: relative; margin: .2rem 0 1.6rem; padding: 0 0 .9rem; }}
.cx-hero::after {{
  content: ""; position: absolute; inset-inline: 0; bottom: 0; height: 1px;
  background: linear-gradient(90deg, var(--gold) 0 54px, var(--line) 54px 100%);
}}
.cx-hero-eyebrow {{
  display: block; margin: 0 0 .35rem; font-family: var(--mono);
  font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: var(--gold);
}}
.cx-hero-title {{
  font-family: var(--display); font-weight: 600; color: var(--gold-bright);
  font-size: clamp(1.55rem, 2.4vw, 2.15rem); line-height: 1.14; letter-spacing: -0.01em;
}}
.cx-hero-sub {{
  margin-top: .45rem; color: var(--text-dim); font-size: .92rem; max-width: 74ch;
  text-wrap: pretty;
}}

/* ---------- metric  (an engraved plaque, no boxy border) ---------- */
[data-testid="stMetric"] {{
  background: linear-gradient(180deg, #121A31 0%, #0D1424 100%);
  border: 1px solid var(--line-soft); border-top: 1px solid var(--gold-line);
  border-radius: 3px; padding: .9rem 1rem .85rem;
}}
[data-testid="stMetric"]:hover {{ border-top-color: var(--gold); }}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
  text-transform: uppercase !important; letter-spacing: .1em !important;
  font-size: .68rem !important; color: var(--gold-text) !important;
  font-weight: 500; font-family: var(--mono);
}}
[data-testid="stMetricValue"] {{
  font-weight: 500; color: var(--text); -webkit-text-fill-color: var(--text);
  font-size: 1.62rem !important; letter-spacing: -0.01em; overflow-wrap: anywhere;
}}
[data-testid="stMetricDelta"] {{ font-weight: 500; }}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {{
  gap: 1.3rem; padding: 0; background: transparent; border: none;
  border-bottom: 1px solid var(--line); border-radius: 0;
}}
.stTabs [data-baseweb="tab"] {{
  height: auto; padding: .5rem .1rem; border-radius: 0; border: none;
  background: transparent; color: var(--text-dim); font-weight: 500;
  font-family: var(--display); font-size: .98rem;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--gold-text); background: transparent; }}
.stTabs [aria-selected="true"] {{
  color: var(--gold-bright) !important; background: transparent !important;
  box-shadow: inset 0 -2px 0 var(--gold);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
  background: transparent !important;
}}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 4px; font-weight: 600; letter-spacing: .01em; font-family: var(--sans);
  border: 1px solid var(--gold-line); background: var(--panel); color: var(--gold-text);
  box-shadow: none; transition: border-color .12s ease, color .12s ease, background .12s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
  border-color: var(--gold); color: var(--gold-bright); background: var(--gold-wash);
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{
  background: linear-gradient(180deg, var(--gold-bright) 0%, var(--gold) 60%, var(--gold-deep) 100%);
  color: #1A1503; border-color: var(--gold-deep); text-shadow: none;
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  filter: brightness(1.06); color: #1A1503;
}}

/* ---------- sidebar (utility drawer, collapsed) ---------- */
[data-testid="stSidebar"] [role="radiogroup"] > label {{ padding: .4rem .55rem; border-radius: 4px; }}
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
[data-testid="stSliderThumbValue"] {{
  color: var(--gold-bright) !important; font-family: var(--mono); font-variant-numeric: tabular-nums;
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
[data-testid="stWidgetLabel"] p, label p {{ color: var(--gold-text) !important; font-weight: 500; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stDateInput"] input {{
  border-radius: 4px !important; border-color: var(--line) !important;
  background: var(--sunk) !important; color: var(--text) !important; font-family: var(--mono);
}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
  border-color: var(--gold) !important;
}}
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
  margin: 0 !important; left: auto !important; right: auto !important;
  transform: none !important; width: 100% !important;
}}

/* ---------- alerts / panels ---------- */
[data-testid="stAlert"], [data-testid="stNotification"],
[data-testid="stAlert"] [data-baseweb="notification"], [data-testid="stAlertContainer"],
[data-testid="stNotificationContentInfo"], [data-testid="stNotificationContentWarning"],
[data-testid="stNotificationContentError"], [data-testid="stNotificationContentSuccess"] {{
  border-radius: 4px; background: var(--panel) !important;
}}
[data-testid="stAlert"] {{ border: 1px solid var(--line); border-inline-start: 3px solid var(--gold); }}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p, [data-testid="stAlertContainer"] p {{
  color: var(--text) !important;
}}
[data-testid="stExpander"] {{
  border-radius: 4px; border: 1px solid var(--line); background: #0E1526; overflow: hidden;
}}
[data-testid="stExpander"] summary:hover {{ background: var(--gold-wash); }}
[data-testid="stForm"] {{ border-radius: 5px; border: 1px solid var(--line); background: #0E1526; }}
/* bordered container -> a raised card off the plate */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 5px;
  background: linear-gradient(180deg, #121A31 0%, #0D1424 100%);
}}
/* st.metric built-in sparkline: sit it on the card, not on a black rectangle */
[data-testid="stMetricChart"], [data-testid="stMetricChart"] canvas,
[data-testid="stMetricChart"] svg {{ background: transparent !important; }}
[data-testid="stMetricChart"] {{ opacity: .9; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border-radius: 4px; overflow: hidden; border: 1px solid var(--line);
}}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], .stPlotlyChart {{ border-radius: 4px; }}
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
}}
@media (max-width: 560px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding: .7rem .6rem 3rem !important; }}
  [data-testid="stColumn"] {{ flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }}
  .cx-hero-title {{ font-size: 1.35rem !important; line-height: 1.2; }}
  h1 {{ font-size: 1.4rem !important; }}  h2 {{ font-size: 1.22rem !important; }}  h3 {{ font-size: 1.05rem !important; }}
  [data-testid="stMetric"] {{ padding: .65rem .75rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.32rem !important; }}
  [data-testid="stMetricLabel"] p {{ white-space: normal; }}
  iframe {{ width: 100% !important; }}
  .stTabs [data-baseweb="tab"] {{ padding: .45rem .7rem; font-size: .88rem; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: .001ms !important; transition-duration: .001ms !important; }}
}}

/* ======================================================================= *
 *  scroll choreography  --  scroll is the timeline (scroll-craft principle)
 *  Driven by _SCROLL_JS below.  Default state is the FINISHED state, so if
 *  the script never runs (or reduced-motion), everything shows normally.
 *  The first page-load gets a reveal cascade; reruns during use do not, so
 *  dragging a slider never re-animates the page.
 * ======================================================================= */
/* scroll-progress meter -- a body pseudo-element, never in the content flow */
body::after {{
  content: ""; position: fixed; top: 0; left: 0; right: 0; height: 2px;
  z-index: 2147483000; pointer-events: none; opacity: .92;
  background: linear-gradient(90deg, var(--gold-deep) 0%, var(--gold) 55%, var(--gold-bright) 100%);
  transform-origin: 0 50%; transform: scaleX(var(--cx-progress, 0));
  transition: transform .12s linear;
}}
html.cx-js .stApp::before {{
  transform: translate3d(0, var(--cx-plate, 0px), 0) scale(1.06);
  transition: transform .18s linear; will-change: transform;
}}

@media (prefers-reduced-motion: no-preference) {{
  html.cx-js [data-testid="stMain"] .block-container [data-testid="stElementContainer"].cx-hide,
  html.cx-js [data-testid="stMain"] .block-container [data-testid="stHorizontalBlock"].cx-hide {{
    opacity: 0; transform: translateY(20px);
  }}
  html.cx-js .cx-hero.cx-hide {{ opacity: 0; transform: translateY(24px); }}
  html.cx-js .cx-hero.cx-hide .cx-hero-title {{ letter-spacing: .03em; filter: blur(2px); }}
  html.cx-js .cx-hide {{
    transition: opacity .62s cubic-bezier(.16,.72,.24,1),
                transform .62s cubic-bezier(.16,.72,.24,1),
                letter-spacing .62s ease, filter .62s ease;
  }}
  html.cx-js .cx-hide .cx-hero-title {{
    transition: letter-spacing .7s ease, filter .7s ease;
  }}
  html.cx-js .cx-rev {{ opacity: 1 !important; transform: none !important; }}
  html.cx-js .cx-rev .cx-hero-title {{ letter-spacing: -0.01em; filter: none; }}
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
    """Inject the global skin.  Call once, immediately after set_page_config.
    The scroll engine is installed separately by ``scroll_boot`` (index.html),
    because Streamlit's ``st.html`` strips a raw <script> even with the flag."""
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
            f'<span style="font-family:{_DISPLAY};font-weight:600;font-size:1.3rem;'
            f'color:{GOLD_BRIGHT};letter-spacing:.02em">CARN-X</span>'
        )


def refresh_button() -> None:
    """A small GitHub-green reload control (clears the data caches and reruns)."""
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
            f'<div class="cx-mast">{_wordmark_svg()}'
            f'<span class="cx-sub">MSTR · Bitcoin — probabilistic instrument</span></div>',
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
    """Drop-in for ``st.header``: a ruled chapter opening in gilt cut-serif."""
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
                "title": {"color": GOLD_TEXT, "fontSize": 14, "font": _DISPLAY, "fontWeight": 600},
                "axis": {
                    "domainColor": "rgba(160,175,220,0.20)",
                    "gridColor": "rgba(160,175,220,0.07)",
                    "tickColor": "rgba(160,175,220,0.20)",
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
