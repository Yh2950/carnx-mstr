"""
CARN-X  --  presentation layer  (visual only; no logic, no data, no decisions)
=============================================================================
"Aurora glass": a near-black ground under a live indigo->violet->cyan colour
mesh that drifts and shifts hue forever; frosted translucent panels over it;
one electric-violet accent with a soft glow.  Display type Space Grotesk,
working type Inter, Hebrew Rubik, numerals IBM Plex Mono.

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
# palette  --  "aurora glass": near-black under a live indigo->violet->cyan
# mesh, frosted translucent panels, one electric-violet accent.
# (var NAMES kept as GOLD*/INK* so every downstream rule keeps working -- the
#  values are the new palette.)
# --------------------------------------------------------------------------- #
# "light glass": a soft pale-grey ground under a faint pastel wash; frosted
# WHITE panels; one indigo accent; the Bitcoin roundel at the nav hub.
# "liquid glass" -- deep electric ground, near-clear glass that refracts it,
# bright specular rims.  (var names still INK*/GOLD* -- values only.)
INK = "#0B0A1E"          # deep indigo-black ground
INK_EDGE = "#060512"     # deepest
PANEL = "rgba(255,255,255,0.085)"  # a thin sheet of glass
PANEL_HI = "rgba(255,255,255,0.14)"
SUNK = "rgba(8,6,22,0.42)"         # sunk well / code
LINE = "rgba(255,255,255,0.16)"    # the lit glass rim
LINE_SOFT = "rgba(255,255,255,0.08)"

GOLD = "#9B8CFF"         # the accent  (electric violet)
GOLD_BRIGHT = "#C9BEFF"  # bright accent / specular
GOLD_DEEP = "#6A54E0"    # deep accent
GOLD_TEXT = "#D6CCFF"    # accent as text on dark

TEXT = "#F2F1FA"
TEXT_DIM = "#A9ACCB"
TEXT_FAINT = "#6C6F92"

UP = "#3DDC97"
DOWN = "#FF6B7A"

CYAN = "#4DE3F0"
VIOLET = "#B08CFF"
MAGENTA = "#FF6EC7"
LIME = "#9BE86B"
AMBER = "#F7931A"        # Bitcoin orange -- the hub logo only
CATEGORICAL = [GOLD, CYAN, UP, MAGENTA, VIOLET, "#6C8CFF", "#37D6E8", DOWN]

# editorial: a dramatic high-contrast serif for the cinematic display type,
# a clean grotesque for working UI, a mono for figures.
_DISPLAY = "'Playfair Display', 'Frank Ruhl Libre', 'Fraunces', Georgia, 'Times New Roman', serif"
_HEBREW_DISPLAY = "'Frank Ruhl Libre', 'Playfair Display', Georgia, serif"
_SANS = "'Inter', 'Assistant', system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
_MONO = "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace"

_FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Playfair+Display:ital,wght@0,500;0,600;0,700;0,800;0,900;1,600"
    "&family=Frank+Ruhl+Libre:wght@500;700;900"
    "&family=Inter:wght@400;500;600;700;800"
    "&family=Assistant:wght@400;500;600;700"
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
    return f"""
<style>
@import url('{_FONTS}');

@property --cx-sec-i {{ syntax: "<number>"; inherits: true; initial-value: 0; }}
@property --spin {{ syntax: "<angle>"; inherits: true; initial-value: 0deg; }}
@property --cx-aura-h {{ syntax: "<angle>"; inherits: true; initial-value: 0deg; }}

:root {{
  --ink:{INK}; --ink-edge:{INK_EDGE}; --panel:{PANEL}; --panel-hi:{PANEL_HI};
  --sunk:{SUNK}; --line:{LINE}; --line-soft:{LINE_SOFT};
  --gold:{GOLD}; --gold-bright:{GOLD_BRIGHT}; --gold-deep:{GOLD_DEEP}; --gold-text:{GOLD_TEXT};
  --gold-wash:rgba(91,84,232,0.10); --gold-line:rgba(91,84,232,0.28);
  --glow:0 0 0 1px rgba(91,84,232,.28), 0 0 24px -4px rgba(91,84,232,.35);
  --text:{TEXT}; --text-dim:{TEXT_DIM}; --text-faint:{TEXT_FAINT};
  --up:{UP}; --down:{DOWN};
  --display:{_DISPLAY}; --sans:{_SANS}; --mono:{_MONO};
  --cx-sec-i: 0;
  transition: --cx-sec-i .9s cubic-bezier(.16,.72,.24,1);
}}

/* ---------- aurora ground ---------- *
 * back -> front:  the live colour mesh (body::before)
 *                 a finer mesh that leans with the pointer (.stApp::before)
 *                 grain + vignette (.stApp::after)
 *                 the section-change sweep + flash (.cx-wipe / .cx-flash)       */
html, body, [class*="stApp"] {{ font-family: var(--sans); }}
html, body {{ background: {INK}; }}
.stApp {{ background: transparent; color: var(--text); }}

body::before {{
  content: ""; position: fixed; inset: -16vmax; z-index: 0; pointer-events: none;
  background:
    radial-gradient(52vmax 48vmax at calc(16% + var(--cx-amx,0px)) calc(10% + var(--cx-amy,0px)),
      rgba(59,107,255,0.55) 0%, rgba(59,107,255,0) 56%),
    radial-gradient(58vmax 54vmax at calc(88% - var(--cx-amx,0px)) 16%,
      rgba(155,77,255,0.55) 0%, rgba(155,77,255,0) 54%),
    radial-gradient(64vmax 58vmax at 44% calc(104% + var(--cx-amy,0px)),
      rgba(45,224,240,0.40) 0%, rgba(45,224,240,0) 56%),
    radial-gradient(46vmax 42vmax at calc(78% + var(--cx-amx,0px)) 92%,
      rgba(255,61,166,0.42) 0%, rgba(255,61,166,0) 56%),
    radial-gradient(40vmax 36vmax at 4% 62%,
      rgba(108,140,255,0.34) 0%, rgba(108,140,255,0) 58%),
    radial-gradient(120vmax 110vmax at 50% 46%,
      rgba(11,10,30,0) 34%, rgba(6,5,18,0.72) 100%);
  filter: saturate(1.35) brightness(1.05) contrast(1.04);
  transition: filter 1.1s ease;
  animation: cx-aurora 32s ease-in-out infinite alternate, cx-aura-hue 44s linear infinite;
  will-change: transform, filter;
}}
.stApp::before {{
  content: ""; position: fixed; inset: -10vmax; z-index: 0; pointer-events: none;
  background:
    linear-gradient(118deg,
      rgba(200,220,255,0) 34%, rgba(200,220,255,0.16) 46%,
      rgba(255,255,255,0.32) 50%, rgba(200,220,255,0.10) 55%, rgba(200,220,255,0) 64%),
    radial-gradient(30vmax 26vmax at 66% 30%, rgba(120,180,255,0.28) 0%, rgba(120,180,255,0) 56%),
    radial-gradient(24vmax 22vmax at 26% 74%, rgba(255,110,199,0.20) 0%, rgba(255,110,199,0) 60%);
  transform: translate3d(calc(var(--cx-mx,0px) * 1.7), calc(var(--cx-plate,0px) + var(--cx-my,0px) * 1.7), 0)
             rotate(var(--cx-rot,0deg)) scale(calc(1.06 * var(--cx-kick,1)));
  filter: blur(3px);
  transition: transform .18s cubic-bezier(.2,.8,.2,1), filter .7s ease;
  will-change: transform, filter;
}}
html.cx-js.cx-kick .stApp::before {{ animation: cx-plate-kick .62s cubic-bezier(.2,.8,.2,1); }}
.stApp::after {{
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.55'/%3E%3C/svg%3E"),
    radial-gradient(150% 130% at 50% 0%, rgba(6,5,18,0) 46%, rgba(4,3,14,0.7) 100%);
  background-size: 150px 150px, cover;
}}
.stApp::after {{ opacity: .5; }}


/* section-change sweep -- an electric prism edge */
.cx-wipe {{
  position: fixed; inset: 0 -46vw; z-index: 6; pointer-events: none;
  opacity: 0; transform: translateX(-125%) skewX(-15deg);
  background: linear-gradient(90deg,
    rgba(155,140,255,0) 0%, rgba(77,227,240,0.22) 30%,
    rgba(255,255,255,0.7) 46%, rgba(255,255,255,0.95) 50%,
    rgba(255,255,255,0.7) 54%, rgba(255,110,199,0.22) 70%, rgba(155,140,255,0) 100%);
  box-shadow: 0 0 100px 16px rgba(155,140,255,0.35);
}}
.cx-wipe.run {{ animation: cx-wipe .8s cubic-bezier(.62,0,.28,1); }}
.cx-flash {{
  position: fixed; inset: 0; z-index: 5; pointer-events: none; opacity: 0;
  background: radial-gradient(120% 120% at 50% 30%,
    hsla(calc(244deg + var(--cx-hue,0deg)), 75%, 66%, 0.12) 0%,
    hsla(calc(244deg + var(--cx-hue,0deg)), 75%, 60%, 0) 62%);
}}
.cx-flash.run {{ animation: cx-flash .7s ease-out; }}

@keyframes cx-aurora {{
  0%   {{ transform: translate3d(0,0,0) scale(1) rotate(0deg); }}
  33%  {{ transform: translate3d(5vmax,-4vmax,0) scale(1.1) rotate(3deg); }}
  66%  {{ transform: translate3d(-3vmax,4vmax,0) scale(1.16) rotate(-2.6deg); }}
  100% {{ transform: translate3d(-5vmax,-2vmax,0) scale(1.08) rotate(1.8deg); }}
}}
@keyframes cx-aura-hue {{
  0% {{ --cx-aura-h: -20deg; }}  50% {{ --cx-aura-h: 22deg; }}  100% {{ --cx-aura-h: -20deg; }}
}}
@keyframes cx-wipe {{
  0%   {{ opacity: 0; transform: translateX(-125%) skewX(-15deg); }}
  18%  {{ opacity: 1; }}
  100% {{ opacity: 0; transform: translateX(125%) skewX(-15deg); }}
}}
@keyframes cx-flash {{
  0% {{ opacity: 0; }}  18% {{ opacity: 1; }}  100% {{ opacity: 0; }}
}}
@keyframes cx-plate-kick {{
  0%   {{ transform: translate3d(var(--cx-mx,0px), var(--cx-plate,0px), 0) rotate(var(--cx-rot,0deg)) scale(1.04); }}
  30%  {{ transform: translate3d(var(--cx-mx,0px), var(--cx-plate,0px), 0) rotate(calc(var(--cx-rot,0deg) - 3deg)) scale(1.1); }}
  100% {{ transform: translate3d(var(--cx-mx,0px), var(--cx-plate,0px), 0) rotate(var(--cx-rot,0deg)) scale(1.04); }}
}}

/* per-section: shift the whole plate's hue + the engraving's rest pose.
   --cx-sec-i (0..10) is set on <html> by scroll_boot from the active menu item. */
:root {{
  --cx-aura-h: 0deg;
  --cx-hue: calc((var(--cx-sec-i, 0) - 5) * 9deg);
  --cx-sat: calc(1 + (var(--cx-sec-i, 0) - 5) * 0.03);
  --cx-rot: calc((var(--cx-sec-i, 0) - 5) * 1.1deg);
  --cx-scale: calc(1.06 + var(--cx-sec-i, 0) * 0.009);
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
[data-testid="stSidebar"] {{ background: rgba(8,8,18,0.75); backdrop-filter: blur(20px); border-right: 1px solid var(--line-soft); }}

/* ---------- masthead ---------- */
.cx-mast {{
  display: flex; align-items: center; gap: .7rem;
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
  padding: .38rem .95rem !important; border-radius: 10px !important;
  color: #fff !important; -webkit-text-fill-color: #fff !important;
  background: linear-gradient(180deg, rgba(61,220,151,0.9), rgba(38,180,120,0.85)) !important;
  -webkit-backdrop-filter: blur(14px) saturate(1.6); backdrop-filter: blur(14px) saturate(1.6);
  border: 1px solid rgba(180,255,220,0.4) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 10px 30px -6px rgba(61,220,151,0.5) !important;
  transition: background .12s ease, transform .06s ease !important;
}}
.st-key-cx_refresh button:hover, .cx-reload button:hover {{
  background: linear-gradient(180deg, rgba(77,235,166,0.95), rgba(46,195,132,0.9)) !important; border-color: rgba(180,255,220,0.55) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 0 36px -6px rgba(61,220,151,0.7) !important;
  color: #fff !important; -webkit-text-fill-color: #fff !important;
}}
.st-key-cx_refresh button:active, .cx-reload button:active {{
  background: #2c974b !important; transform: translateY(.5px);
}}
.st-key-cx_refresh button:focus-visible, .cx-reload button:focus-visible {{
  outline: 2px solid rgba(46,160,67,.55) !important; outline-offset: 2px;
}}
.st-key-cx_refresh button p, .cx-reload button p {{ color: #fff !important; }}

/* ---------- orbital navigation ---------- *
 * page buttons ride a ring around the Bitcoin hub; the ring spins on hover /
 * touch, and drag-scrubs to a page.  Built by scroll_boot from the keyed radio,
 * which stays in the DOM (hidden) so the JS can click it and AppTest can read it. */
.st-key-cx_nav {{
  position: absolute !important; left: -9999px !important; top: 0 !important;
  width: 1px !important; height: 0 !important; overflow: hidden !important;
  opacity: 0 !important; pointer-events: none !important; margin: 0 !important; border: 0 !important;
}}

.cx-orbit {{
  --r: 202px; --size: 466px; --orbit-rot: 0deg; --spin: 0deg;
  position: relative; width: var(--size); height: var(--size);
  margin: .4rem auto 3.6rem; z-index: 20;
  touch-action: none; -webkit-user-select: none; user-select: none;
}}
/* the glass ring band */
.cx-orbit::before {{
  content: ""; position: absolute; inset: 34px; border-radius: 50%;
  background: rgba(255,255,255,0.05);
  -webkit-backdrop-filter: blur(14px) saturate(1.5); backdrop-filter: blur(14px) saturate(1.5);
  border: 1px solid rgba(255,255,255,0.14);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.35), inset 0 0 60px -20px rgba(155,140,255,0.5),
              0 30px 80px -30px rgba(0,0,0,0.6);
  -webkit-mask: radial-gradient(circle, transparent calc(50% - 34px), #000 calc(50% - 33px));
          mask: radial-gradient(circle, transparent calc(50% - 34px), #000 calc(50% - 33px));
}}
/* a rotating specular sweep + graduation ticks on the band */
.cx-orbit::after {{
  content: ""; position: absolute; inset: 30px; border-radius: 50%; pointer-events: none;
  background:
    conic-gradient(from calc(var(--orbit-rot) + var(--spin)),
      rgba(255,255,255,0) 0deg, rgba(255,255,255,0.28) 24deg, rgba(255,255,255,0) 70deg,
      rgba(255,255,255,0) 190deg, rgba(200,220,255,0.16) 210deg, rgba(255,255,255,0) 250deg),
    repeating-conic-gradient(from 0deg,
      rgba(255,255,255,0.34) 0deg 0.5deg, rgba(255,255,255,0) 0.5deg 6deg);
  -webkit-mask: radial-gradient(circle, transparent calc(50% - 38px), #000 calc(50% - 36px), #000 calc(50% - 4px), transparent calc(50% - 2px));
          mask: radial-gradient(circle, transparent calc(50% - 38px), #000 calc(50% - 36px), #000 calc(50% - 4px), transparent calc(50% - 2px));
}}
.cx-orbit-ring {{
  position: absolute; inset: 0; border-radius: 50%;
  transform: rotate(calc(var(--orbit-rot) + var(--spin)));
  transition: transform .6s cubic-bezier(.2,.85,.25,1);
}}
.cx-orbit.dragging .cx-orbit-ring, .cx-orbit.spinning .cx-orbit-ring {{ transition: none; }}

.cx-orbit-item {{
  position: absolute; left: 50%; top: 50%; width: 0; height: 0;
  transform: rotate(var(--a)) translateY(calc(-1 * var(--r)));
}}
.cx-orbit-item > button {{
  position: absolute; left: 50%; top: 50%;
  transform: translate(-50%,-50%) rotate(calc(-1 * var(--a) - var(--orbit-rot) - var(--spin)));
  white-space: nowrap; cursor: pointer; line-height: 1;
  font-family: var(--display); font-weight: 600; font-size: .76rem; letter-spacing: .005em;
  color: var(--text-dim);
  background: rgba(255,255,255,0.08);
  -webkit-backdrop-filter: blur(20px) saturate(1.7); backdrop-filter: blur(20px) saturate(1.7);
  border: 1px solid rgba(255,255,255,0.16); border-radius: 999px; padding: .4rem .78rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.45), inset 0 0 16px -6px rgba(255,255,255,0.2),
              0 12px 30px -14px rgba(0,0,0,0.5);
  transition: color .18s, background .18s, box-shadow .18s, border-color .18s, transform .3s cubic-bezier(.2,.85,.25,1);
}}
.cx-orbit.spinning .cx-orbit-item > button {{ transition: color .18s, background .18s, box-shadow .18s, border-color .18s; }}
.cx-orbit-item > button:hover {{
  color: var(--text); border-color: rgba(201,190,255,0.5); background: rgba(255,255,255,0.14);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.6), 0 0 26px -6px rgba(155,140,255,0.6), 0 14px 32px -14px rgba(0,0,0,0.5);
}}
.cx-orbit-item.on > button {{
  color: #fff; -webkit-text-fill-color: #fff; font-weight: 700;
  background: linear-gradient(135deg, rgba(201,190,255,0.5), rgba(155,140,255,0.35));
  border-color: rgba(201,190,255,0.7);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.7), inset 0 0 20px -4px rgba(255,255,255,0.4),
              0 0 40px -4px rgba(155,140,255,0.8), 0 16px 40px -12px rgba(0,0,0,0.55);
  transform: translate(-50%,-50%) rotate(calc(-1 * var(--a) - var(--orbit-rot) - var(--spin))) scale(1.16);
  z-index: 4;
}}

.cx-orbit-hub {{
  position: absolute; left: 50%; top: 50%; width: 128px; height: 128px;
  transform: translate(-50%,-50%); border-radius: 50%; overflow: visible; cursor: grab; z-index: 5;
}}
.cx-orbit-hub:active {{ cursor: grabbing; }}
.cx-orbit-hub svg, .cx-orbit-hub img {{
  width: 100%; height: 100%; display: block; border-radius: 50%; position: relative; z-index: 2;
  box-shadow: 0 24px 60px -12px rgba(155,140,255,.6), 0 4px 16px rgba(0,0,0,.35),
              inset 0 0 0 2px rgba(255,255,255,.35), inset 0 3px 10px rgba(255,255,255,.25);
}}
.cx-orbit-hub::before {{  /* soft always-on aura */
  content: ""; position: absolute; inset: -18px; border-radius: 50%; z-index: 0;
  background: radial-gradient(circle, rgba(155,140,255,.5) 0%, rgba(77,227,240,.2) 45%, rgba(155,140,255,0) 72%);
  filter: blur(6px); animation: cx-hub-pulse 4.5s ease-in-out infinite;
}}
.cx-orbit-hub::after {{  /* rotating conic ring */
  content: ""; position: absolute; inset: -12px; border-radius: 50%; z-index: 1;
  background: conic-gradient(from 0deg, rgba(155,140,255,0), rgba(200,220,255,.85), rgba(77,227,240,.4), rgba(255,110,199,.3), rgba(155,140,255,0) 70%);
  opacity: .5; transition: opacity .3s; animation: cx-orbit-spin 8s linear infinite;
  -webkit-mask: radial-gradient(circle, transparent calc(50% - 5px), #000 calc(50% - 3px));
          mask: radial-gradient(circle, transparent calc(50% - 5px), #000 calc(50% - 3px));
}}
.cx-orbit.spinning {{ animation: cx-spin-var 22s linear infinite; }}
.cx-orbit.spinning .cx-orbit-hub::after {{ opacity: 1; animation: cx-orbit-spin 2.4s linear infinite; }}

.cx-orbit-name {{
  position: absolute; left: 50%; bottom: -20px; transform: translateX(-50%);
  font-family: var(--display); font-weight: 700; font-size: 1.05rem; color: #fff;
  white-space: nowrap; letter-spacing: -.014em;
  background: rgba(255,255,255,0.10);
  -webkit-backdrop-filter: blur(22px) saturate(1.7); backdrop-filter: blur(22px) saturate(1.7);
  border: 1px solid rgba(255,255,255,0.2); border-radius: 999px; padding: .34rem 1.15rem; z-index: 6;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.5), 0 0 34px -6px rgba(155,140,255,0.5), 0 14px 40px -14px rgba(0,0,0,0.55);
}}
.cx-orbit-hint {{
  position: absolute; left: 50%; top: 100%; transform: translateX(-50%); margin-top: 2.3rem;
  font-family: var(--mono); font-size: .56rem; letter-spacing: .2em; text-transform: uppercase;
  color: var(--text-faint); white-space: nowrap; transition: opacity .4s;
}}
.cx-orbit.touched .cx-orbit-hint {{ opacity: 0; }}

@keyframes cx-hub-pulse {{
  0%,100% {{ transform: scale(1); opacity: .8; }}
  50% {{ transform: scale(1.12); opacity: 1; }}
}}
@keyframes cx-orbit-spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes cx-spin-var {{ from {{ --spin: 0deg; }} to {{ --spin: 360deg; }} }}

/* ---------- typography (cinematic editorial) ---------- */
h1, h2 {{ font-family: 'Playfair Display', 'Frank Ruhl Libre', 'Fraunces', Georgia, serif !important; }}
:lang(he), [dir="rtl"] {{ }}
.cx-hero-title:dir(rtl), h1:dir(rtl), h2:dir(rtl), h3:dir(rtl) {{
  font-family: 'Frank Ruhl Libre', 'Playfair Display', Georgia, serif !important; }}
h1, h2, h3, h4, h5 {{
  font-family: var(--display); letter-spacing: -0.01em; text-wrap: balance; font-weight: 700;
}}
h1, h2 {{ color: var(--text); font-weight: 800; }}
h3, h4, h5 {{ color: var(--text); font-weight: 700; letter-spacing: -0.005em; }}
h1 {{ font-size: clamp(2.1rem, 3vw, 3rem); line-height: 1.02; }}
h2 {{ font-size: clamp(1.7rem, 2.8vw, 2.6rem); line-height: 1.06; }}
h3 {{ font-size: 1.28rem; }}  h4 {{ font-size: 1.06rem; font-family: var(--sans); font-weight: 700; }}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li {{
  color: #C7C9E0; line-height: 1.66; text-wrap: pretty; font-family: var(--sans);
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
  border: 1px solid var(--gold-line); border-radius: 6px; padding: .05em .38em; font-size: .84em;
}}
pre, [data-testid="stCode"] {{
  background: rgba(255,255,255,0.7) !important; -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
  border: 1px solid var(--line); border-radius: 12px;
}}
.katex {{ color: var(--text); }}

/* ---------- the chapter opening -- a title card from a launch film ---------- */
.cx-hero {{
  position: relative; margin: 1.4rem 0 2.4rem; padding: 2.6rem 0 1.2rem;
  overflow: visible; isolation: isolate;
}}
.cx-hero-ghost {{
  position: absolute; left: -0.06em; top: -0.06em; z-index: 0; pointer-events: none;
  font-family: var(--display); font-weight: 900; text-transform: uppercase;
  font-size: clamp(4.5rem, 13vw, 10rem); line-height: .8; letter-spacing: -0.04em;
  white-space: nowrap; color: transparent;
  -webkit-text-stroke: 1px rgba(255,255,255,0.05);
  background: linear-gradient(180deg, rgba(155,140,255,0.10), rgba(255,255,255,0.0) 70%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-mask: linear-gradient(90deg, #000 0 62%, transparent 92%);
          mask: linear-gradient(90deg, #000 0 62%, transparent 92%);
}}
.cx-hero-eyebrow {{
  position: relative; z-index: 2;
  display: inline-flex; align-items: center; gap: .4rem; margin: 0 0 .7rem;
  font-family: var(--mono); font-size: .62rem; letter-spacing: .28em;
  text-transform: uppercase; color: var(--gold-text);
}}
.cx-hero-eyebrow::before {{
  content: ""; width: 22px; height: 1px; background: var(--gold);
  box-shadow: 0 0 8px var(--gold);
}}
.cx-hero-title {{
  position: relative; z-index: 2; display: block;
  font-family: var(--display) !important; font-weight: 700; color: #FBFBFF;
  font-size: clamp(2.6rem, 5.2vw, 4.8rem) !important; line-height: 1.0 !important; letter-spacing: -0.028em;
  text-shadow: 0 2px 50px rgba(155,140,255,0.4);
}}
.cx-hero-title::before, .cx-hero-title::after {{
  content: attr(data-text); position: absolute; inset: 0; z-index: -1;
  pointer-events: none; opacity: .32; mix-blend-mode: screen;
}}
.cx-hero-title::before {{ color: #37D6E8; transform: translate3d(-2px,0,0); }}
.cx-hero-title::after  {{ color: #FF4FB0; transform: translate3d(2px,1px,0); }}
.cx-hero-title.cx-glitch::before {{ animation: cx-rgb-l 5.5s steps(1) infinite; }}
.cx-hero-title.cx-glitch::after  {{ animation: cx-rgb-r 5.5s steps(1) infinite; }}
.cx-hero-title.cx-glitch {{ animation: cx-glitch-slice 5.5s steps(1) infinite; }}
.cx-hero-sub {{
  position: relative; z-index: 2; margin-top: 1rem; color: var(--text-dim);
  font-size: .96rem; max-width: 64ch; text-wrap: pretty; font-family: var(--sans);
  padding-left: 26px; border-left: 1px solid rgba(255,255,255,0.14);
}}
@keyframes cx-rgb-l {{
  0%,88%,100% {{ transform: translate3d(-2px,0,0); }}
  90% {{ transform: translate3d(-7px,-2px,0); }}  93% {{ transform: translate3d(3px,1px,0); }}
  96% {{ transform: translate3d(-4px,2px,0); }}
}}
@keyframes cx-rgb-r {{
  0%,88%,100% {{ transform: translate3d(2px,1px,0); }}
  90% {{ transform: translate3d(7px,2px,0); }}   93% {{ transform: translate3d(-3px,-1px,0); }}
  96% {{ transform: translate3d(5px,-2px,0); }}
}}
@keyframes cx-glitch-slice {{
  0%,89%,100% {{ clip-path: none; }}
  90% {{ clip-path: inset(18% 0 62% 0); }}  92% {{ clip-path: inset(72% 0 8% 0); }}
  94% {{ clip-path: inset(42% 0 38% 0); }}  96% {{ clip-path: none; }}
}}

/* ---------- metric  (an engraved plaque, no boxy border) ---------- */
[data-testid="stMetric"] {{
  background: var(--panel); -webkit-backdrop-filter: blur(30px) saturate(1.28);
  backdrop-filter: blur(30px) saturate(1.28);
  border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.15rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.45), inset 0 0 30px -8px rgba(255,255,255,0.14), inset 0 -24px 44px -34px rgba(155,140,255,0.5), 0 24px 60px -22px rgba(0,0,0,0.55);
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}}
[data-testid="stMetric"]:hover {{
  background: rgba(255,255,255,0.50);
  box-shadow: inset 0 1px 0 rgba(255,255,255,1), inset 0 0 0 1px rgba(91,84,232,0.16), 0 0 24px -8px rgba(91,84,232,0.26), 0 18px 48px -18px rgba(0,0,0,0.26);
  transform: translateY(-2px);
}}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{
  text-transform: uppercase !important; letter-spacing: .09em !important;
  font-size: .66rem !important; color: var(--text-dim) !important;
  font-weight: 600; font-family: var(--mono);
}}
[data-testid="stMetricValue"] {{
  font-weight: 600; color: var(--text); -webkit-text-fill-color: var(--text);
  font-family: var(--display);
  font-size: 1.7rem !important; letter-spacing: -0.02em; overflow-wrap: anywhere;
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
  color: var(--gold-text) !important; background: transparent !important;
  box-shadow: inset 0 -2px 0 var(--gold);
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
  background: transparent !important;
}}

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 11px; font-weight: 600; letter-spacing: .005em; font-family: var(--sans);
  border: 1px solid var(--gold-line); background: var(--panel); color: var(--gold-text);
  -webkit-backdrop-filter: blur(12px); backdrop-filter: blur(12px);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
  transition: border-color .14s ease, color .14s ease, background .14s ease, box-shadow .14s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
  border-color: var(--gold); color: var(--gold-deep); background: var(--gold-wash);
  box-shadow: 0 0 22px -6px rgba(91,84,232,0.4), inset 0 1px 0 rgba(255,255,255,0.8);
}}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button {{
  background: linear-gradient(135deg, #9E8CFF 0%, #7B67EC 55%, #5F4BD6 100%);
  color: #fff; -webkit-text-fill-color: #fff; border-color: rgba(191,169,255,0.5);
  box-shadow: 0 8px 28px -6px rgba(123,103,236,0.65), inset 0 1px 0 rgba(255,255,255,0.22);
}}
.stButton > button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] > button:hover {{
  filter: brightness(1.08); color: #fff;
  box-shadow: 0 10px 36px -6px rgba(123,103,236,0.85), inset 0 1px 0 rgba(255,255,255,0.28);
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
[data-testid="stWidgetLabel"] p, label p {{ color: var(--text-dim) !important; font-weight: 600; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"],
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stDateInput"] input {{
  border-radius: 10px !important; border-color: var(--line) !important;
  background: rgba(255,255,255,0.7) !important; color: var(--text) !important; font-family: var(--mono);
}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
  border-color: var(--gold) !important;
  box-shadow: 0 0 0 3px var(--gold-wash), 0 0 22px -6px rgba(142,123,240,0.6) !important;
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
  border-radius: 14px; background: var(--panel) !important;
  -webkit-backdrop-filter: blur(24px) saturate(1.2); backdrop-filter: blur(24px) saturate(1.2);
}}
[data-testid="stAlert"] {{ border: 1px solid var(--line); border-inline-start: 3px solid var(--gold); }}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p, [data-testid="stAlertContainer"] p {{
  color: var(--text) !important;
}}
[data-testid="stExpander"] {{
  border-radius: 16px; border: 1px solid var(--line); overflow: hidden;
  background: var(--panel); -webkit-backdrop-filter: blur(28px) saturate(1.25); backdrop-filter: blur(28px) saturate(1.25);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.45), inset 0 0 30px -8px rgba(255,255,255,0.14), inset 0 -24px 44px -34px rgba(155,140,255,0.5), 0 24px 60px -22px rgba(0,0,0,0.55);
}}
[data-testid="stExpander"] summary:hover {{ background: var(--gold-wash); }}
[data-testid="stForm"] {{
  border-radius: 18px; border: 1px solid var(--line);
  background: var(--panel); -webkit-backdrop-filter: blur(32px) saturate(1.25); backdrop-filter: blur(32px) saturate(1.25);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.45), inset 0 0 30px -8px rgba(255,255,255,0.14), inset 0 -24px 44px -34px rgba(155,140,255,0.5), 0 24px 60px -22px rgba(0,0,0,0.55);
}}
/* bordered container -> a barely-there frame so the inner card carries the weight */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 20px;
  background: rgba(255,255,255,0.16); -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.4);
}}
/* st.metric built-in sparkline: sit it on the card, not on a black rectangle */
[data-testid="stMetricChart"], [data-testid="stMetricChart"] canvas,
[data-testid="stMetricChart"] svg {{ background: transparent !important; }}
[data-testid="stMetricChart"] {{ opacity: .9; }}
[data-testid="stDataFrame"], [data-testid="stTable"] {{
  border-radius: 14px; overflow: hidden; border: 1px solid var(--line);
}}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"], .stPlotlyChart {{ border-radius: 14px; }}
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
/* ---------- orbital nav: responsive ---------- */
@media (max-width: 900px) {{
  .cx-orbit {{ --r: 150px; --size: 340px; margin: .2rem auto 3rem; }}
  .cx-orbit-item > button {{ font-size: .64rem; padding: .3rem .58rem; }}
  .cx-orbit-hub {{ width: 92px; height: 92px; }}
  .cx-orbit-name {{ font-size: .9rem; }}
}}
@media (max-width: 560px) {{
  .cx-orbit {{ --r: 124px; --size: 286px; }}
  .cx-orbit-item > button {{ font-size: .58rem; padding: .24rem .48rem; }}
  .cx-orbit-hub {{ width: 78px; height: 78px; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .cx-orbit.spinning, .cx-orbit.spinning .cx-orbit-hub::after {{ animation: none !important; }}
}}

@media (max-width: 560px) {{
  [data-testid="stMainBlockContainer"], .block-container {{ padding: .7rem .6rem 3rem !important; }}
  [data-testid="stColumn"] {{ flex: 1 1 100% !important; min-width: 100% !important; width: 100% !important; }}
  .cx-hero {{ padding: 1.8rem 0 .9rem; margin: .8rem 0 1.6rem; }}
  .cx-hero-title {{ font-size: 2rem !important; line-height: 1.04; }}
  .cx-hero-ghost {{ font-size: 4.4rem !important; }}
  h1 {{ font-size: 1.7rem !important; }}  h2 {{ font-size: 1.4rem !important; }}  h3 {{ font-size: 1.14rem !important; }}
  [data-testid="stMetric"] {{ padding: .65rem .75rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.32rem !important; }}
  [data-testid="stMetricLabel"] p {{ white-space: normal; }}
  iframe {{ width: 100% !important; }}
  .stTabs [data-baseweb="tab"] {{ padding: .45rem .7rem; font-size: .88rem; }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: .001ms !important; transition-duration: .001ms !important; }}
  .cx-hero-title::before, .cx-hero-title::after {{ display: none !important; }}
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
/* (the full plate transform lives on .stApp::before above; nothing to override) */

@media (prefers-reduced-motion: no-preference) {{
  html.cx-js [data-testid="stMain"] .block-container [data-testid="stElementContainer"].cx-hide,
  html.cx-js [data-testid="stMain"] .block-container [data-testid="stHorizontalBlock"].cx-hide {{
    opacity: 0; transform: translateY(34px) scale(.985); filter: blur(3px);
  }}
  html.cx-js .cx-hero.cx-hide {{ opacity: 0; transform: translateY(40px); filter: blur(4px); }}
  html.cx-js .cx-hero.cx-hide .cx-hero-title {{ letter-spacing: .03em; filter: blur(2px); }}
  html.cx-js .cx-hide {{
    transition: opacity .7s cubic-bezier(.16,.72,.24,1),
                transform .7s cubic-bezier(.16,.72,.24,1),
                letter-spacing .7s ease, filter .7s ease;
  }}
  html.cx-js .cx-rev {{ filter: blur(0) !important; }}
  html.cx-js .cx-hide .cx-hero-title {{
    transition: letter-spacing .7s ease, filter .7s ease;
  }}
  html.cx-js .cx-rev {{ opacity: 1 !important; transform: none !important; }}
  html.cx-js .cx-rev .cx-hero-title {{ letter-spacing: -0.01em; filter: none; }}
}}

/* ============================ Apple-style LIQUID GLASS ============================ *
 * a refracting glass sheet (::before, backdrop-blur + SVG displacement) + a
 * specular rim that catches light top-left (::after).  Added on top of the
 * existing panels -- nothing else about the design changes.                        */
[data-testid="stMetric"],
div[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stForm"],
[data-testid="stExpander"],
.cx-orbit-item > button,
.cx-orbit-name,
.st-key-cx_refresh button {{
  position: relative; overflow: hidden; isolation: isolate;
}}
[data-testid="stMetric"]::before,
div[data-testid="stVerticalBlockBorderWrapper"]::before,
[data-testid="stForm"]::before,
[data-testid="stExpander"]::before,
.cx-orbit-item > button::before,
.cx-orbit-name::before,
.st-key-cx_refresh button::before {{
  content: ""; position: absolute; inset: 0; z-index: -1; border-radius: inherit;
  -webkit-backdrop-filter: blur(2px) saturate(1.6) brightness(1.06);
          backdrop-filter: blur(2px) saturate(1.6) brightness(1.06);
  filter: url(#cx-lg);
}}
[data-testid="stMetric"]::after,
div[data-testid="stVerticalBlockBorderWrapper"]::after,
[data-testid="stForm"]::after,
[data-testid="stExpander"]::after,
.cx-orbit-item > button::after,
.cx-orbit-name::after,
.st-key-cx_refresh button::after {{
  content: ""; position: absolute; inset: 0; z-index: 1; border-radius: inherit;
  padding: 1.2px; pointer-events: none; mix-blend-mode: screen;
  background: linear-gradient(135deg,
    rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.25) 18%,
    rgba(255,255,255,0) 42%, rgba(255,255,255,0) 60%,
    rgba(200,220,255,0.30) 84%, rgba(255,255,255,0.55) 100%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
          mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;  mask-composite: exclude;
}}
/* a faint edge-lens brightening just inside the rim */
[data-testid="stMetric"],
div[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stForm"],
[data-testid="stExpander"] {{
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.45),
    inset 0 0 22px -4px rgba(255,255,255,0.18),
    inset 0 -30px 60px -46px rgba(155,140,255,0.6),
    0 26px 64px -24px rgba(0,0,0,0.6);
}}
.cx-orbit-name {{ overflow: visible; }}
.cx-orbit-name::before {{ overflow: hidden; }}
@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {{
  [data-testid="stMetric"]::before, div[data-testid="stVerticalBlockBorderWrapper"]::before,
  [data-testid="stForm"]::before, [data-testid="stExpander"]::before {{ background: rgba(20,18,42,0.7); }}
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


_LG_SVG = (
    '<svg aria-hidden="true" width="0" height="0" '
    'style="position:absolute;pointer-events:none">'
    '<defs><filter id="cx-lg" x="-25%" y="-25%" width="150%" height="150%" '
    'color-interpolation-filters="sRGB">'
    '<feTurbulence type="fractalNoise" baseFrequency="0.011 0.013" numOctaves="2" '
    'seed="11" result="n"/>'
    '<feGaussianBlur in="n" stdDeviation="1.3" result="nb"/>'
    '<feDisplacementMap in="SourceGraphic" in2="nb" scale="26" '
    'xChannelSelector="R" yChannelSelector="G"/>'
    '</filter></defs></svg>'
)


def inject_theme() -> None:
    """Inject the global skin.  Call once, immediately after set_page_config.
    The scroll engine is installed separately by ``scroll_boot`` (index.html),
    because Streamlit's ``st.html`` strips a raw <script> even with the flag."""
    st.markdown(_css(), unsafe_allow_html=True)
    st.markdown(_LG_SVG, unsafe_allow_html=True)
    try:
        st.html(_LTR_SHIM, unsafe_allow_javascript=True)
    except TypeError:
        pass
    _register_altair_theme()


def _wordmark_svg() -> str:
    return (
        '<img src="./carnx/mark.png" alt="" '
        'style="height:42px;width:42px;border-radius:11px;display:block;'
        'box-shadow:0 8px 22px -6px rgba(155,140,255,.5), inset 0 0 0 1px rgba(255,255,255,.35)">'
        f'<span style="font-family:{_DISPLAY};font-weight:700;font-size:1.24rem;'
        f'color:{TEXT};letter-spacing:-.01em">CARN-X</span>'
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
    """A title card from a launch film: a huge ghost word, a chromatic-glitch
    serif headline, one hairline sub-line.  Cosmetic only."""
    import html as _h
    ghost = _h.escape((eyebrow or "CARN-X").split()[0])
    t = _h.escape(title)
    parts = [
        '<div class="cx-hero">',
        f'<span class="cx-hero-ghost" aria-hidden="true">{ghost}</span>',
    ]
    parts.append(f'<h2 class="cx-hero-title cx-glitch" data-text="{t}">{t}</h2>')
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
                "title": {"color": GOLD_TEXT, "fontSize": 14, "font": _DISPLAY, "fontWeight": 600},
                "axis": {
                    "domainColor": "rgba(180,170,255,0.20)",
                    "gridColor": "rgba(180,170,255,0.07)",
                    "tickColor": "rgba(180,170,255,0.20)",
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
                    "heatmap": ["#12121F", "#4C3AA8", GOLD, CYAN],
                    "ramp": ["#12121F", "#5A49C6", GOLD, "#BCA9FF"],
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
