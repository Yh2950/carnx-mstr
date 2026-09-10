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
INK = "#0A0A0B"          # neutral near-black ground (no colour)
INK_EDGE = "#050505"     # deepest
PANEL = "rgba(255,255,255,0.085)"  # a thin sheet of glass
PANEL_HI = "rgba(255,255,255,0.14)"
SUNK = "rgba(6,6,8,0.42)"         # sunk well / code
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

# "electric ocean" backdrop -- a bright tropical-lagoon aqua the whole page
# floats on, with lightning.  Content rides a dark translucent slab so text and
# the section cards keep their contrast.
OCEAN_TOP = "#93ECFF"
OCEAN_MID = "#5CD8F1"
OCEAN_DEEP = "#31BFE3"
SLAB = "rgba(11,14,18,0.60)"      # the reading surface over the water

_BOLT_A = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 160 620'%3E"
    "%3Cdefs%3E%3Cfilter id='g' x='-90%25' y='-12%25' width='280%25' height='124%25'%3E"
    "%3CfeGaussianBlur stdDeviation='8'/%3E%3C/filter%3E%3C/defs%3E"
    "%3Cpath d='M112 0 L62 205 L104 198 L44 372 L90 360 L34 620' fill='none' "
    "stroke='%23d6f7ff' stroke-width='12' filter='url(%23g)' opacity='0.85'/%3E"
    "%3Cpath d='M112 0 L62 205 L104 198 L44 372 L90 360 L34 620' fill='none' "
    "stroke='%23ffffff' stroke-width='3' stroke-linejoin='round'/%3E%3C/svg%3E"
)
_BOLT_B = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 160 620'%3E"
    "%3Cdefs%3E%3Cfilter id='g' x='-90%25' y='-12%25' width='280%25' height='124%25'%3E"
    "%3CfeGaussianBlur stdDeviation='8'/%3E%3C/filter%3E%3C/defs%3E"
    "%3Cpath d='M52 0 L104 195 L64 202 L120 366 L74 356 L126 620' fill='none' "
    "stroke='%23d6f7ff' stroke-width='12' filter='url(%23g)' opacity='0.85'/%3E"
    "%3Cpath d='M52 0 L104 195 L64 202 L120 366 L74 356 L126 620' fill='none' "
    "stroke='%23ffffff' stroke-width='3' stroke-linejoin='round'/%3E%3C/svg%3E"
)

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
    "&family=Space+Grotesk:wght@400;500;600;700"
    "&family=Archivo+Black"
    "&family=Orbitron:wght@500;700;900"
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


# --------------------------------------------------------------------------- #
# per-section design languages
# --------------------------------------------------------------------------- #
# Each nav section speaks a different visual dialect.  scroll_boot writes
# data-cx-section="0..10" on <html> from the active menu item; every rule below
# is scoped to that + [data-testid=stMain], so the masthead, the orbital nav and
# the scroll chrome stay constant while the content area transforms.  If the JS
# never runs, no attribute is set and the base "glass" look shows -- nothing here
# is load-bearing.
#
#   0 סקירה .............. Minimalism      6 מבנים מתמטיים ...... Neo-Brutalism
#   1 טרמינל מסחר ........ Cyberpunk       7 מחזור ביטקוין ...... Maximalism
#   2 מחשבון הסתברויות ... Neumorphism     8 ראיות Walk-Forward . Material
#   3 אבחון סטטיסטי ...... Editorial       9 סיכון ומינוף ....... Claymorphism
#   4 תחזית הסתברותית .... Glassmorphism  10 הגדרות ............. Skeuomorphism
#   5 Monte Carlo ........ Y2K
_C = (
    '[data-testid="stMain"] :is([data-testid="stMetric"],'
    'div[data-testid="stVerticalBlockBorderWrapper"],[data-testid="stForm"],'
    '[data-testid="stExpander"],[data-testid="stAlert"])'
)
_H = '[data-testid="stMain"] :is(h1,h2,h3,.cx-hero-title)'
_MV = '[data-testid="stMain"] [data-testid="stMetricValue"]'
_ML = '[data-testid="stMain"] [data-testid="stMetricLabel"] p'

_SECTION_THEMES_RAW = r"""
/* 0 · סקירה — MINIMALISM : near-monochrome, air, one hairline, no gloss ---- */
html[data-cx-section="0"]{
  --gold:#E9E9F1;--gold-bright:#fff;--gold-deep:#B4B4CA;--gold-text:#ECECF3;
  --gold-wash:rgba(255,255,255,.04);--gold-line:rgba(255,255,255,.12);
  --glow:0 0 0 1px rgba(255,255,255,.14);
}
html[data-cx-section="0"] .stApp::after{opacity:.2;}
html[data-cx-section="0"] §CARDS§{
  background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.10)!important;
  border-radius:8px!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  box-shadow:none!important;
}
html[data-cx-section="0"] §CARDS§::before,html[data-cx-section="0"] §CARDS§::after{display:none!important;}
html[data-cx-section="0"] §HEADS§{font-family:'Space Grotesk',var(--sans)!important;font-weight:500!important;letter-spacing:0!important;}
html[data-cx-section="0"] .cx-hero-title{font-weight:600!important;text-shadow:none!important;letter-spacing:-.02em!important;}
html[data-cx-section="0"] .cx-hero-ghost{opacity:.3;}
html[data-cx-section="0"] §MV§{font-family:'Space Grotesk',var(--sans)!important;font-weight:600!important;}

/* 1 · טרמינל מסחר — CYBERPUNK : neon cyan/magenta, scanlines, hard glow ----- */
html[data-cx-section="1"]{
  --gold:#00F0FF;--gold-bright:#8BFEFF;--gold-deep:#00AEBB;--gold-text:#9DFBFF;
  --gold-wash:rgba(0,240,255,.08);--gold-line:rgba(0,240,255,.34);
  --glow:0 0 0 1px rgba(0,240,255,.4),0 0 26px -2px rgba(0,240,255,.5);
}
html[data-cx-section="1"] §CARDS§{
  background:linear-gradient(180deg,rgba(4,10,20,.62),rgba(6,14,26,.5))!important;
  border:1px solid rgba(0,240,255,.32)!important;border-radius:3px!important;
  box-shadow:0 0 0 1px rgba(0,240,255,.12),0 0 34px -10px rgba(0,240,255,.45),inset 0 0 24px -14px rgba(255,0,200,.55)!important;
}
html[data-cx-section="1"] §CARDS§::before{display:none!important;}
html[data-cx-section="1"] §CARDS§::after{
  display:block!important;content:""!important;padding:0!important;
  background:repeating-linear-gradient(0deg,rgba(0,240,255,.055) 0 1px,transparent 1px 3px)!important;
  -webkit-mask:none!important;mask:none!important;mix-blend-mode:screen!important;opacity:.7;
}
html[data-cx-section="1"] §HEADS§{font-family:'Orbitron',var(--sans)!important;text-transform:uppercase!important;letter-spacing:.04em!important;font-weight:700!important;}
html[data-cx-section="1"] .cx-hero-title{color:#CFFEFF!important;-webkit-text-fill-color:#CFFEFF!important;text-shadow:0 0 22px rgba(0,240,255,.6),2px 0 3px rgba(255,0,200,.5)!important;}
html[data-cx-section="1"] §MV§{font-family:var(--mono)!important;color:#8BFEFF!important;-webkit-text-fill-color:#8BFEFF!important;text-shadow:0 0 14px rgba(0,240,255,.55)!important;}
html[data-cx-section="1"] §ML§{color:#5FD8E4!important;}

/* 2 · מחשבון הסתברויות — NEUMORPHISM : soft extruded, dual shadow, no border  */
html[data-cx-section="2"]{
  --gold:#9AA7D4;--gold-bright:#C2CBEC;--gold-deep:#6D7AA6;--gold-text:#C8CFEC;
  --gold-wash:rgba(154,167,212,.10);--gold-line:rgba(154,167,212,.24);
  --glow:0 0 0 1px rgba(154,167,212,.24);
}
html[data-cx-section="2"] .stApp::after{opacity:.28;}
html[data-cx-section="2"] §CARDS§{
  background:#1B1B1D!important;border:1px solid rgba(255,255,255,.03)!important;border-radius:22px!important;
  backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  box-shadow:-8px -8px 20px rgba(255,255,255,.035),10px 10px 26px rgba(0,0,0,.55)!important;
}
html[data-cx-section="2"] §CARDS§::before,html[data-cx-section="2"] §CARDS§::after{display:none!important;}
html[data-cx-section="2"] §CARDS§:hover{box-shadow:-4px -4px 12px rgba(255,255,255,.03),5px 5px 14px rgba(0,0,0,.5)!important;transform:none!important;}
html[data-cx-section="2"] §HEADS§{font-family:var(--sans)!important;font-weight:600!important;color:#D6DAF0!important;}
html[data-cx-section="2"] .cx-hero-title{color:#D9DDF2!important;-webkit-text-fill-color:#D9DDF2!important;text-shadow:-3px -3px 8px rgba(255,255,255,.04),4px 4px 12px rgba(0,0,0,.5)!important;}
html[data-cx-section="2"] §MV§{color:#D6DAF0!important;-webkit-text-fill-color:#D6DAF0!important;}
html[data-cx-section="2"] [data-baseweb="slider"] [role="slider"]{box-shadow:-3px -3px 8px rgba(255,255,255,.04),4px 4px 10px rgba(0,0,0,.5)!important;}

/* 3 · אבחון סטטיסטי — EDITORIAL : serif, rules not boxes, long measure ------ */
html[data-cx-section="3"]{
  --gold:#D8C6A0;--gold-bright:#F1E6CB;--gold-deep:#A88E5E;--gold-text:#E7DAC0;
  --gold-wash:rgba(216,198,160,.08);--gold-line:rgba(216,198,160,.28);
  --glow:0 0 0 1px rgba(216,198,160,.28);
}
html[data-cx-section="3"] §CARDS§{
  background:rgba(255,255,255,.016)!important;border:0!important;
  border-top:1px solid rgba(216,198,160,.32)!important;border-bottom:1px solid rgba(216,198,160,.14)!important;
  border-radius:0!important;backdrop-filter:none!important;-webkit-backdrop-filter:none!important;box-shadow:none!important;
}
html[data-cx-section="3"] §CARDS§::before,html[data-cx-section="3"] §CARDS§::after{display:none!important;}
html[data-cx-section="3"] §HEADS§{font-family:'Playfair Display',var(--display)!important;font-weight:800!important;letter-spacing:-.015em!important;}
html[data-cx-section="3"] [data-testid="stMain"] h2{border-bottom:1px solid rgba(216,198,160,.24);padding-bottom:.3rem;}
html[data-cx-section="3"] .cx-hero-title{font-style:italic!important;font-weight:700!important;color:#F4ECDA!important;-webkit-text-fill-color:#F4ECDA!important;text-shadow:none!important;}
html[data-cx-section="3"] [data-testid="stMain"] [data-testid="stMarkdownContainer"] p{font-family:'Frank Ruhl Libre','Playfair Display',Georgia,serif!important;font-size:1.02rem!important;line-height:1.75!important;color:#D6D2C4!important;}
html[data-cx-section="3"] §MV§{font-family:'Playfair Display',var(--display)!important;font-weight:700!important;}

/* 5 · Monte Carlo — Y2K : chrome, iridescent conic rim, holo gradient text -- */
html[data-cx-section="5"]{
  --gold:#C6B4FF;--gold-bright:#E9DEFF;--gold-deep:#8E79DE;--gold-text:#E4DAFF;
  --gold-wash:rgba(198,180,255,.12);--gold-line:rgba(198,180,255,.34);
  --glow:0 0 0 1px rgba(255,255,255,.18),0 0 24px -4px rgba(150,120,255,.5);
}
html[data-cx-section="5"] §CARDS§{
  background:linear-gradient(160deg,rgba(255,255,255,.16),rgba(200,190,255,.06) 45%,rgba(120,200,255,.1))!important;
  border:1px solid transparent!important;border-radius:22px!important;
  box-shadow:0 0 0 1.5px rgba(255,255,255,.18),0 20px 50px -18px rgba(150,120,255,.5),inset 0 2px 8px rgba(255,255,255,.4)!important;
}
html[data-cx-section="5"] §CARDS§::before{display:none!important;}
html[data-cx-section="5"] §CARDS§::after{
  display:block!important;content:""!important;padding:1.6px!important;mix-blend-mode:normal!important;opacity:.9;
  background:conic-gradient(from 0deg,#88CCFF,#C9B4FF,#FFB3E6,#9BE8FF,#88CCFF)!important;
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0)!important;
  mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0)!important;
  -webkit-mask-composite:xor!important;mask-composite:exclude!important;
}
html[data-cx-section="5"] §HEADS§{font-family:'Orbitron',var(--sans)!important;font-weight:700!important;letter-spacing:.01em!important;}
html[data-cx-section="5"] .cx-hero-title::before,html[data-cx-section="5"] .cx-hero-title::after{display:none!important;}
html[data-cx-section="5"] .cx-hero-title{
  background:linear-gradient(180deg,#fff,#C9B4FF 58%,#8AD4FF)!important;-webkit-background-clip:text!important;
  background-clip:text!important;-webkit-text-fill-color:transparent!important;color:transparent!important;
  text-shadow:0 6px 30px rgba(150,120,255,.5)!important;
}
html[data-cx-section="5"] §MV§{
  background:linear-gradient(180deg,#fff,#C9B4FF)!important;-webkit-background-clip:text!important;
  background-clip:text!important;-webkit-text-fill-color:transparent!important;
}

/* 6 · מבנים מתמטיים — NEO-BRUTALISM : raw, 2px borders, hard offset shadow -- */
html[data-cx-section="6"]{
  --gold:#C9FF3D;--gold-bright:#E3FF8A;--gold-deep:#9BD400;--gold-text:#DBFF7A;
  --gold-wash:rgba(201,255,61,.12);--gold-line:rgba(201,255,61,.5);
  --glow:0 0 0 2px #C9FF3D;
}
html[data-cx-section="6"] .stApp::after{opacity:.42;}
html[data-cx-section="6"] §CARDS§{
  background:#0C0C12!important;border:2px solid #F4F4F8!important;border-radius:0!important;
  backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  box-shadow:6px 6px 0 0 #C9FF3D,11px 11px 0 0 rgba(0,0,0,.6)!important;
}
html[data-cx-section="6"] §CARDS§::before,html[data-cx-section="6"] §CARDS§::after{display:none!important;}
html[data-cx-section="6"] §CARDS§:hover{transform:translate(-2px,-2px)!important;box-shadow:8px 8px 0 0 #C9FF3D,14px 14px 0 0 rgba(0,0,0,.6)!important;}
html[data-cx-section="6"] §HEADS§{font-family:'Archivo Black',var(--sans)!important;text-transform:uppercase!important;letter-spacing:-.01em!important;font-weight:400!important;color:#F4F4F8!important;}
html[data-cx-section="6"] .cx-hero-title{color:#F4F4F8!important;-webkit-text-fill-color:#F4F4F8!important;text-shadow:4px 4px 0 #C9FF3D!important;}
html[data-cx-section="6"] .cx-hero-ghost{-webkit-text-stroke:2px rgba(201,255,61,.28)!important;opacity:1;}
html[data-cx-section="6"] §MV§{font-family:'Archivo Black',var(--mono)!important;color:#F4F4F8!important;-webkit-text-fill-color:#F4F4F8!important;}
html[data-cx-section="6"] §ML§{color:#C9FF3D!important;text-transform:uppercase!important;}

/* 7 · מחזור ביטקוין → MSTR — MAXIMALISM : layered orange/magenta, saturated - */
html[data-cx-section="7"]{
  --gold:#F7931A;--gold-bright:#FFC061;--gold-deep:#C46F00;--gold-text:#FFB454;
  --gold-wash:rgba(247,147,26,.14);--gold-line:rgba(247,147,26,.4);
  --glow:0 0 0 1px rgba(255,180,84,.35),0 0 30px -4px rgba(247,147,26,.5);
}
html[data-cx-section="7"] .stApp::after{opacity:.6;}
html[data-cx-section="7"] §CARDS§{
  background:
    radial-gradient(120% 140% at 0% 0%,rgba(247,147,26,.22),transparent 55%),
    radial-gradient(120% 140% at 100% 100%,rgba(255,61,166,.18),transparent 55%),
    linear-gradient(180deg,rgba(30,16,4,.6),rgba(20,10,26,.55))!important;
  border:1px solid rgba(247,147,26,.42)!important;border-radius:16px!important;
  box-shadow:0 0 0 1px rgba(255,180,84,.2),0 0 40px -8px rgba(247,147,26,.5),0 26px 60px -20px rgba(0,0,0,.7),inset 0 1px 0 rgba(255,220,180,.35)!important;
}
html[data-cx-section="7"] §CARDS§::before{display:none!important;}
html[data-cx-section="7"] §HEADS§{font-family:'Playfair Display',var(--display)!important;font-weight:900!important;letter-spacing:-.02em!important;}
html[data-cx-section="7"] .cx-hero-title::before,html[data-cx-section="7"] .cx-hero-title::after{display:none!important;}
html[data-cx-section="7"] .cx-hero-title{
  background:linear-gradient(92deg,#FFD9A8,#F7931A 45%,#FF3DA6)!important;-webkit-background-clip:text!important;
  background-clip:text!important;-webkit-text-fill-color:transparent!important;color:transparent!important;
  text-shadow:0 8px 40px rgba(247,147,26,.55)!important;
}
html[data-cx-section="7"] §MV§{color:#FFC061!important;-webkit-text-fill-color:#FFC061!important;text-shadow:0 0 20px rgba(247,147,26,.4)!important;}
html[data-cx-section="7"] §ML§{color:#FFB454!important;}

/* 8 · ראיות Walk-Forward — MATERIAL : tonal surface, crisp elevation, calm -- */
html[data-cx-section="8"]{
  --gold:#7C9EFF;--gold-bright:#A9C1FF;--gold-deep:#4E6FD8;--gold-text:#AFC4FF;
  --gold-wash:rgba(124,158,255,.10);--gold-line:rgba(124,158,255,.3);
  --glow:0 0 0 1px rgba(124,158,255,.3);
}
html[data-cx-section="8"] .stApp::after{opacity:.28;}
html[data-cx-section="8"] §CARDS§{
  background:#1C1C1F!important;border:1px solid rgba(255,255,255,.05)!important;border-radius:12px!important;
  backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  box-shadow:0 1px 2px rgba(0,0,0,.5),0 8px 18px -6px rgba(0,0,0,.5)!important;
}
html[data-cx-section="8"] §CARDS§::before,html[data-cx-section="8"] §CARDS§::after{display:none!important;}
html[data-cx-section="8"] §CARDS§:hover{box-shadow:0 2px 4px rgba(0,0,0,.5),0 14px 30px -8px rgba(0,0,0,.6)!important;transform:translateY(-1px)!important;}
html[data-cx-section="8"] §HEADS§{font-family:var(--sans)!important;font-weight:600!important;letter-spacing:0!important;}
html[data-cx-section="8"] .cx-hero-title{font-weight:700!important;color:#EEF1FF!important;-webkit-text-fill-color:#EEF1FF!important;text-shadow:none!important;}
html[data-cx-section="8"] §MV§{font-family:var(--sans)!important;font-weight:600!important;}

/* 9 · סיכון ומינוף — CLAYMORPHISM : puffy tactile clay, deep soft shadow ---- */
html[data-cx-section="9"]{
  --gold:#FF8FA3;--gold-bright:#FFB3C1;--gold-deep:#D65E76;--gold-text:#FFB0BE;
  --gold-wash:rgba(255,143,163,.12);--gold-line:rgba(255,143,163,.34);
  --glow:0 0 0 1px rgba(255,143,163,.34);
}
html[data-cx-section="9"] §CARDS§{
  background:#26202E!important;border:0!important;border-radius:26px!important;
  backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  box-shadow:0 18px 40px -12px rgba(0,0,0,.6),inset 6px 6px 14px rgba(255,255,255,.06),inset -8px -10px 20px rgba(0,0,0,.5)!important;
}
html[data-cx-section="9"] §CARDS§::before,html[data-cx-section="9"] §CARDS§::after{display:none!important;}
html[data-cx-section="9"] §HEADS§{font-family:'Space Grotesk',var(--sans)!important;font-weight:700!important;color:#FDE9EC!important;}
html[data-cx-section="9"] .cx-hero-title{color:#FFE7EC!important;-webkit-text-fill-color:#FFE7EC!important;text-shadow:0 6px 16px rgba(0,0,0,.4)!important;}
html[data-cx-section="9"] §MV§{color:#FFB3C1!important;-webkit-text-fill-color:#FFB3C1!important;}
html[data-cx-section="9"] [data-testid="stMain"] .stButton>button{border-radius:16px!important;box-shadow:0 8px 18px -6px rgba(0,0,0,.5),inset 3px 3px 8px rgba(255,255,255,.08),inset -4px -5px 10px rgba(0,0,0,.4)!important;}

/* 10 · הגדרות — SKEUOMORPHISM : brushed metal, bevel, physical depth -------- */
html[data-cx-section="10"]{
  --gold:#BFC5D0;--gold-bright:#E0E4EC;--gold-deep:#8C92A0;--gold-text:#D8DCE6;
  --gold-wash:rgba(191,197,208,.10);--gold-line:rgba(191,197,208,.3);
  --glow:0 0 0 1px rgba(191,197,208,.3);
}
html[data-cx-section="10"] §CARDS§{
  background:linear-gradient(180deg,#2C2C2F 0%,#202023 48%,#1A1A1C 52%,#212124 100%)!important;
  border:1px solid #3C3C40!important;border-top-color:#4E4E54!important;border-radius:10px!important;
  backdrop-filter:none!important;-webkit-backdrop-filter:none!important;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.12),inset 0 -1px 0 rgba(0,0,0,.5),0 2px 3px rgba(0,0,0,.4),0 14px 30px -14px rgba(0,0,0,.6)!important;
}
html[data-cx-section="10"] §CARDS§::before,html[data-cx-section="10"] §CARDS§::after{display:none!important;}
html[data-cx-section="10"] §HEADS§{font-family:var(--sans)!important;font-weight:700!important;color:#E7EAF2!important;text-shadow:0 1px 0 rgba(0,0,0,.5)!important;}
html[data-cx-section="10"] .cx-hero-title{color:#EDEFF5!important;-webkit-text-fill-color:#EDEFF5!important;text-shadow:0 1px 0 rgba(0,0,0,.6),0 2px 6px rgba(0,0,0,.4)!important;}
html[data-cx-section="10"] §MV§{color:#E0E4EC!important;-webkit-text-fill-color:#E0E4EC!important;text-shadow:0 1px 0 rgba(0,0,0,.5)!important;}
html[data-cx-section="10"] [data-baseweb="slider"] [data-testid="stSliderTrack"]{box-shadow:inset 0 1px 3px rgba(0,0,0,.6)!important;}
"""


def _section_themes() -> str:
    return (
        _SECTION_THEMES_RAW
        .replace("§CARDS§", _C)
        .replace("§HEADS§", _H)
        .replace("§MV§", _MV)
        .replace("§ML§", _ML)
    )


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
  --gold-wash:rgba(91,84,232,0.10); --gold-line:rgba(91,84,232,0.28);
  --glow:0 0 0 1px rgba(91,84,232,.28), 0 0 24px -4px rgba(91,84,232,.35);
  --text:{TEXT}; --text-dim:{TEXT_DIM}; --text-faint:{TEXT_FAINT};
  --up:{UP}; --down:{DOWN};
  --display:{_DISPLAY}; --sans:{_SANS}; --mono:{_MONO};
  --cx-sec-i: 0;
}}

/* ---------- aurora ground ---------- *
 * back -> front:  the live colour mesh (body::before)
 *                 a finer mesh that leans with the pointer (.stApp::before)
 *                 grain + vignette (.stApp::after)
 *                 the section-change sweep + flash (.cx-wipe / .cx-flash)       */
html, body, [class*="stApp"] {{ font-family: var(--sans); }}
html, body {{
  background: linear-gradient(178deg, {OCEAN_TOP} 0%, {OCEAN_MID} 44%, {OCEAN_DEEP} 100%);
}}
.stApp {{ background: transparent; color: var(--text); }}

/* ELECTRIC OCEAN -- a bright tropical lagoon the whole app floats on.
   back -> front:  caustic light (body::before, drifts on the compositor)
                   a moving surface glare (.stApp::before, pointer/scroll)
                   a soft depth wash so the reading slab detaches (.stApp::after)
                   the storm layer (.cx-storm) + click strikes are added by JS   */
body::before {{
  content: ""; position: fixed; inset: -16vmax; z-index: 0; pointer-events: none;
  background:
    radial-gradient(38vmax 26vmax at 18% 12%, rgba(255,255,255,0.55), rgba(255,255,255,0) 60%),
    radial-gradient(52vmax 40vmax at 82% 22%, rgba(220,252,255,0.5), rgba(220,252,255,0) 62%),
    radial-gradient(64vmax 52vmax at 44% 108%, rgba(120,236,255,0.55), rgba(120,236,255,0) 66%),
    radial-gradient(46vmax 40vmax at 6% 88%, rgba(255,255,255,0.4), rgba(255,255,255,0) 60%),
    conic-gradient(from 200deg at 60% 40%, rgba(255,255,255,0.10), rgba(120,236,255,0) 30%, rgba(255,255,255,0.12) 62%, rgba(120,236,255,0) 100%);
  animation: cx-caustic 34s ease-in-out infinite alternate;
  transform: translateZ(0); will-change: transform; backface-visibility: hidden;
}}
.stApp::before {{
  content: ""; position: fixed; inset: -12vmax; z-index: 0; pointer-events: none;
  background:
    linear-gradient(116deg,
      rgba(255,255,255,0) 36%, rgba(255,255,255,0.18) 47%,
      rgba(255,255,255,0.42) 50%, rgba(255,255,255,0.16) 53%, rgba(255,255,255,0) 64%),
    radial-gradient(30vmax 22vmax at 70% 26%, rgba(255,255,255,0.28), rgba(255,255,255,0) 60%);
  transform: translate3d(calc(var(--cx-mx,0px) * 1.5), calc(var(--cx-plate,0px) + var(--cx-my,0px) * 1.5), 0) scale(1.06);
  transition: transform .2s cubic-bezier(.2,.8,.2,1);
  will-change: transform; backface-visibility: hidden;
}}
.stApp::after {{
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: radial-gradient(160% 130% at 50% 40%, rgba(6,26,36,0) 30%, rgba(6,24,34,0.36) 100%);
  transform: translateZ(0); backface-visibility: hidden;
}}

/* the reading slab -- a single frosted panel the content rides over the water.
   ONE backdrop-filter on ONE element (not per-card) -> perf-safe. */
[data-testid="stMain"] .block-container {{
  background: {SLAB};
  -webkit-backdrop-filter: blur(9px) saturate(1.15); backdrop-filter: blur(9px) saturate(1.15);
  border: 1px solid rgba(255,255,255,0.13);
  border-radius: 26px;
  box-shadow: 0 44px 130px -44px rgba(3,20,30,0.78), inset 0 1px 0 rgba(255,255,255,0.15);
  margin: 1.4rem auto 3rem;
  padding-left: 2.1rem; padding-right: 2.1rem;
}}


/* section-change sweep -- an electric prism edge */
.cx-wipe {{
  position: fixed; inset: 0 -46vw; z-index: 6; pointer-events: none;
  opacity: 0; transform: translateX(-125%) skewX(-15deg);
  background: linear-gradient(90deg,
    rgba(255,255,255,0) 0%, rgba(255,255,255,0.28) 34%,
    rgba(255,255,255,0.72) 46%, rgba(255,255,255,0.96) 50%,
    rgba(255,255,255,0.72) 54%, rgba(255,255,255,0.28) 66%, rgba(255,255,255,0) 100%);
  box-shadow: 0 0 100px 16px rgba(255,255,255,0.18);
}}
.cx-wipe.run {{ animation: cx-wipe .8s cubic-bezier(.62,0,.28,1); }}
.cx-flash {{
  position: fixed; inset: 0; z-index: 5; pointer-events: none; opacity: 0;
  background: radial-gradient(120% 120% at 50% 30%,
    rgba(255,255,255,0.07) 0%, rgba(255,255,255,0) 62%);
}}
.cx-flash.run {{ animation: cx-flash .7s ease-out; }}

@keyframes cx-wipe {{
  0%   {{ opacity: 0; transform: translateX(-125%) skewX(-15deg); }}
  18%  {{ opacity: 1; }}
  100% {{ opacity: 0; transform: translateX(125%) skewX(-15deg); }}
}}
@keyframes cx-flash {{
  0% {{ opacity: 0; }}  18% {{ opacity: 1; }}  100% {{ opacity: 0; }}
}}
@keyframes cx-caustic {{
  0%   {{ transform: translate3d(0,0,0) scale(1); }}
  100% {{ transform: translate3d(-3vmax, 2.4vmax, 0) scale(1.14); }}
}}

/* ===================== the storm ===================== *
 * ambient bolts at the edges of the lagoon (.cx-storm, behind the slab) and a
 * strike on every click (.cx-strike-wrap, above everything).  Opacity-only
 * animations -- compositor cheap.  Both layers are built by scroll_boot.        */
.cx-storm {{
  position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;
  contain: strict;
}}
.cx-storm i {{
  position: absolute; top: -6vh; display: block; opacity: 0;
  background: no-repeat center top / 100% 100%; will-change: opacity;
}}
.cx-storm .b1 {{ left: 3%;  width: 13vmin; height: 66vh; background-image: url("{_BOLT_A}"); animation: cx-ambient 8.5s linear infinite 1.5s; }}
.cx-storm .b2 {{ right: 2%; width: 11vmin; height: 58vh; background-image: url("{_BOLT_B}"); animation: cx-ambient 12s linear infinite 5s; }}
.cx-storm .b3 {{ left: 44%; width: 9vmin;  height: 40vh; background-image: url("{_BOLT_A}"); animation: cx-ambient 17s linear infinite 10s; }}
.cx-storm::after {{
  content: ""; position: absolute; inset: 0; opacity: 0; mix-blend-mode: screen;
  background: radial-gradient(120% 90% at 50% 0%, rgba(255,255,255,0.5), rgba(210,245,255,0) 70%);
  animation: cx-ambient 8.5s linear infinite 1.5s;
}}
@keyframes cx-ambient {{
  0%, 6%, 100% {{ opacity: 0; }}
  1.4% {{ opacity: .95; }}
  2.6% {{ opacity: .12; }}
  3.6% {{ opacity: .8; }}
  5% {{ opacity: 0; }}
}}
.cx-strike-wrap {{ position: fixed; inset: 0; z-index: 1200; pointer-events: none; }}
.cx-strike {{
  position: absolute; inset: 0; opacity: 0;
  background: radial-gradient(70vmax 70vmax at var(--sx,50%) var(--sy,38%),
    rgba(232,251,255,0.62), rgba(190,238,255,0) 58%);
}}
.cx-strike-bolt {{
  position: absolute; left: var(--sx,50%); top: -8vh; width: 24vmin;
  height: calc(var(--sy,38vh) + 10vh); transform: translateX(-50%); opacity: 0;
  background: url("{_BOLT_A}") no-repeat center bottom / 100% 100%;
}}
.cx-strike-wrap.on .cx-strike {{ animation: cx-strike-flash .42s ease-out; }}
.cx-strike-wrap.on .cx-strike-bolt {{ animation: cx-strike-bolt .42s ease-out; }}
@keyframes cx-strike-flash {{
  0% {{ opacity: 0; }} 8% {{ opacity: 1; }} 20% {{ opacity: .22; }}
  30% {{ opacity: .66; }} 100% {{ opacity: 0; }}
}}
@keyframes cx-strike-bolt {{
  0% {{ opacity: 0; }} 6% {{ opacity: 1; }} 16% {{ opacity: .28; }}
  26% {{ opacity: .92; }} 55% {{ opacity: 0; }} 100% {{ opacity: 0; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .cx-storm, .cx-strike-wrap {{ display: none !important; }}
}}

/* per-section hue shift for the section-change flash (cheap; one var). */
:root {{ --cx-hue: calc((var(--cx-sec-i, 0) - 5) * 9deg); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stHeader"], [data-testid="stSidebarContent"] {{ background: transparent; }}
[data-testid="stAppViewContainer"] > .main, .block-container {{ position: relative; z-index: 1; }}
[data-testid="stToolbar"] {{ right: .5rem; }}
[data-testid="stAppDeployButton"], [data-testid="stDeployButton"] {{ display: none !important; }}
[data-testid="stMain"] .block-container {{ padding-top: 1.1rem; max-width: 1180px; }}
[data-testid="stSidebarCollapsedControl"] {{ opacity: .5; }}
[data-testid="stSidebar"] {{ background: rgba(10,10,11,0.92); -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px); border-right: 1px solid var(--line-soft); }}

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
  -webkit-backdrop-filter: blur(6px) saturate(1.4); backdrop-filter: blur(6px) saturate(1.4);
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
  background: rgba(16,16,18,0.34);
  -webkit-backdrop-filter: blur(5px) saturate(1.3); backdrop-filter: blur(5px) saturate(1.3);
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
  background: rgba(26,26,28,0.62);
  -webkit-backdrop-filter: blur(6px) saturate(1.4); backdrop-filter: blur(6px) saturate(1.4);
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
  background: linear-gradient(135deg, rgba(150,236,255,0.5), rgba(90,210,240,0.34));
  border-color: rgba(180,245,255,0.75);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.75), inset 0 0 20px -4px rgba(255,255,255,0.45),
              0 0 42px -4px rgba(110,225,255,0.85), 0 16px 40px -12px rgba(0,0,0,0.55);
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
  box-shadow: 0 24px 60px -12px rgba(110,225,255,.65), 0 4px 16px rgba(0,0,0,.35),
              inset 0 0 0 2px rgba(255,255,255,.35), inset 0 3px 10px rgba(255,255,255,.25);
}}
.cx-orbit-hub::before {{  /* soft always-on aura */
  content: ""; position: absolute; inset: -18px; border-radius: 50%; z-index: 0;
  background: radial-gradient(circle, rgba(120,230,255,.55) 0%, rgba(255,255,255,.25) 42%, rgba(120,230,255,0) 72%);
  filter: blur(6px); animation: cx-hub-pulse 4.5s ease-in-out infinite;
}}
.cx-orbit-hub::after {{  /* rotating conic ring */
  content: ""; position: absolute; inset: -12px; border-radius: 50%; z-index: 1;
  background: conic-gradient(from 0deg, rgba(120,230,255,0), rgba(240,252,255,.9), rgba(90,216,241,.5), rgba(255,255,255,.35), rgba(120,230,255,0) 70%);
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
  -webkit-backdrop-filter: blur(8px) saturate(1.5); backdrop-filter: blur(8px) saturate(1.5);
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
  background: rgba(8,8,9,0.62) !important;
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

/* ---------- metric  (translucent plaque -- NO backdrop-filter: there can be
   30+ of these on one screen; a real blur per card destroys scroll) ---------- */
[data-testid="stMetric"] {{
  background: linear-gradient(180deg, rgba(26,26,28,0.62), rgba(15,15,16,0.54));
  border: 1px solid var(--line); border-radius: 18px; padding: 1rem 1.15rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.28), 0 18px 42px -22px rgba(0,0,0,0.6);
  contain: layout style;
  transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
}}
[data-testid="stMetric"]:hover {{
  background: linear-gradient(180deg, rgba(36,36,39,0.7), rgba(20,20,22,0.6));
  border-color: rgba(155,140,255,0.35);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.4), 0 0 22px -8px rgba(91,84,232,0.3), 0 20px 48px -20px rgba(0,0,0,0.5);
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
  border: 1px solid var(--gold-line); background: rgba(255,255,255,0.06); color: var(--gold-text);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.22);
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
  background: rgba(255,255,255,0.055) !important; color: var(--text) !important; font-family: var(--mono);
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
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(24,24,26,0.64), rgba(15,15,16,0.56)) !important;
}}
[data-testid="stAlert"] {{ border: 1px solid var(--line); border-inline-start: 3px solid var(--gold); }}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p, [data-testid="stAlertContainer"] p {{
  color: var(--text) !important;
}}
[data-testid="stExpander"] {{
  border-radius: 16px; border: 1px solid var(--line); overflow: hidden;
  background: linear-gradient(180deg, rgba(23,23,25,0.6), rgba(14,14,15,0.52));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.24), 0 18px 42px -22px rgba(0,0,0,0.6);
  contain: layout style;
}}
[data-testid="stExpander"] summary:hover {{ background: var(--gold-wash); }}
[data-testid="stForm"] {{
  border-radius: 18px; border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(24,22,42,0.58), rgba(15,14,28,0.5));
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.24), 0 18px 42px -22px rgba(0,0,0,0.6);
  contain: layout style;
}}
/* bordered container -> a barely-there frame so the inner card carries the weight */
div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-radius: 20px;
  background: rgba(255,255,255,0.035);
  box-shadow: inset 0 0 0 1px rgba(255,255,255,0.14);
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
  /* reveal cascade -- compositor-only (opacity + translate); no filter/blur so
     20 blocks animating at once on first paint stays smooth. */
  html.cx-js [data-testid="stMain"] .block-container [data-testid="stElementContainer"].cx-hide,
  html.cx-js [data-testid="stMain"] .block-container [data-testid="stHorizontalBlock"].cx-hide {{
    opacity: 0; transform: translateY(26px);
  }}
  html.cx-js .cx-hero.cx-hide {{ opacity: 0; transform: translateY(32px); }}
  html.cx-js .cx-hero.cx-hide .cx-hero-title {{ letter-spacing: .02em; }}
  html.cx-js .cx-hide {{
    transition: opacity .6s cubic-bezier(.16,.72,.24,1),
                transform .6s cubic-bezier(.16,.72,.24,1),
                letter-spacing .6s ease;
    will-change: opacity, transform;
  }}
  html.cx-js .cx-hide .cx-hero-title {{ transition: letter-spacing .6s ease; }}
  html.cx-js .cx-rev {{ opacity: 1 !important; transform: none !important; will-change: auto; }}
  html.cx-js .cx-rev .cx-hero-title {{ letter-spacing: -0.01em; }}
}}

/* ====================== glass finish (static -- zero scroll cost) ============ *
 * The Apple-"liquid glass" refraction (SVG feDisplacementMap + per-card
 * backdrop-filter) was removed: on a 30-metric screen it re-ran a turbulence
 * filter per card per frame and wrecked scrolling.  Replaced with a purely
 * static sheen (::before, one gradient) + specular rim (::after, gradient + mask,
 * rasterised once).  No filters, no backdrop-filter -- nothing repaints on
 * scroll.  Look is preserved: bright top-left rim, soft inner glow.            */
.cx-orbit-item > button,
.st-key-cx_refresh button {{
  position: relative; overflow: hidden; isolation: isolate;
}}
[data-testid="stMetric"]::before,
[data-testid="stForm"]::before,
[data-testid="stExpander"]::before,
.cx-orbit-item > button::before,
.st-key-cx_refresh button::before {{
  content: ""; position: absolute; inset: 0; z-index: 0; border-radius: inherit;
  pointer-events: none;
  background:
    linear-gradient(135deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0) 38%),
    radial-gradient(120% 90% at 12% 0%, rgba(255,255,255,0.06), rgba(255,255,255,0) 60%);
}}
[data-testid="stMetric"]::after,
[data-testid="stForm"]::after,
[data-testid="stExpander"]::after,
.cx-orbit-item > button::after,
.st-key-cx_refresh button::after {{
  content: ""; position: absolute; inset: 0; z-index: 1; border-radius: inherit;
  padding: 1.1px; pointer-events: none; mix-blend-mode: screen;
  background: linear-gradient(135deg,
    rgba(255,255,255,0.85) 0%, rgba(255,255,255,0.20) 18%,
    rgba(255,255,255,0) 42%, rgba(255,255,255,0) 62%,
    rgba(200,220,255,0.22) 86%, rgba(255,255,255,0.42) 100%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
          mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;  mask-composite: exclude;
}}
[data-testid="stMetric"], [data-testid="stForm"], [data-testid="stExpander"] {{
  position: relative; isolation: isolate;
}}
[data-testid="stMetric"] > *, [data-testid="stForm"] > *, [data-testid="stExpander"] > * {{ position: relative; z-index: 1; }}

/* ====================== per-section design languages ====================== */
{_section_themes()}

/* ================= CARN agent -- carved mahogany ================= *
 * The one warm surface in the app: deep figured mahogany with gold inlay,
 * a carved bevel, sharp bright specular.  Lives only in the sidebar.        */
[data-testid="stSidebar"] {{
  background:
    linear-gradient(115deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 22%),
    repeating-linear-gradient(94deg, rgba(0,0,0,0.05) 0 2px, rgba(255,255,255,0.018) 2px 5px),
    linear-gradient(160deg, #6E3320 0%, #521F12 40%, #3A1109 74%, #2A0B05 100%) !important;
  border-right: 1px solid rgba(247,227,166,0.35) !important;
  box-shadow: inset 0 0 0 1px rgba(247,227,166,0.10),
              inset 0 40px 90px -60px rgba(255,220,180,0.5),
              12px 0 48px -12px rgba(0,0,0,0.6) !important;
  -webkit-backdrop-filter: none !important; backdrop-filter: none !important;
}}
[data-testid="stSidebar"] * {{ --gold: #F1D89A; --gold-text: #F6E7C3; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] .stMarkdown {{
  color: #EAD9C2 !important;
}}
.cx-agent-head {{
  display: flex; align-items: center; gap: .6rem; padding: .2rem 0 .7rem;
  border-bottom: 1px solid rgba(247,227,166,0.22); margin-bottom: .6rem;
}}
.cx-agent-head svg {{
  filter: drop-shadow(0 3px 8px rgba(0,0,0,0.5)) drop-shadow(0 0 10px rgba(127,233,255,0.25));
  flex: 0 0 auto;
}}
.cx-agent-id {{ display: flex; flex-direction: column; line-height: 1.15; }}
.cx-agent-id b {{
  font-family: var(--display); font-size: 1.12rem; letter-spacing: .02em;
  color: #FBEFD6; text-shadow: 0 1px 0 rgba(0,0,0,0.5), 0 0 14px rgba(247,227,166,0.3);
}}
.cx-agent-id span {{
  font-family: var(--mono); font-size: .58rem; letter-spacing: .12em;
  text-transform: uppercase; color: #C9A86E;
}}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{
  background: rgba(20,7,3,0.5) !important;
  border: 1px solid rgba(247,227,166,0.16) !important; border-radius: 12px !important;
  box-shadow: inset 0 2px 10px rgba(0,0,0,0.55), inset 0 0 0 1px rgba(255,220,180,0.05) !important;
  padding: .5rem !important;
}}
.cx-a-msg {{
  font-size: .82rem; line-height: 1.6; border-radius: 12px;
  padding: .55rem .7rem; margin: .38rem 0; max-width: 92%;
  border: 1px solid rgba(247,227,166,0.18);
  box-shadow: 0 6px 16px -8px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,235,200,0.14);
}}
.cx-a-msg b {{ color: #FBEAC6; }}
.cx-a-msg i {{ color: #C7AE8A; font-style: italic; }}
.cx-a-bot {{
  background: linear-gradient(165deg, #5A281A, #3D160C);
  color: #EEDCC4; margin-inline-end: auto; border-bottom-left-radius: 4px;
}}
.cx-a-me {{
  background: linear-gradient(165deg, #8A5A2C, #6B3F1C);
  color: #FBF0DC; margin-inline-start: auto;
  border-bottom-right-radius: 4px; border-color: rgba(247,227,166,0.32);
}}
[data-testid="stSidebar"] [data-testid="stChatInput"],
[data-testid="stSidebar"] [data-testid="stChatInput"] > div,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea {{
  background: linear-gradient(180deg, #3A1509, #2A0D05) !important;
  border: 1px solid rgba(247,227,166,0.34) !important; border-radius: 12px !important;
  color: #F6E7C3 !important;
  box-shadow: inset 0 2px 8px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,220,180,0.06) !important;
}}
[data-testid="stSidebar"] [data-testid="stChatInput"] textarea::placeholder {{ color: rgba(180,144,96,0.6) !important; }}
[data-testid="stSidebar"] [data-testid="stChatInputSubmitButton"] {{ color: #F1D89A !important; }}
[data-testid="stSidebar"] .stFormSubmitButton button {{
  background: linear-gradient(180deg, #C9973F, #9A6B23) !important;
  color: #2A0D05 !important; -webkit-text-fill-color: #2A0D05 !important;
  border: 1px solid rgba(247,227,166,0.5) !important; font-weight: 700 !important;
}}
.cx-agent-rule {{
  height: 2px; margin: .5rem 0 .2rem;
  background: linear-gradient(90deg, transparent, rgba(247,227,166,0.5), transparent);
}}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {{ color: #FBEFD6 !important; -webkit-text-fill-color: #FBEFD6 !important; }}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {{ color: #C9A86E !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(247,227,166,0.2) !important; }}
[data-testid="stSidebar"] [data-testid="stAlert"] {{
  background: rgba(20,7,3,0.5) !important; border: 1px solid rgba(247,227,166,0.2) !important;
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
