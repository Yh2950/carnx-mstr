"""
CARN-X brand mark  --  an engraved instrument face, not generator art.
=====================================================================
The composition is  ∫₀ᵀ  taken over a price path, with the calibrated
distribution drawn beneath it -- exactly the operation the Monte-Carlo engine
performs.  Antique-gold linework on a deep-navy plate.

  svg_app(size)       full-bleed app / favicon icon
  svg_maskable(size)  Android maskable (mark inside the safe circle)
  svg_wordmark(...)   horizontal lockup for the sidebar

Icons are rendered to PNG once by ``render.py`` and committed; the app shows the
wordmark SVG live (its serif loads through the page's own @font-face).
"""

from __future__ import annotations

NAVY_0 = "#141426"       # plate centre
NAVY_1 = "#08070F"       # plate edge
GOLD = "#9E8CFF"         # electric violet — the mark
GOLD_HI = "#C9BCFF"      # bright
GOLD_LO = "#4FD8EC"      # cyan foot
PATH_GOLD = "#7FE9F5"    # the curve
DIM = "#8A94AC"

SERIF = "Fraunces, Spectral, 'EB Garamond', Georgia, 'Times New Roman', serif"
_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,400&display=swap');"
)


def _defs(uid: str) -> str:
    return f"""<defs>
    <radialGradient id="pl{uid}" cx="38%" cy="30%" r="90%">
      <stop offset="0" stop-color="{NAVY_0}"/>
      <stop offset="0.62" stop-color="#0A0E1C"/>
      <stop offset="1" stop-color="{NAVY_1}"/>
    </radialGradient>
    <linearGradient id="gd{uid}" x1="0.15" y1="0" x2="0.6" y2="1">
      <stop offset="0" stop-color="{GOLD_HI}"/>
      <stop offset="0.45" stop-color="{GOLD}"/>
      <stop offset="1" stop-color="#4FD8EC"/>
    </linearGradient>
    <radialGradient id="bl{uid}" cx="46%" cy="42%" r="55%">
      <stop offset="0" stop-color="#8E7BF0" stop-opacity="0.40"/>
      <stop offset="0.5" stop-color="#A855F7" stop-opacity="0.12"/>
      <stop offset="1" stop-color="#A855F7" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="vg{uid}" cx="50%" cy="46%" r="62%">
      <stop offset="0" stop-color="#000" stop-opacity="0"/>
      <stop offset="0.8" stop-color="#000" stop-opacity="0"/>
      <stop offset="1" stop-color="#000" stop-opacity="0.34"/>
    </radialGradient>
  </defs>"""


def _scene(uid: str, scale: float = 1.0) -> str:
    """A single confident integral over a quiet Gaussian -- clean, luminous, centred."""
    # one smooth Gaussian, low on the grid, thin gilt, quiet
    bell = ("M64 372 C150 372 176 250 256 250 "
            "C336 250 362 372 448 372")
    axis = '<line x1="72" y1="356" x2="440" y2="356" stroke="%s" stroke-width="2" opacity="0.15"/>' % GOLD

    integral = (
        f'<text x="256" y="372" text-anchor="middle" font-family={SERIF!r} '
        f'font-weight="500" font-size="376" fill="url(#gd{uid})">∫</text>'
    )
    inner = f"""
    <circle cx="256" cy="252" r="188" fill="url(#bl{uid})"/>
    <path d="{bell}" fill="none" stroke="{GOLD}" stroke-width="3"
          stroke-linecap="round" opacity="0.34"/>
    {axis}
    {integral}"""
    if scale != 1.0:
        inner = f'<g transform="translate(256 256) scale({scale}) translate(-256 -256)">{inner}</g>'
    return inner


def _frame(size: int, inset: float) -> str:
    b = size * inset
    return (
        f'<rect x="{b:.1f}" y="{b:.1f}" width="{size-2*b:.1f}" height="{size-2*b:.1f}" '
        f'rx="{size*0.05:.1f}" fill="none" stroke="{GOLD}" '
        f'stroke-width="{max(1.0, size*0.0022):.2f}" opacity="0.12"/>'
    )


def svg_app(size: int = 512) -> str:
    r = size * 0.225
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}"><style>{_FONT_IMPORT}</style>{_defs("a")}'
        f'<rect width="{size}" height="{size}" rx="{r:.1f}" fill="url(#pla)"/>'
        f'<g transform="scale({size/512})">{_scene("a")}</g>'
        f'<rect width="{size}" height="{size}" rx="{r:.1f}" fill="url(#vga)"/></svg>'
    )


def svg_maskable(size: int = 512) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}"><style>{_FONT_IMPORT}</style>{_defs("m")}'
        f'<rect width="{size}" height="{size}" fill="url(#plm)"/>'
        f'<g transform="scale({size/512})">{_scene("m", scale=0.66)}</g>'
        f'<rect width="{size}" height="{size}" fill="url(#vgm)"/></svg>'
    )


def svg_wordmark(w: int = 540, h: int = 128) -> str:
    m = h - 14
    r = m * 0.235
    mark = (
        f'<g transform="translate(7 7)">'
        f'<rect width="{m}" height="{m}" rx="{m*0.24:.1f}" fill="url(#plw)"/>'
        f'<g transform="scale({m/512})">{_scene("w", scale=0.98)}</g>'
        f'<rect width="{m}" height="{m}" rx="{m*0.24:.1f}" fill="url(#vgw)"/></g>'
    )
    tx = h + 8
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <style>{_FONT_IMPORT}</style>{_defs("w")}
  {mark}
  <text x="{tx}" y="{h*0.50:.0f}" font-family={SERIF!r} font-weight="500"
        font-size="{h*0.37:.0f}" letter-spacing="0.5" fill="{GOLD_HI}">CARN<tspan fill="{GOLD_LO}" dx="1">-</tspan>X</text>
  <line x1="{tx+1}" y1="{h*0.61:.0f}" x2="{w-16}" y2="{h*0.61:.0f}" stroke="{GOLD}" stroke-width="1" opacity="0.35"/>
  <text x="{tx+1}" y="{h*0.80:.0f}" font-family="'IBM Plex Mono', ui-monospace, monospace"
        font-size="{h*0.115:.0f}" letter-spacing="1.6" fill="{DIM}">PROBABILISTIC · FORECASTING</text>
</svg>"""


if __name__ == "__main__":
    import pathlib

    d = pathlib.Path(__file__).parent
    (d / "icon_app.svg").write_text(svg_app(512), encoding="utf-8")
    (d / "icon_maskable.svg").write_text(svg_maskable(512), encoding="utf-8")
    (d / "wordmark.svg").write_text(svg_wordmark(), encoding="utf-8")
    print("wrote icon_app.svg, icon_maskable.svg, wordmark.svg")
