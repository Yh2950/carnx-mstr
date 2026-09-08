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

NAVY_0 = "#0B1222"       # plate centre
NAVY_1 = "#070C17"       # plate edge
GOLD = "#C6A052"         # antique gold — the linework
GOLD_HI = "#E9CD8B"      # engraved highlight
GOLD_LO = "#A07E37"      # engraved shadow
PATH_GOLD = "#D8BE83"    # the price path
DIM = "#8A94AC"

SERIF = "Spectral, 'EB Garamond', Georgia, 'Times New Roman', serif"
_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Spectral:ital,wght@0,400;0,500;1,400&display=swap');"
)


def _defs(uid: str) -> str:
    return f"""<defs>
    <radialGradient id="pl{uid}" cx="32%" cy="22%" r="98%">
      <stop offset="0" stop-color="{NAVY_0}"/><stop offset="1" stop-color="{NAVY_1}"/>
    </radialGradient>
    <linearGradient id="gd{uid}" x1="0.15" y1="0" x2="0.4" y2="1">
      <stop offset="0" stop-color="{GOLD_HI}"/>
      <stop offset="0.5" stop-color="{GOLD}"/>
      <stop offset="1" stop-color="{GOLD_LO}"/>
    </linearGradient>
  </defs>"""


def _scene(uid: str, scale: float = 1.0) -> str:
    """Everything on the 512 grid, scaled about the centre if asked."""
    # a de-trended close series — low amplitude, irregular step, sits on the
    # upper counter of the ∫
    pts = [
        (70, 238), (92, 232), (114, 244), (140, 222), (164, 233), (190, 216),
        (214, 230), (242, 240), (270, 220), (300, 234), (326, 226), (352, 246),
        (382, 231), (410, 250), (440, 238), (452, 242),
    ]
    path = "M" + " L".join(f"{x} {y}" for x, y in pts)
    bell = "M184 400 C222 400 232 332 262 332 C292 332 302 400 340 400"

    integral = (
        f'<text x="250" y="374" text-anchor="middle" font-family={SERIF!r} '
        f'font-weight="500" font-size="456" fill="url(#gd{uid})" '
        f'stroke="{NAVY_1}" stroke-width="9" paint-order="stroke">∫</text>'
    )
    limits = (
        f'<text x="338" y="176" font-family={SERIF!r} font-style="italic" '
        f'font-size="47" fill="{GOLD}" opacity="0.85">T</text>'
        f'<text x="150" y="408" font-family={SERIF!r} font-style="italic" '
        f'font-size="47" fill="{GOLD}" opacity="0.85">0</text>'
    )
    inner = f"""
    <path d="{path}" fill="none" stroke="{GOLD}" stroke-width="2.4"
          stroke-linejoin="round" stroke-linecap="round" opacity="0.42"/>
    <path d="{bell}" fill="none" stroke="{GOLD}" stroke-width="2.4" opacity="0.55"/>
    {integral}
    {limits}"""
    if scale != 1.0:
        inner = f'<g transform="translate(256 256) scale({scale}) translate(-256 -256)">{inner}</g>'
    return inner


def _frame(size: int, inset: float) -> str:
    b = size * inset
    return (
        f'<rect x="{b:.1f}" y="{b:.1f}" width="{size-2*b:.1f}" height="{size-2*b:.1f}" '
        f'rx="{size*0.014:.1f}" fill="none" stroke="{GOLD}" '
        f'stroke-width="{max(1.0, size*0.0032):.2f}" opacity="0.34"/>'
    )


def svg_app(size: int = 512) -> str:
    r = size * 0.235
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}"><style>{_FONT_IMPORT}</style>{_defs("a")}'
        f'<rect width="{size}" height="{size}" rx="{r:.1f}" fill="url(#pla)"/>'
        f'{_frame(size, 0.085)}'
        f'<g transform="scale({size/512})">{_scene("a")}</g></svg>'
    )


def svg_maskable(size: int = 512) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}"><style>{_FONT_IMPORT}</style>{_defs("m")}'
        f'<rect width="{size}" height="{size}" fill="url(#plm)"/>'
        f'<g transform="scale({size/512})">{_scene("m", scale=0.63)}</g></svg>'
    )


def svg_wordmark(w: int = 540, h: int = 128) -> str:
    m = h - 14
    r = m * 0.235
    mark = (
        f'<g transform="translate(7 7)">'
        f'<rect width="{m}" height="{m}" rx="{r:.1f}" fill="url(#plw)"/>'
        f'{_frame(m, 0.085)}'
        f'<g transform="scale({m/512})">{_scene("w", scale=0.92)}</g></g>'
    )
    tx = h + 8
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <style>{_FONT_IMPORT}</style>{_defs("w")}
  {mark}
  <text x="{tx}" y="{h*0.50:.0f}" font-family={SERIF!r} font-weight="500"
        font-size="{h*0.37:.0f}" letter-spacing="0.5" fill="{GOLD_HI}">CARN<tspan fill="{GOLD_LO}" dx="1">-</tspan>X</text>
  <line x1="{tx+1}" y1="{h*0.61:.0f}" x2="{w-16}" y2="{h*0.61:.0f}" stroke="{GOLD}" stroke-width="1" opacity="0.35"/>
  <text x="{tx+1}" y="{h*0.80:.0f}" font-family="Inter, system-ui, sans-serif"
        font-size="{h*0.125:.0f}" letter-spacing="0.4" fill="{DIM}">probabilistic forecasting instrument</text>
</svg>"""


if __name__ == "__main__":
    import pathlib

    d = pathlib.Path(__file__).parent
    (d / "icon_app.svg").write_text(svg_app(512), encoding="utf-8")
    (d / "icon_maskable.svg").write_text(svg_maskable(512), encoding="utf-8")
    (d / "wordmark.svg").write_text(svg_wordmark(), encoding="utf-8")
    print("wrote icon_app.svg, icon_maskable.svg, wordmark.svg")
