"""Render the brand SVGs to exact-size PNGs with headless Chromium, wait for the
web font, and build a review contact sheet at real icon sizes."""
import pathlib
from playwright.sync_api import sync_playwright
import brandmark as B

OUT = pathlib.Path(__file__).parent

# (filename, svg, css px)  -- PNG is written at exactly this pixel size
ICONS = [
    ("icon_512.png",          B.svg_app(512),      512),
    ("icon_192.png",          B.svg_app(192),      192),
    ("icon_180.png",          B.svg_app(180),      180),   # apple-touch
    ("icon_maskable_512.png", B.svg_maskable(512), 512),
    ("favicon_32.png",        B.svg_app(32),        32),
    ("favicon_64.png",        B.svg_app(64),        64),
]


def _page(pw, w, h, dsf):
    b = pw.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_context(device_scale_factor=dsf, viewport={"width": w, "height": h}).new_page()
    return b, pg


def render_exact(svg, size):
    with sync_playwright() as pw:
        b, pg = _page(pw, size, size, 1)
        pg.set_content(f'<!doctype html><meta charset=utf8><style>*{{margin:0}}</style>{svg}')
        pg.wait_for_timeout(2600)  # web font
        png = pg.screenshot(clip={"x": 0, "y": 0, "width": size, "height": size})
        b.close()
    return png


def contact_sheet():
    tiles = [64, 96, 128, 180, 256]
    cells = "".join(
        f'<div style="text-align:center"><div>{B.svg_app(s)}</div>'
        f'<div style="font:12px Inter,sans-serif;color:#8A94AC;margin-top:6px">{s}px</div></div>'
        for s in tiles
    )
    word = B.svg_wordmark()
    mask = f'<div style="text-align:center"><div style="border-radius:50%;overflow:hidden;width:180px;height:180px">{B.svg_maskable(180)}</div><div style="font:12px Inter;color:#8A94AC">maskable ∩ circle</div></div>'
    html = f"""<!doctype html><meta charset=utf8>
    <div style="background:#05070E;padding:40px;display:flex;flex-direction:column;gap:34px;width:1000px">
      <div style="display:flex;gap:26px;align-items:flex-end">{cells}</div>
      <div style="display:flex;gap:40px;align-items:center">{mask}
        <div style="background:#0B1222;padding:18px 22px;border-radius:12px">{word}</div></div>
    </div>"""
    with sync_playwright() as pw:
        b, pg = _page(pw, 1080, 640, 2)
        pg.set_content(html)
        pg.wait_for_timeout(2800)
        (OUT / "review_contact.png").write_bytes(pg.screenshot(full_page=True))
        b.close()


if __name__ == "__main__":
    for name, svg, size in ICONS:
        (OUT / name).write_bytes(render_exact(svg, size))
        print("wrote", name)
    contact_sheet()
    print("wrote review_contact.png")
