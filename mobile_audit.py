"""
CARN-X  --  mobile-responsiveness audit
======================================
Drives the running Streamlit app in a real headless browser at an iPhone
viewport and fails on horizontal overflow / elements wider than the screen.

    # start the app on some port, then:
    .venv/bin/python mobile_audit.py --port 8599 [--shot]

Exit 0 == every screen fits an iPhone with no sideways scroll.
"""

from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

PORT = 8599
SHOT = "--shot" in sys.argv
for i, a in enumerate(sys.argv):
    if a == "--port" and i + 1 < len(sys.argv):
        PORT = int(sys.argv[i + 1])

# iPhone 12/13/14 CSS viewport
VIEWPORT = {"width": 390, "height": 844}
DEVICE_SCALE = 3
UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KhtML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
)

SCREENS = [
    "סקירה",
    "טרמינל מסחר",
    "מחשבון הסתברויות",
    "אבחון סטטיסטי",
    "תחזית הסתברותית",
    "Monte Carlo",
    "מבנים מתמטיים",
    "מחזור ביטקוין → MSTR",
    "ראיות Walk-Forward",
    "סיכון ומינוף",
    "הגדרות",
]

_OVERFLOW_JS = r"""
() => {
  const vw = document.documentElement.clientWidth;
  const doc = document.documentElement;
  const bad = [];
  const seen = new Set();
  document.querySelectorAll('body *').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return;
    // element sticks out past the right edge by > 2px, and isn't itself scrollable
    const overRight = r.right - vw;
    const overLeft = -r.left;
    if (overRight > 2 || overLeft > 2) {
      const style = getComputedStyle(el);
      const scrollable = /(auto|scroll)/.test(style.overflowX);
      if (scrollable && el.scrollWidth > el.clientWidth) return; // intentional scroll area
      const tag = el.tagName.toLowerCase();
      const cls = (el.getAttribute('class') || '').split(' ').slice(0, 2).join('.');
      const tid = el.getAttribute('data-testid') || '';
      const key = tag + '|' + tid + '|' + cls + '|' + Math.round(r.width);
      if (seen.has(key)) return;
      seen.add(key);
      bad.push({ tag, tid, cls, w: Math.round(r.width), right: Math.round(r.right),
                 over: Math.round(Math.max(overRight, overLeft)),
                 text: (el.innerText || '').slice(0, 40).replace(/\s+/g, ' ') });
    }
  });
  return {
    scrollW: doc.scrollWidth, clientW: doc.clientWidth,
    horizontalScroll: doc.scrollWidth - doc.clientWidth,
    offenders: bad.sort((a, b) => b.over - a.over).slice(0, 12),
  };
};
"""


def _open_sidebar(page):
    for sel in (
        '[data-testid="stSidebarCollapseButton"] button',
        '[data-testid="stSidebarCollapseButton"]',
        '[data-testid="stExpandSidebarButton"]',
        '[data-testid="collapsedControl"]',
        'button[kind="header"]',
    ):
        try:
            el = page.locator(sel).first
            if el.is_visible(timeout=400):
                el.click()
                page.wait_for_timeout(350)
                return True
        except Exception:
            pass
    return False


def _sidebar_visible(page) -> bool:
    try:
        return page.locator('[data-testid="stSidebar"] [role="radiogroup"]').first.is_visible(
            timeout=400
        )
    except Exception:
        return False


def main() -> int:
    url = f"http://localhost:{PORT}/"
    fails = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path="/usr/bin/google-chrome")
        ctx = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=DEVICE_SCALE,
            user_agent=UA,
            is_mobile=True,
            has_touch=True,
        )
        page = ctx.new_page()
        page.set_default_timeout(45000)
        page.goto(url, wait_until="domcontentloaded")
        # wait for first render
        page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=45000)
        page.wait_for_timeout(3500)

        for sc in SCREENS:
            t0 = time.time()
            if not _sidebar_visible(page):
                _open_sidebar(page)
            try:
                page.get_by_text(sc, exact=True).first.click(timeout=8000)
            except Exception as e:
                fails.append(f"{sc}: could not navigate ({type(e).__name__})")
                continue
            # let the screen compute (models, sims, iframes)
            page.wait_for_timeout(1200)
            try:
                page.wait_for_selector(
                    '[data-testid="stStatusWidget"]', state="detached", timeout=40000
                )
            except Exception:
                pass
            page.wait_for_timeout(2500)
            # collapse the sidebar so we measure the real content width
            if _sidebar_visible(page):
                _open_sidebar(page)
                page.wait_for_timeout(400)
            page.evaluate("window.scrollTo(0, 0)")

            res = page.evaluate(_OVERFLOW_JS)
            hs = res["horizontalScroll"]
            offs = res["offenders"]
            status = "ok  " if hs <= 2 and not offs else "FAIL"
            if status == "FAIL":
                detail = " | ".join(
                    f"{o['tid'] or o['tag']}.{o['cls']} w={o['w']} over={o['over']} “{o['text']}”"
                    for o in offs[:6]
                )
                fails.append(f"{sc}: hscroll={hs}px  ::  {detail}")
            print(
                f"  [{status}] {sc:26s} hscroll={hs:>3}px  offenders={len(offs)}  ({time.time() - t0:.1f}s)"
            )
            if SHOT:
                safe = sc.replace(" ", "_").replace("→", "to").replace("/", "-")
                page.screenshot(
                    path=f"/tmp/claude-1000/-home-yedidyahkim-Desktop/c19fc0c5-5629-40d4-9627-9e9a76b4ffca/scratchpad/m_{safe}.png",
                    full_page=True,
                )

        browser.close()

    print("\n" + "=" * 66)
    if fails:
        print(f"MOBILE AUDIT: {len(fails)} screen(s) overflow the iPhone viewport\n")
        for f in fails:
            print("  ✗ " + f)
        return 1
    print(f"MOBILE AUDIT: ALL {len(SCREENS)} screens fit a 390px iPhone viewport ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
