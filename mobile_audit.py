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
  document.querySelectorAll('[data-testid="stMain"] *, [data-testid="stMainBlockContainer"] *').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return;
    if (el.closest('[data-testid="stSidebar"]')) return;
    // element sticks out past the right edge by > 2px, and isn't itself scrollable
    const overRight = r.right - vw;
    const overLeft = -r.left;
    if (overRight > 2 || overLeft > 2) {
      const style = getComputedStyle(el);
      const scrollable = /(auto|scroll|clip|hidden)/.test(style.overflowX);
      if (scrollable && el.scrollWidth > el.clientWidth) return; // intentional scroll area
      // overflow that an ancestor clips or scrolls is invisible to the user
      let anc = el.parentElement;
      let clipped = false;
      while (anc && !anc.matches('[data-testid="stMain"]')) {
        const as = getComputedStyle(anc);
        if (/(auto|scroll|clip|hidden)/.test(as.overflowX)) { clipped = true; break; }
        anc = anc.parentElement;
      }
      if (clipped) return;
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


def _sidebar_open(page) -> bool:
    return bool(
        page.evaluate(
            "() => { const s=document.querySelector('[data-testid=stSidebar]');"
            "return !!s && s.getAttribute('aria-expanded') === 'true'; }"
        )
    )


def _open_sidebar(page):
    if _sidebar_open(page):
        return True
    for sel in (
        '[data-testid="stExpandSidebarButton"]',
        '[data-testid="stExpandSidebarButton"] button',
        '[data-testid="stSidebarCollapsedControl"]',
    ):
        try:
            page.locator(sel).first.click(timeout=1500, force=True)
            page.wait_for_function(
                "() => document.querySelector('[data-testid=stSidebar]')"
                "?.getAttribute('aria-expanded') === 'true'",
                timeout=4000,
            )
            return True
        except Exception:
            continue
    return False


def _close_sidebar(page):
    if not _sidebar_open(page):
        return
    for sel in (
        '[data-testid="stSidebarCollapseButton"]',
        '[data-testid="stSidebarCollapseButton"] button',
    ):
        try:
            page.locator(sel).first.click(timeout=1500, force=True)
            page.wait_for_function(
                "() => document.querySelector('[data-testid=stSidebar]')"
                "?.getAttribute('aria-expanded') !== 'true'",
                timeout=4000,
            )
            return
        except Exception:
            continue


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

        # Fixed iPhone viewport throughout. The mobile sidebar is a fixed
        # overlay (see theme.py) so it does not reflow stMain -- we navigate
        # with it open, via a pure-JS click that needs no element geometry.
        radio = page.locator('[data-testid="stSidebar"] [data-testid="stRadioOption"]')
        radio.first.wait_for(state="attached", timeout=20000)

        for idx, sc in enumerate(SCREENS):
            t0 = time.time()
            try:
                _open_sidebar(page)
                radio.nth(idx).locator("input").evaluate("el => el.click()")
            except Exception as e:
                fails.append(f"{sc}: could not navigate ({type(e).__name__})")
                continue
            page.wait_for_timeout(1400)
            try:
                page.wait_for_selector(
                    '[data-testid="stStatusWidget"]', state="detached", timeout=45000
                )
            except Exception:
                pass
            page.wait_for_timeout(2800)
            _close_sidebar(page)
            page.wait_for_timeout(600)
            page.evaluate("window.scrollTo(0, 0)")

            res = page.evaluate(_OVERFLOW_JS)
            hs = res["horizontalScroll"]
            offs = res["offenders"]
            status = "ok  " if hs <= 2 and not offs else "FAIL"
            if status == "FAIL":
                detail = " | ".join(
                    f"{o['tid'] or o['tag']}[{o['cls']}] w={o['w']} over={o['over']} {o['text']!r}"
                    for o in offs[:7]
                )
                fails.append(f"{sc}: hscroll={hs}px :: {detail}")
            print(
                f"  [{status}] {sc:26s} hscroll={hs:>3}px  offenders={len(offs):>2}  ({time.time() - t0:.1f}s)"
            )
            if SHOT:
                safe = sc.replace(" ", "_").replace("/", "-")
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
