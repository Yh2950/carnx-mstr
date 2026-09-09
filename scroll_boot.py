"""
CARN-X  --  scroll engine  (presentation only, no app logic)
===========================================================
Streamlit's ``st.html`` strips a raw ``<script>`` even with
``unsafe_allow_javascript=True``, so -- exactly like ``ltr_boot`` and
``brand_boot`` -- this module writes one small ``<script>`` into Streamlit's
static ``index.html`` (just before ``</body>``).  It drives, in the real page:

* a hair-thin gilt scroll-progress meter (``--cx-progress``);
* a slow parallax on the engraved backdrop (``--cx-plate``);
* a one-time reveal cascade: on the first page load each content block rises in
  as it enters the viewport.  After the first load settles, blocks render
  normally, so dragging a slider or switching a tab never re-animates the page.

Every visual rule lives in ``theme.py``'s stylesheet (guarded by ``html.cx-js``
and ``prefers-reduced-motion``).  If this script never runs, the page shows
everything in its finished state -- nothing depends on it.

Idempotent (marker-guarded, strip-and-replace) and self-healing (re-applied every
process start, so ``pip install -U streamlit`` can't quietly undo it).  Fully
exception-guarded: a cosmetic patch must never break app startup.
"""

from __future__ import annotations

import os
import re

_MARKER = "carnx-scroll-boot"

# Bare engine.  Waits for <body>, then for Streamlit's async content via a
# MutationObserver.  All state on window/documentElement so it survives reruns.
_ENGINE = r"""
(function(){
  function boot(){
   try{
    if(window.__cxScroll) return; window.__cxScroll=true;
    var d=document, r=d.documentElement;
    var reduce=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    r.classList.add('cx-js');

    // one full-bleed element for the section-change sweep
    var wipe=d.querySelector('.cx-wipe');
    if(!wipe){ wipe=d.createElement('div'); wipe.className='cx-wipe'; d.body.appendChild(wipe); }

    function scroller(){
      var c=[d.querySelector('[data-testid=stMain]'),
             d.querySelector('[data-testid=stAppViewContainer]'),
             d.scrollingElement, d.documentElement].filter(Boolean);
      for(var i=0;i<c.length;i++){ if(c[i].scrollHeight-c[i].clientHeight>40) return c[i]; }
      return d.scrollingElement||d.documentElement;
    }
    function onScroll(){
      var s=scroller();
      var mx=Math.max(1, s.scrollHeight - s.clientHeight);
      var p=Math.min(1, Math.max(0, (s.scrollTop||0)/mx));
      r.style.setProperty('--cx-progress', p.toFixed(4));
      r.style.setProperty('--cx-plate', (-42*p).toFixed(1)+'px');
    }
    var bound=null;
    function bindScroll(){
      var s=scroller();
      var t=(s===d.documentElement||s===d.scrollingElement)?window:s;
      if(bound===t){ onScroll(); return; }
      if(bound) bound.removeEventListener('scroll', onScroll);
      t.addEventListener('scroll', onScroll, {passive:true});
      bound=t; onScroll();
    }
    bindScroll();
    window.addEventListener('resize', bindScroll, {passive:true});

    if(reduce) return;

    // pointer parallax -- the plate leans away from the cursor (decorative only)
    var px=0, py=0, tick=false;
    window.addEventListener('pointermove', function(e){
      var w=window.innerWidth||1, h=window.innerHeight||1;
      px=((e.clientX/w)-0.5); py=((e.clientY/h)-0.5);
      if(tick) return; tick=true;
      requestAnimationFrame(function(){
        tick=false;
        r.style.setProperty('--cx-mx', (-px*14).toFixed(1)+'px');
        r.style.setProperty('--cx-my', (-py*12).toFixed(1)+'px');
        r.style.setProperty('--cx-amx', (px*24).toFixed(1)+'px');
        r.style.setProperty('--cx-amy', (py*22).toFixed(1)+'px');
      });
    }, {passive:true});

    var SEL='[data-testid=stMain] .block-container [data-testid=stElementContainer],'
          + '[data-testid=stMain] .block-container [data-testid=stHorizontalBlock],'
          + '[data-testid=stMain] .cx-hero';
    var io=new IntersectionObserver(function(ents){
      ents.forEach(function(e){
        if(e.isIntersecting){
          e.target.classList.remove('cx-hide');
          e.target.classList.add('cx-rev');
          io.unobserve(e.target);
        }
      });
    }, {rootMargin:'0px 0px -6% 0px', threshold:0.03});

    function pass(hide){
      d.querySelectorAll(SEL).forEach(function(el){
        if(el.classList.contains('cx-rev')||el.classList.contains('cx-seen')) return;
        el.classList.add('cx-seen');
        if(hide) el.classList.add('cx-hide');
        io.observe(el);
      });
    }

    var rt;
    function bumpReady(){
      if(window.__cxReady) return;
      clearTimeout(rt);
      rt=setTimeout(function(){ window.__cxReady=true; }, 1400);
    }
    setTimeout(function(){ window.__cxReady=true; }, 6000);
    setTimeout(function(){
      d.querySelectorAll('.cx-hide').forEach(function(el){ el.classList.add('cx-rev'); });
    }, 6800);

    // which menu item is active -> drive per-section backdrop + the sweep
    var lastSec=null;
    function navIndex(){
      var labels=d.querySelectorAll('.st-key-cx_nav [role=radiogroup] label');
      for(var i=0;i<labels.length;i++){
        var inp=labels[i].querySelector('input');
        if(inp && inp.checked) return i;
      }
      return -1;
    }
    function syncSection(){
      var i=navIndex();
      if(i<0) return;
      if(lastSec===null){ lastSec=i; r.style.setProperty('--cx-sec-i', i); r.dataset.cxSection=i; return; }
      if(i===lastSec) return;
      lastSec=i;
      r.style.setProperty('--cx-sec-i', i);
      r.dataset.cxSection=i;
      // the sweep
      wipe.classList.remove('run'); void wipe.offsetWidth; wipe.classList.add('run');
      // re-arm the reveal cascade for the "new page"
      window.__cxReady=false; clearTimeout(rt);
      pass(true); bumpReady();
      setTimeout(function(){ window.__cxReady=true;
        d.querySelectorAll('.cx-hide').forEach(function(el){
          var b=el.getBoundingClientRect();
          if(b.top < (window.innerHeight||0)*1.05) el.classList.add('cx-rev');
        });
      }, 5000);
    }

    pass(true); bumpReady(); syncSection();

    var mo=new MutationObserver(function(){
      syncSection();
      pass(!window.__cxReady);
      bumpReady();
      bindScroll();
    });
    mo.observe(d.body, {childList:true, subtree:true});
    setInterval(syncSection, 500);
   }catch(e){
    try{
      document.documentElement.classList.remove('cx-js');
      document.querySelectorAll('.cx-hide').forEach(function(el){ el.classList.remove('cx-hide'); });
    }catch(_){}
   }
  }
  if(document.body) boot();
  else document.addEventListener('DOMContentLoaded', boot, {once:true});
})();
"""

_SNIPPET = f"<script>/* {_MARKER} */{_ENGINE}</script>"


def _index_html_path() -> str | None:
    try:
        import streamlit
    except Exception:  # pragma: no cover
        return None
    p = os.path.join(os.path.dirname(streamlit.__file__), "static", "index.html")
    return p if os.path.isfile(p) else None


def apply() -> bool:
    """Patch static/index.html if needed.  Returns True when a write happened."""
    path = _index_html_path()
    if not path:
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            html = fh.read()
    except Exception:
        return False
    # drop any earlier carnx-scroll-boot script so a corrected snippet always wins
    cleaned = re.sub(
        r"\s*<script>/\* " + _MARKER + r" \*/.*?</script>", "", html, flags=re.S
    )
    if _SNIPPET in cleaned:
        patched = cleaned
    else:
        anchor = "</body>"
        i = cleaned.rfind(anchor)
        if i == -1:
            return False
        patched = cleaned[:i] + _SNIPPET + "\n" + cleaned[i:]
    if patched == html:
        return False
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(patched)
    except Exception:
        return False
    return True


try:
    _PATCHED = apply()
except Exception:  # a cosmetic patch must never break app startup
    _PATCHED = False
