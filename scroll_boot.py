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
    var scT;
    function onScroll(){
      var s=scroller();
      var mx=Math.max(1, s.scrollHeight - s.clientHeight);
      var p=Math.min(1, Math.max(0, (s.scrollTop||0)/mx));
      r.style.setProperty('--cx-progress', p.toFixed(4));
      r.style.setProperty('--cx-plate', (-42*p).toFixed(1)+'px');
      if(wheel){ wheel.classList.add('scrolling'); clearTimeout(scT);
        scT=setTimeout(function(){ wheel.classList.remove('scrolling'); }, 550); }
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
      if(lastSec===null){ lastSec=i; r.style.setProperty('--cx-sec-i', i); r.dataset.cxSection=i; wheelSync(i); return; }
      if(i===lastSec){ wheelSync(i); return; }
      lastSec=i;
      r.style.setProperty('--cx-sec-i', i);
      r.dataset.cxSection=i;
      // the sweep
      wipe.classList.remove('run'); void wipe.offsetWidth; wipe.classList.add('run');
      // re-arm the reveal cascade for the "new page"
      window.__cxReady=false; clearTimeout(rt);
      pass(true); bumpReady(); wheelSync(i);
      setTimeout(function(){ window.__cxReady=true;
        d.querySelectorAll('.cx-hide').forEach(function(el){
          var b=el.getBoundingClientRect();
          if(b.top < (window.innerHeight||0)*1.05) el.classList.add('cx-rev');
        });
      }, 5000);
    }


    /* ---- mobile navigation dial (iOS camera-zoom-wheel style) ---- */
    var BTC='<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">'
      +'<circle cx="32" cy="32" r="31" fill="#F7931A"/>'
      +'<text x="32" y="47" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" '
      +'font-weight="700" font-size="40" fill="#fff" transform="rotate(-13 32 32)">B</text>'
      +'<rect x="27" y="10" width="3.6" height="9" fill="#fff" transform="rotate(-13 32 32)"/>'
      +'<rect x="27" y="45" width="3.6" height="9" fill="#fff" transform="rotate(-13 32 32)"/>'
      +'<rect x="35" y="10" width="3.6" height="9" fill="#fff" transform="rotate(-13 32 32)"/>'
      +'<rect x="35" y="45" width="3.6" height="9" fill="#fff" transform="rotate(-13 32 32)"/></svg>';

    function navLabels(){ return d.querySelectorAll('.st-key-cx_nav [role=radiogroup] label'); }
    function navNames(){
      var out=[], L=navLabels();
      for(var i=0;i<L.length;i++){
        var pnode=L[i].querySelector('[data-testid=stMarkdownContainer] p') || L[i];
        out.push((pnode.textContent||'').trim());
      }
      return out;
    }

    var wheel=d.querySelector('.cx-wheel'), ticks, reelPrev, reelCur, reelNext, dial;
    var wSel=0, wN=0, STEP=32.72, dragging=false, navT=null;

    function buildWheel(){
      var names=navNames();
      if(names.length<2){ return false; }
      wN=names.length; STEP=360/wN;
      if(!wheel){
        wheel=d.createElement('div'); wheel.className='cx-wheel';
        wheel.innerHTML=
          '<div class="cx-wheel-reel"><span class="adj prev"></span>'
         +'<span class="cur"></span><span class="adj next"></span></div>'
         +'<div class="cx-dial"><div class="cx-dial-ticks"></div>'
         +'<div class="cx-dial-hub">'+BTC+'</div></div>'
         +'<div class="cx-wheel-hint">סובב לניווט</div>';
        d.body.appendChild(wheel);
        dial=wheel.querySelector('.cx-dial');
        ticks=wheel.querySelector('.cx-dial-ticks');
        reelPrev=wheel.querySelector('.prev');
        reelCur=wheel.querySelector('.cur');
        reelNext=wheel.querySelector('.next');
        bindDial();
      }
      if(ticks.children.length!==wN){
        ticks.innerHTML='';
        for(var i=0;i<wN;i++){
          var t=d.createElement('i');
          t.style.transform='rotate('+(i*STEP)+'deg)';
          ticks.appendChild(t);
        }
      }
      renderWheel(wSel);
      return true;
    }

    function renderWheel(sel){
      if(!ticks) return;
      var names=navNames(); if(names.length!==wN){ buildWheel(); return; }
      wSel=Math.max(0,Math.min(wN-1,sel|0));
      reelCur.textContent=names[wSel]||'';
      reelPrev.textContent=wSel>0?names[wSel-1]:'';
      reelNext.textContent=wSel<wN-1?names[wSel+1]:'';
      for(var i=0;i<ticks.children.length;i++) ticks.children[i].classList.toggle('on', i===wSel);
      ticks.style.transform='rotate('+(-wSel*STEP)+'deg)';
    }

    function goTo(sel, now){
      sel=Math.max(0,Math.min(wN-1,sel|0));
      var fire=function(){
        var L=navLabels();
        if(L[sel] && !L[sel].querySelector('input:checked')) L[sel].click();
      };
      clearTimeout(navT);
      if(now) fire(); else navT=setTimeout(fire, 190);
    }

    function bindDial(){
      var a0=0, sel0=0, acc=0, aPrev=0;
      function ang(e){
        var t=e.touches?e.touches[0]:e, R=dial.getBoundingClientRect();
        return Math.atan2(t.clientY-(R.top+R.height/2), t.clientX-(R.left+R.width/2))*180/Math.PI;
      }
      function down(e){
        dragging=true; dial.classList.add('spin'); wheel.classList.add('touched');
        a0=aPrev=ang(e); sel0=wSel; acc=0;
        e.preventDefault();
      }
      function move(e){
        if(!dragging) return;
        var a=ang(e), da=a-aPrev;
        if(da>180) da-=360; else if(da<-180) da+=360;
        acc+=da; aPrev=a;
        ticks.style.transform='rotate('+((-sel0*STEP)-acc)+'deg)';
        var tgt=Math.max(0,Math.min(wN-1, Math.round(sel0 + acc/STEP)));
        if(tgt!==wSel){
          wSel=tgt;
          var names=navNames();
          reelCur.textContent=names[wSel]||'';
          reelPrev.textContent=wSel>0?names[wSel-1]:'';
          reelNext.textContent=wSel<wN-1?names[wSel+1]:'';
          for(var i=0;i<ticks.children.length;i++) ticks.children[i].classList.toggle('on', i===wSel);
          if(navigator.vibrate) navigator.vibrate(6);
          goTo(wSel, false);
        }
        e.preventDefault();
      }
      function up(){
        if(!dragging) return;
        dragging=false; dial.classList.remove('spin');
        renderWheel(wSel);
        goTo(wSel, true);
      }
      dial.addEventListener('touchstart', down, {passive:false});
      dial.addEventListener('touchmove', move, {passive:false});
      dial.addEventListener('touchend', up);
      dial.addEventListener('touchcancel', up);
      dial.addEventListener('pointerdown', function(e){ if(e.pointerType!=='touch'){ down(e); } });
      window.addEventListener('pointermove', function(e){ if(e.pointerType!=='touch') move(e); });
      window.addEventListener('pointerup', function(e){ if(e.pointerType!=='touch') up(e); });
      reelPrev.addEventListener('click', function(){ var n=wSel-1; if(n>=0){ renderWheel(n); goTo(n,true);} });
      reelNext.addEventListener('click', function(){ var n=wSel+1; if(n<wN){ renderWheel(n); goTo(n,true);} });
      reelPrev.style.pointerEvents='auto'; reelNext.style.pointerEvents='auto'; reelPrev.style.cursor='pointer'; reelNext.style.cursor='pointer';
      dial.addEventListener('wheel', function(e){
        e.preventDefault();
        goTo(wSel + (e.deltaY>0?1:-1), false);
        renderWheel(wSel + (e.deltaY>0?1:-1));
      }, {passive:false});
    }

    function wheelSync(i){
      if(dragging) return;
      if(!wheel && !buildWheel()) return;
      if(i!==wSel) renderWheel(i);
    }

    pass(true); bumpReady(); syncSection();
    buildWheel();

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
