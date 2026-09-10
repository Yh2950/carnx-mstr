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
    var flash=d.querySelector('.cx-flash');
    if(!flash){ flash=d.createElement('div'); flash.className='cx-flash'; d.body.appendChild(flash); }

    // the storm: ambient bolts at the lagoon edges + a strike on every click
    if(!d.querySelector('.cx-storm')){
      var storm=d.createElement('div'); storm.className='cx-storm';
      storm.innerHTML='<i class="b1"></i><i class="b2"></i><i class="b3"></i>';
      d.body.appendChild(storm);
    }
    var strikeW=d.querySelector('.cx-strike-wrap');
    if(!strikeW){
      strikeW=d.createElement('div'); strikeW.className='cx-strike-wrap';
      strikeW.innerHTML='<div class="cx-strike"></div><div class="cx-strike-bolt"></div>';
      d.body.appendChild(strikeW);
    }
    var strikeT=null, lastStrike=0;
    function strike(x,y){
      var now=Date.now(); if(now-lastStrike<140) return; lastStrike=now;
      strikeW.style.setProperty('--sx', (x|0)+'px');
      strikeW.style.setProperty('--sy', (y|0)+'px');
      strikeW.classList.remove('on'); void strikeW.offsetWidth; strikeW.classList.add('on');
      if(navigator.vibrate) navigator.vibrate(8);
      clearTimeout(strikeT);
      strikeT=setTimeout(function(){ strikeW.classList.remove('on'); }, 460);
    }
    if(!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)){
      d.addEventListener('pointerdown', function(e){
        if(e.button && e.button!==0) return;
        strike(e.clientX, e.clientY);
      }, {passive:true, capture:true});
    }

    function scroller(){
      var c=[d.querySelector('[data-testid=stMain]'),
             d.querySelector('[data-testid=stAppViewContainer]'),
             d.scrollingElement, d.documentElement].filter(Boolean);
      for(var i=0;i<c.length;i++){ if(c[i].scrollHeight-c[i].clientHeight>40) return c[i]; }
      return d.scrollingElement||d.documentElement;
    }
    // cache scroll geometry -- NEVER read layout inside the scroll handler
    var _sc=null, _rng=1, _spending=false, _scrolling=0;
    function measure(){
      _sc=scroller();
      _rng=Math.max(1, _sc.scrollHeight - _sc.clientHeight);
    }
    function paintScroll(){
      _spending=false;
      var top=_sc?(_sc.scrollTop||0):0;
      var p=Math.min(1, Math.max(0, top/_rng));
      r.style.setProperty('--cx-progress', p.toFixed(4));
      r.style.setProperty('--cx-plate', (-42*p).toFixed(1)+'px');
    }
    function onScroll(){
      _scrolling=Date.now();
      if(_spending) return;
      _spending=true; requestAnimationFrame(paintScroll);
    }
    var bound=null;
    function bindScroll(){
      var s=scroller();
      var t=(s===d.documentElement||s===d.scrollingElement)?window:s;
      measure();
      if(bound===t){ paintScroll(); return; }
      if(bound) bound.removeEventListener('scroll', onScroll);
      t.addEventListener('scroll', onScroll, {passive:true});
      bound=t; paintScroll();
    }
    bindScroll();
    window.addEventListener('resize', function(){ measure(); paintScroll(); }, {passive:true});

    if(reduce) return;

    // pointer parallax -- fine pointers only (pointless + costly on touch);
    // suppressed while the page is scrolling so it can't add work to a scroll frame.
    var fine=!window.matchMedia||window.matchMedia('(pointer: fine)').matches;
    if(fine){
      var px=0, py=0, tick=false;
      window.addEventListener('pointermove', function(e){
        if(Date.now()-_scrolling<220) return;
        var w=window.innerWidth||1, h=window.innerHeight||1;
        px=((e.clientX/w)-0.5); py=((e.clientY/h)-0.5);
        if(tick) return; tick=true;
        requestAnimationFrame(function(){
          tick=false;
          r.style.setProperty('--cx-mx', (-px*24).toFixed(1)+'px');
          r.style.setProperty('--cx-my', (-py*20).toFixed(1)+'px');
        });
      }, {passive:true});
    }

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
      if(lastSec===null){ lastSec=i; r.style.setProperty('--cx-sec-i', i); r.dataset.cxSection=i; orbitSync(i); return; }
      if(i===lastSec){ orbitSync(i); return; }
      lastSec=i;
      r.style.setProperty('--cx-sec-i', i);
      r.dataset.cxSection=i;
      // the section-change sweep + hue flash (transform/opacity only)
      wipe.classList.remove('run'); void wipe.offsetWidth; wipe.classList.add('run');
      flash.classList.remove('run'); void flash.offsetWidth; flash.classList.add('run');
      if(navigator.vibrate) navigator.vibrate([4,18,8]);
      // re-arm the reveal cascade for the "new page"
      window.__cxReady=false; clearTimeout(rt);
      pass(true); bumpReady(); orbitSync(i);
      setTimeout(function(){ window.__cxReady=true;
        d.querySelectorAll('.cx-hide').forEach(function(el){
          var b=el.getBoundingClientRect();
          if(b.top < (window.innerHeight||0)*1.05) el.classList.add('cx-rev');
        });
      }, 5000);
    }


    /* ---- orbital navigation: page buttons on a ring around the Bitcoin hub ---- */
    var BTC='<img src="./carnx/hub.png" alt="" '
      +'style="width:100%;height:100%;display:block;border-radius:50%;object-fit:cover">';

    function navLabels(){ return d.querySelectorAll('.st-key-cx_nav [role=radiogroup] label'); }
    function navNames(){
      var out=[], L=navLabels();
      for(var i=0;i<L.length;i++){
        var pn=L[i].querySelector('[data-testid=stMarkdownContainer] p') || L[i];
        out.push((pn.textContent||'').trim());
      }
      return out;
    }
    function shortName(n){
      n=(n||'').trim();
      if(/^Monte/i.test(n)) return 'Monte';
      if(/Walk-Forward/i.test(n)) return 'Walk-Fwd';
      var w=n.split(/[ ·–—>]/).filter(Boolean);
      return w[0]||n;
    }

    var orbit=d.querySelector('.cx-orbit'), oRing, oHub, oItems=[];
    var oSel=0, oN=0, STEP=32.72, dragging=false, navT=null, rot=0;

    function buildOrbit(){
      var names=navNames();
      if(names.length<2) return false;
      oN=names.length; STEP=360/oN;
      if(!orbit){
        orbit=d.createElement('div'); orbit.className='cx-orbit';
        orbit.innerHTML='<div class="cx-orbit-ring"></div>'
          +'<div class="cx-orbit-hub">'+BTC+'</div>';
        var mast=d.querySelector('.cx-mast');
        var row=mast && mast.closest('[data-testid=stHorizontalBlock]');
        var host=(row && row.closest('[data-testid=stElementContainer]')) || row;
        if(host && host.parentElement) host.parentElement.insertBefore(orbit, host.nextSibling);
        else (d.querySelector('[data-testid=stMain] .block-container [data-testid=stVerticalBlock]')||d.body).prepend(orbit);
        oRing=orbit.querySelector('.cx-orbit-ring');
        oHub=orbit.querySelector('.cx-orbit-hub');
        bindOrbit();
      }
      if(oItems.length!==oN){
        oRing.innerHTML=''; oItems=[];
        for(var i=0;i<oN;i++){
          var it=d.createElement('div'); it.className='cx-orbit-item';
          it.style.setProperty('--a', (i*STEP)+'deg');
          var bt=d.createElement('button'); bt.type='button';
          bt.textContent=shortName(names[i]);
          (function(idx){ bt.addEventListener('click', function(ev){
            ev.stopPropagation();
            if(!dragging){ oSel=idx; renderOrbit(idx); goTo(idx,true); }
          }); })(i);
          it.appendChild(bt); oRing.appendChild(it); oItems.push(it);
        }
      }
      renderOrbit(oSel);
      return true;
    }

    function renderOrbit(sel){
      if(!oRing) return;
      var names=navNames(); if(names.length!==oN){ buildOrbit(); return; }
      oSel=Math.max(0,Math.min(oN-1,sel|0));
      rot = -oSel*STEP;
      orbit.style.setProperty('--orbit-rot', rot+'deg');
      for(var i=0;i<oItems.length;i++) oItems[i].classList.toggle('on', i===oSel);
    }

    function goTo(sel, now){
      sel=Math.max(0,Math.min(oN-1,sel|0));
      var fire=function(){
        var L=navLabels();
        if(L[sel] && !L[sel].querySelector('input:checked')) L[sel].click();
      };
      clearTimeout(navT);
      if(now) fire(); else navT=setTimeout(fire, 200);
    }

    function bindOrbit(){
      var aPrev=0, sel0=0, acc=0, hoverT=null;
      function ang(e){
        var t=e.touches?e.touches[0]:e, R=orbit.getBoundingClientRect();
        return Math.atan2(t.clientY-(R.top+R.height/2), t.clientX-(R.left+R.width/2))*180/Math.PI;
      }
      function down(e){
        dragging=true; orbit.classList.remove('spinning'); orbit.classList.add('dragging','touched');
        aPrev=ang(e); sel0=oSel; acc=0;
      }
      function move(e){
        if(!dragging) return;
        var a=ang(e), da=a-aPrev;
        if(da>180) da-=360; else if(da<-180) da+=360;
        acc+=da; aPrev=a;
        orbit.style.setProperty('--orbit-rot', ((-sel0*STEP)+acc)+'deg');
        var tgt=Math.max(0,Math.min(oN-1, Math.round(sel0 - acc/STEP)));
        if(tgt!==oSel){
          oSel=tgt;
          for(var i=0;i<oItems.length;i++) oItems[i].classList.toggle('on', i===oSel);
          if(navigator.vibrate) navigator.vibrate(5);
          goTo(oSel, false);
        }
        if(e.cancelable) e.preventDefault();
      }
      function up(){
        if(!dragging) return;
        dragging=false; orbit.classList.remove('dragging');
        renderOrbit(oSel); goTo(oSel, true);
      }
      orbit.addEventListener('touchstart', down, {passive:true});
      orbit.addEventListener('touchmove', move, {passive:false});
      orbit.addEventListener('touchend', up);
      orbit.addEventListener('touchcancel', up);
      orbit.addEventListener('pointerdown', function(e){ if(e.pointerType!=='touch') down(e); });
      window.addEventListener('pointermove', function(e){ if(e.pointerType!=='touch') move(e); });
      window.addEventListener('pointerup', function(e){ if(e.pointerType!=='touch') up(e); });

      // hover -> the ring spins; leave -> it eases back to rest
      orbit.addEventListener('pointerenter', function(e){
        if(e.pointerType==='touch') return;
        clearTimeout(hoverT); orbit.classList.add('spinning','touched');
      });
      orbit.addEventListener('pointerleave', function(e){
        if(e.pointerType==='touch') return;
        clearTimeout(hoverT);
        hoverT=setTimeout(function(){ orbit.classList.remove('spinning'); renderOrbit(oSel); }, 120);
      });
      // long-press on touch also spins
      orbit.addEventListener('touchstart', function(){
        clearTimeout(hoverT);
        hoverT=setTimeout(function(){ if(!dragging) orbit.classList.add('spinning'); }, 420);
      }, {passive:true});
      orbit.addEventListener('touchend', function(){
        clearTimeout(hoverT);
        setTimeout(function(){ orbit.classList.remove('spinning'); renderOrbit(oSel); }, 200);
      });

      orbit.addEventListener('wheel', function(e){
        e.preventDefault();
        var n=oSel + (e.deltaY>0?1:-1);
        if(n<0) n=oN-1; else if(n>=oN) n=0;
        renderOrbit(n); goTo(n, false);
      }, {passive:false});
    }

    function orbitSync(i){
      if(dragging) return;
      if(!orbit && !buildOrbit()) return;
      if(i!==oSel) renderOrbit(i);
    }

    pass(true); bumpReady(); syncSection();
    buildOrbit();

    // Streamlit reruns fire a storm of mutations (every slider tick) -- coalesce
    // to one rAF-gated pass so the observer can't compete with scroll/interaction.
    var moPend=false;
    function moFlush(){
      moPend=false;
      syncSection();
      pass(!window.__cxReady);
      bumpReady();
      bindScroll();
    }
    var mo=new MutationObserver(function(){
      if(moPend) return;
      moPend=true;
      (window.requestIdleCallback||window.requestAnimationFrame)(moFlush);
    });
    mo.observe(d.body, {childList:true, subtree:true});
    setInterval(syncSection, 600);
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
