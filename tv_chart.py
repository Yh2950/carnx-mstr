"""
CARN-X  --  institutional charting terminal  (presentation only)
==============================================================
``price_chart_html(df, ...)`` builds a self-contained HTML document that renders
a professional, multi-pane, interactive price chart with TradingView's
open-source ``lightweight-charts`` engine.  Rendered in the app via
``st.iframe(price_chart_html(...), height=...)``.

Everything the chart can do -- candles / Heikin-Ashi / line / area, log /
linear / percent scale, EMA · SMA · Bollinger · VWAP overlays, RSI + MACD
sub-panes (time-synced), the model's Monte-Carlo forecast cone drawn forward
from the last bar, event markers, price lines, a normalised comparison series,
and a live OHLC + indicator readout -- is done **inside the iframe in
JavaScript**, so toolbar toggles are instant and never lose zoom/pan state.

Nothing here computes a model quantity.  The forecast cone, if shown, is passed
in already-computed by ``inference.monte_carlo_paths``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import pandas as pd

_LWC_CDN = "https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"

_DARK = dict(
    bg="#0A0F26",
    grid="rgba(150,170,255,0.055)",
    border="rgba(150,170,255,0.14)",
    text="#8B93B8",
    strong="#EAF2FF",
    panel="#0C1130",
    up="#26C281",
    down="#F0616D",
    accent="#38E8FF",
    violet="#A855F7",
    amber="#FBBF24",
    pink="#F472B6",
    wm="rgba(150,170,255,0.045)",
)
_LIGHT = dict(
    bg="#FFFFFF",
    grid="rgba(20,25,45,0.06)",
    border="rgba(20,25,45,0.12)",
    text="#5B6178",
    strong="#141826",
    panel="#F6F7FB",
    up="#0E9F6E",
    down="#E02424",
    accent="#0E7490",
    violet="#7C3AED",
    amber="#B45309",
    pink="#BE185D",
    wm="rgba(20,25,45,0.05)",
)


def price_chart_html(
    df: pd.DataFrame,
    *,
    date: str = "date",
    o: str = "open",
    h: str = "high",
    l: str = "low",
    c: str = "close",
    volume: str | None = "volume",
    title: str = "MSTR",
    subtitle: str = "",
    height: int = 560,
    dark: bool = True,
    chart_type: str = "candles",  # candles | heikin | line | area
    scale: str = "linear",  # linear | log | percent
    log_scale: bool = False,  # back-compat alias for scale="log"
    mas: Sequence[int] = (),  # back-compat: shown as SMA overlays
    overlays: Sequence[str] | None = None,  # subset of EMA21 EMA50 SMA200 BB VWAP
    panes: Sequence[str] = (),  # subset of: rsi, macd
    toolbar: bool = True,
    forecast: Mapping | None = None,  # {"dates":[...],"p05":[...],...,"p95":[...]}
    events: Sequence[Mapping] | None = None,
    markers: Sequence[Mapping] | None = None,  # back-compat alias for events
    price_lines: Sequence[Mapping] | None = None,
    compare: Mapping | None = None,  # {"title":str,"series":pd.Series}
) -> str:
    pal = _DARK if dark else _LIGHT
    if log_scale:
        scale = "log"

    d = df.copy()
    d[date] = pd.to_datetime(d[date])
    d = d.dropna(subset=[o, h, l, c]).sort_values(date)
    d = d[~d[date].duplicated(keep="last")]
    tstr = d[date].dt.strftime("%Y-%m-%d").tolist()

    candles = [
        {
            "time": t,
            "open": round(float(a), 4),
            "high": round(float(b), 4),
            "low": round(float(x), 4),
            "close": round(float(y), 4),
        }
        for t, a, b, x, y in zip(tstr, d[o], d[h], d[l], d[c])
    ]
    vols = []
    if volume and volume in d.columns:
        vv = d[volume].fillna(0.0)
        vols = [
            {"time": t, "value": float(x), "color": (pal["up"] if cc >= oo else pal["down"]) + "40"}
            for t, x, cc, oo in zip(tstr, vv, d[c], d[o])
        ]

    ov = list(overlays) if overlays is not None else []
    for n in mas:  # back-compat: (20,50,200) -> SMA overlays
        tag = f"SMA{int(n)}"
        if tag not in ov:
            ov.append(tag)

    ev = list(events or markers or [])

    cmp_payload = None
    if compare is not None and compare.get("series") is not None:
        s = pd.Series(compare["series"]).dropna()
        s = s.reindex(d[date].values).ffill().dropna()
        if len(s) > 2:
            base = float(s.iloc[0])
            anchor = float(d[c].iloc[0])
            cmp_payload = {
                "title": compare.get("title", "compare"),
                "data": [
                    {
                        "time": pd.Timestamp(ix).strftime("%Y-%m-%d"),
                        "value": round(anchor * float(v) / base, 4),
                    }
                    for ix, v in s.items()
                ],
            }

    fc = None
    if forecast and forecast.get("dates"):
        keys = [k for k in ("p05", "p25", "p50", "p75", "p95") if k in forecast]
        fc = {
            "dates": list(forecast["dates"]),
            **{k: [round(float(x), 4) for x in forecast[k]] for k in keys},
        }

    payload = json.dumps(
        {
            "candles": candles,
            "vols": vols,
            "overlays": ov,
            "panes": list(panes),
            "events": ev,
            "priceLines": list(price_lines or []),
            "compare": cmp_payload,
            "forecast": fc,
            "chartType": chart_type,
            "scale": scale,
            "title": title,
            "subtitle": subtitle,
            "pal": pal,
            "toolbar": bool(toolbar),
        },
        separators=(",", ":"),
    )

    return (
        _TEMPLATE.replace("__CDN__", _LWC_CDN)
        .replace("__PAYLOAD__", payload)
        .replace("__BG__", pal["bg"])
        .replace("__TEXT__", pal["text"])
        .replace("__STRONG__", pal["strong"])
        .replace("__GRID__", pal["grid"])
        .replace("__BORDER__", pal["border"])
        .replace("__ACCENT__", pal["accent"])
        .replace("__PANEL__", pal["panel"])
    )


# ------------------------------------------------------------------ helpers
def forecast_cone(mc_result, start_date, add_trading_days) -> dict:
    """Turn an ``inference.MonteCarloResult`` into a ``forecast`` payload:
    percentile bands mapped from forward trading-day index to calendar dates,
    anchored so day 0 == the last close."""
    P = mc_result.percentiles
    horizon = mc_result.horizon
    dates = [pd.Timestamp(start_date).strftime("%Y-%m-%d")]
    for i in range(1, horizon + 1):
        dates.append(add_trading_days(start_date, i).strftime("%Y-%m-%d"))
    out = {"dates": dates}
    for k in ("p05", "p25", "p50", "p75", "p95"):
        if k in P:
            arr = np.asarray(P[k], float)
            arr[0] = mc_result.last_price
            out[k] = arr[: horizon + 1].tolist()
    return out


# ------------------------------------------------------------------ template
_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
  :root{ --bg:__BG__; --text:__TEXT__; --strong:__STRONG__; --grid:__GRID__;
         --border:__BORDER__; --accent:__ACCENT__; --panel:__PANEL__; }
  html,body{margin:0;height:100%;background:var(--bg);
    font:12px/1.4 ui-monospace,SFMono-Regular,"IBM Plex Mono",Menlo,monospace;color:var(--text)}
  #app{position:absolute;inset:0;display:flex;flex-direction:column}
  #tb{display:flex;align-items:center;gap:.35rem;flex-wrap:wrap;
      padding:.4rem .55rem;border-bottom:1px solid var(--border);background:var(--panel)}
  #tb .grp{display:flex;gap:.15rem;align-items:center}
  #tb .sep{width:1px;height:16px;background:var(--border);margin:0 .2rem}
  #tb button{font:500 10px ui-monospace,monospace;letter-spacing:.04em;color:var(--text);
    background:transparent;border:1px solid var(--border);border-radius:5px;
    padding:.2rem .45rem;cursor:pointer;white-space:nowrap}
  #tb button:hover{color:var(--strong);border-color:var(--accent)}
  #tb button.on{color:var(--bg);background:var(--accent);border-color:var(--accent)}
  #legend{flex:1;min-width:170px;font-size:11px;color:var(--text);padding-left:.2rem}
  #legend b{color:var(--strong);font-weight:600}
  #charts{flex:1;position:relative;display:flex;flex-direction:column}
  .pane{position:relative;width:100%}
  #main{flex:1}
  .sub{height:96px;border-top:1px solid var(--border)}
  .plabel{position:absolute;left:8px;top:4px;z-index:4;font-size:10px;
    letter-spacing:.06em;color:var(--text);pointer-events:none}
</style></head>
<body><div id="app">
  <div id="tb">
    <div id="legend"></div>
    <div class="grp" id="g-type"></div><div class="sep"></div>
    <div class="grp" id="g-scale"></div><div class="sep"></div>
    <div class="grp" id="g-ind"></div><div class="sep"></div>
    <div class="grp" id="g-range"></div>
  </div>
  <div id="charts">
    <div class="pane" id="main"><div class="plabel" id="lbl-main"></div></div>
  </div>
</div>
<script src="__CDN__"></script>
<script>
(function(){
var P = __PAYLOAD__, pal = P.pal;
var LWC = window.LightweightCharts;
var lg = document.getElementById('legend');
if (!LWC){ lg.textContent = 'chart engine unavailable (offline)'; return; }

var C = P.candles;
if (!C.length){ lg.textContent = 'אין נתונים'; return; }
var T  = C.map(function(b){return b.time;});
var O  = C.map(function(b){return b.open;});
var H  = C.map(function(b){return b.high;});
var LO = C.map(function(b){return b.low;});
var CL = C.map(function(b){return b.close;});
function nf(v,d){ return (v==null||isNaN(v)) ? '—'
  : Number(v).toLocaleString(undefined,{maximumFractionDigits:d==null?2:d}); }

/* ---- indicator math ---- */
function sma(a,n){var o=[],s=0;for(var i=0;i<a.length;i++){s+=a[i];if(i>=n)s-=a[i-n];o.push(i>=n-1?s/n:null);}return o;}
function ema(a,n){var o=[],k=2/(n+1),p=null;for(var i=0;i<a.length;i++){p=(p==null)?a[i]:a[i]*k+p*(1-k);o.push(i>=n-1?p:null);}return o;}
function stdev(a,n){var o=[];for(var i=0;i<a.length;i++){if(i<n-1){o.push(null);continue;}var m=0,j;for(j=i-n+1;j<=i;j++)m+=a[j];m/=n;var s=0;for(j=i-n+1;j<=i;j++)s+=(a[j]-m)*(a[j]-m);o.push(Math.sqrt(s/n));}return o;}
function rsiCalc(a,n){var o=[null],g=0,ls=0;for(var i=1;i<a.length;i++){var ch=a[i]-a[i-1],up=Math.max(ch,0),dn=Math.max(-ch,0);
  if(i<n){g+=up;ls+=dn;o.push(null);}
  else if(i===n){g=(g+up)/n;ls=(ls+dn)/n;o.push(100-100/(1+g/(ls||1e-9)));}
  else{g=(g*(n-1)+up)/n;ls=(ls*(n-1)+dn)/n;o.push(100-100/(1+g/(ls||1e-9)));}}return o;}
function macdCalc(a){var f=ema(a,12),s=ema(a,26);
  var m=a.map(function(_,i){return (f[i]==null||s[i]==null)?null:f[i]-s[i];});
  var sig=ema(m.map(function(x){return x==null?0:x;}),9).map(function(x,i){return m[i]==null?null:x;});
  var hist=m.map(function(x,i){return (x==null||sig[i]==null)?null:x-sig[i];});
  return {macd:m,signal:sig,hist:hist};}
function heikin(){var ha=[],pO,pC;for(var i=0;i<C.length;i++){var b=C[i];var hc=(b.open+b.high+b.low+b.close)/4;
  var ho=(i===0)?(b.open+b.close)/2:(pO+pC)/2;
  ha.push({time:b.time,open:ho,high:Math.max(b.high,ho,hc),low:Math.min(b.low,ho,hc),close:hc});pO=ho;pC=hc;}return ha;}
function pts(vals){var o=[];for(var i=0;i<T.length;i++)if(vals[i]!=null&&!isNaN(vals[i]))o.push({time:T[i],value:vals[i]});return o;}
var IC = {};                                    /* indicator series cache */
function ic(key,fn){ if(!(key in IC)) IC[key]=fn(); return IC[key]; }

/* ---- chart factory ---- */
function mkChart(el,extra){
  return LWC.createChart(el, Object.assign({
    autoSize:true,
    layout:{background:{color:pal.bg},textColor:pal.text,fontFamily:'ui-monospace, monospace',fontSize:11},
    grid:{vertLines:{color:pal.grid},horzLines:{color:pal.grid}},
    rightPriceScale:{borderColor:pal.grid},
    timeScale:{borderColor:pal.grid,rightOffset:6},
    crosshair:{mode:LWC.CrosshairMode.Magnet,
      vertLine:{color:pal.accent,width:1,style:3,labelBackgroundColor:pal.accent},
      horzLine:{color:pal.accent,width:1,style:3,labelBackgroundColor:pal.accent}},
  }, extra||{}));
}
var scaleModeMap={linear:0,log:1,percent:2};
var main = mkChart(document.getElementById('main'), {
  rightPriceScale:{borderColor:pal.grid,mode:scaleModeMap[P.scale]||0},
  watermark:{visible:true,text:P.title+'  ·  CARN-X',color:pal.wm,fontSize:44,
    horzAlign:'center',vertAlign:'center',fontFamily:'Space Grotesk, monospace'},
});
document.getElementById('lbl-main').textContent = P.title + (P.subtitle?('  ·  '+P.subtitle):'');

/* ---- time-scale sync ---- */
var subs={}, syncing=false;
function allCharts(){var a=[main];for(var k in subs)a.push(subs[k].ch);return a;}
function wireSync(ch){
  ch.timeScale().subscribeVisibleLogicalRangeChange(function(r){
    if(syncing||!r)return; syncing=true;
    allCharts().forEach(function(t){ if(t!==ch) t.timeScale().setVisibleLogicalRange(r); });
    syncing=false;
  });
}
wireSync(main);

/* ---- indicator overlays on the main pane ---- */
var indState={}; (P.overlays||[]).forEach(function(k){indState[k]=true;});
var indSeries=[];
var OVL=[
  ['EMA21', pal.accent, 1,   function(){return ema(CL,21);}],
  ['EMA50', pal.violet, 1,   function(){return ema(CL,50);}],
  ['SMA20', pal.accent, 1,   function(){return sma(CL,20);}],
  ['SMA50', pal.violet, 1,   function(){return sma(CL,50);}],
  ['SMA200',pal.amber,  1.6, function(){return sma(CL,200);}],
];
function ovlRaw(tag){ var d=OVL.filter(function(x){return x[0]===tag;})[0]; return d?ic('raw:'+tag,d[3]):null; }
function buildIndicators(){
  indSeries.forEach(function(s){try{main.removeSeries(s);}catch(e){}});
  indSeries=[];
  OVL.forEach(function(d){
    if(!indState[d[0]])return;
    var s=main.addLineSeries({color:d[1],lineWidth:d[2],priceLineVisible:false,
      lastValueVisible:false,crosshairMarkerVisible:false});
    s.setData(pts(ic('raw:'+d[0],d[3]))); s._tag=d[0]; indSeries.push(s);
  });
  if(indState.BB){
    var m=sma(CL,20),sd=stdev(CL,20);
    var up=CL.map(function(_,i){return m[i]==null?null:m[i]+2*sd[i];});
    var dn=CL.map(function(_,i){return m[i]==null?null:m[i]-2*sd[i];});
    [[up,pal.text+'66',2],[m,pal.pink,0],[dn,pal.text+'66',2]].forEach(function(p){
      var s=main.addLineSeries({color:p[1],lineWidth:1,lineStyle:p[2],priceLineVisible:false,
        lastValueVisible:false,crosshairMarkerVisible:false});
      s.setData(pts(p[0])); indSeries.push(s);
    });
  }
  if(indState.VWAP){
    var pv=0,vv=0,w=[];
    for(var i=0;i<C.length;i++){var tp=(H[i]+LO[i]+CL[i])/3,vol=(P.vols[i]?P.vols[i].value:1);
      pv+=tp*vol;vv+=vol;w.push(vv?pv/vv:null);}
    var s=main.addLineSeries({color:pal.amber,lineWidth:1.4,priceLineVisible:false,
      lastValueVisible:false,crosshairMarkerVisible:false});
    s.setData(pts(w)); indSeries.push(s);
  }
}

/* ---- price series (swappable) ---- */
var priceSeries=null, priceType=P.chartType||'candles';
function setPrice(type){
  if(priceSeries){ main.removeSeries(priceSeries); priceSeries=null; }
  priceType=type;
  if(type==='line'||type==='area'){
    priceSeries=(type==='area')
      ? main.addAreaSeries({lineColor:pal.accent,topColor:pal.accent+'44',bottomColor:pal.accent+'03',lineWidth:2})
      : main.addLineSeries({color:pal.accent,lineWidth:2});
    priceSeries.setData(C.map(function(b){return {time:b.time,value:b.close};}));
  } else {
    priceSeries=main.addCandlestickSeries({upColor:pal.up,downColor:pal.down,
      borderUpColor:pal.up,borderDownColor:pal.down,wickUpColor:pal.up,wickDownColor:pal.down});
    priceSeries.setData(type==='heikin'?heikin():C);
  }
  if(P.events&&P.events.length) priceSeries.setMarkers(P.events);
  (P.priceLines||[]).forEach(function(pl){
    priceSeries.createPriceLine({price:pl.price,color:pl.color||pal.amber,lineWidth:1,
      lineStyle:2,axisLabelVisible:true,title:pl.title||''});
  });
  buildIndicators();
}

/* ---- volume overlay ---- */
if(P.vols.length){
  var vs=main.addHistogramSeries({priceFormat:{type:'volume'},priceScaleId:'vol'});
  main.priceScale('vol').applyOptions({scaleMargins:{top:0.86,bottom:0}});
  vs.setData(P.vols);
}

setPrice(priceType);   /* creates the price series + indicators */

/* ---- forecast cone (forward from last bar) ---- */
if(P.forecast&&P.forecast.dates){
  var fd=P.forecast.dates;
  [['p05',pal.text+'66',1,2],['p25',pal.violet,1,0],['p50',pal.accent,2,0],
   ['p75',pal.violet,1,0],['p95',pal.text+'66',1,2]].forEach(function(b){
    var arr=P.forecast[b[0]]; if(!arr)return;
    var s=main.addLineSeries({color:b[1],lineWidth:b[2],lineStyle:b[3],priceLineVisible:false,
      lastValueVisible:(b[0]==='p50'),crosshairMarkerVisible:false});
    s.setData(fd.map(function(t,i){return {time:t,value:arr[i]};}));
  });
}

/* ---- normalised comparison overlay ---- */
if(P.compare&&P.compare.data){
  var cmp=main.addLineSeries({color:pal.pink,lineWidth:1.4,priceLineVisible:false,
    lastValueVisible:false,crosshairMarkerVisible:false});
  cmp.setData(P.compare.data);
}

/* ---- sub-panes: RSI / MACD ---- */
function addSub(kind){
  if(subs[kind])return;
  var el=document.createElement('div'); el.className='pane sub'; el.id='p-'+kind;
  var lb=document.createElement('div'); lb.className='plabel';
  lb.textContent=kind==='rsi'?'RSI 14':'MACD 12 26 9'; el.appendChild(lb);
  document.getElementById('charts').appendChild(el);
  var ch=mkChart(el,{timeScale:{visible:false,borderColor:pal.grid}});
  if(kind==='rsi'){
    var s=ch.addLineSeries({color:pal.accent,lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});
    s.setData(pts(rsiCalc(CL,14)));
    [30,50,70].forEach(function(lv){s.createPriceLine({price:lv,color:pal.grid,lineWidth:1,
      lineStyle:lv===50?3:2,axisLabelVisible:true});});
  } else {
    var mm=macdCalc(CL);
    var hs=ch.addHistogramSeries({priceLineVisible:false});
    hs.setData(T.map(function(t,i){return mm.hist[i]==null?null:
      {time:t,value:mm.hist[i],color:(mm.hist[i]>=0?pal.up:pal.down)+'99'};}).filter(Boolean));
    var ml=ch.addLineSeries({color:pal.accent,lineWidth:1.5,priceLineVisible:false,lastValueVisible:false});
    ml.setData(pts(mm.macd));
    var sl=ch.addLineSeries({color:pal.amber,lineWidth:1,priceLineVisible:false,lastValueVisible:false});
    sl.setData(pts(mm.signal));
  }
  subs[kind]={ch:ch}; wireSync(ch);
  var r=main.timeScale().getVisibleLogicalRange(); if(r) ch.timeScale().setVisibleLogicalRange(r);
}
function removeSub(kind){
  if(!subs[kind])return;
  try{subs[kind].ch.remove();}catch(e){}
  var el=document.getElementById('p-'+kind); if(el&&el.parentNode)el.parentNode.removeChild(el);
  delete subs[kind];
}
(P.panes||[]).forEach(addSub);

/* ---- legend / OHLC + indicator readout ---- */
function paint(bar){
  bar = bar || C[C.length-1];
  var i = T.indexOf(bar.time); if(i<0) i=C.length-1;
  var chg = bar.open ? (bar.close/bar.open-1)*100 : 0;
  var col = chg>=0?pal.up:pal.down;
  var s = '<b>'+P.title+'</b> &nbsp; O '+nf(bar.open)+'  H '+nf(bar.high)+'  L '+nf(bar.low)
        + '  C <b style="color:'+col+'">'+nf(bar.close)+'</b>'
        + '  <span style="color:'+col+'">('+(chg>=0?'+':'')+chg.toFixed(2)+'%)</span>';
  var tags=[];
  Object.keys(indState).forEach(function(k){
    if(!indState[k] || k==='VWAP') return;
    if(k==='BB'){ tags.push('BB '+nf(sma(CL,20)[i])); return; }
    var raw=ovlRaw(k); if(raw && raw[i]!=null) tags.push(k+' '+nf(raw[i]));
  });
  if(subs.rsi) tags.push('RSI '+nf(ic('rsiv',function(){return rsiCalc(CL,14);})[i],1));
  if(tags.length) s += '&nbsp;&nbsp;<span style="opacity:.72">'+tags.join('   ')+'</span>';
  lg.innerHTML = s;
}
paint();
main.subscribeCrosshairMove(function(p){
  var b = p && p.seriesData && p.seriesData.get(priceSeries);
  paint(b && b.close!=null ? b : null);
});

/* ---- toolbar ---- */
function grp(id, items, isActive, onPick){
  var g=document.getElementById(id);
  items.forEach(function(it){
    var b=document.createElement('button'); b.textContent=it[0];
    if(isActive(it[1])) b.classList.add('on');
    b.onclick=function(){ onPick(it[1], b, g); };
    g.appendChild(b);
  });
}
function single(g,b){ [].forEach.call(g.children,function(x){x.classList.remove('on');}); b.classList.add('on'); }
if(P.toolbar){
  grp('g-type',[['נרות','candles'],['HA','heikin'],['קו','line'],['שטח','area']],
    function(v){return v===priceType;}, function(v,b,g){ single(g,b); setPrice(v); paint(); });
  grp('g-scale',[['לינארי','linear'],['לוג','log'],['%','percent']],
    function(v){return v===P.scale;}, function(v,b,g){ single(g,b);
      main.priceScale('right').applyOptions({mode:scaleModeMap[v]}); });
  grp('g-ind',[['EMA21','EMA21'],['EMA50','EMA50'],['SMA200','SMA200'],['BB','BB'],['VWAP','VWAP'],
    ['RSI','#rsi'],['MACD','#macd']],
    function(v){ return v==='#rsi'?!!subs.rsi : v==='#macd'?!!subs.macd : !!indState[v]; },
    function(v,b){
      if(v==='#rsi'){ subs.rsi?removeSub('rsi'):addSub('rsi'); b.classList.toggle('on',!!subs.rsi); return; }
      if(v==='#macd'){ subs.macd?removeSub('macd'):addSub('macd'); b.classList.toggle('on',!!subs.macd); return; }
      indState[v]=!indState[v]; b.classList.toggle('on',indState[v]); buildIndicators(); paint();
    });
} else if(!((P.overlays&&P.overlays.length)||P.subtitle)){
  document.getElementById('tb').style.display='none';
}
grp('g-range',[['1M',21],['3M',63],['6M',126],['1Y',252],['2Y',504],['ALL',0]],
  function(){return false;}, function(n,b,g){
    single(g,b); var L=C.length;
    if(n===0||n>=L){ main.timeScale().fitContent(); return; }
    main.timeScale().setVisibleRange({from:C[Math.max(0,L-n)].time,to:C[L-1].time});
  });

if(C.length>252) main.timeScale().setVisibleRange({from:C[C.length-252].time,to:C[C.length-1].time});
else main.timeScale().fitContent();
var rb=document.querySelectorAll('#g-range button'); if(rb[3]) rb[3].classList.add('on');
})();
</script>
</body></html>"""
