"""
CARN-X  --  MSTR Analysis Console
================================
A dedicated Streamlit app for analysing MicroStrategy (MSTR): live snapshot,
statistical diagnosis, the calibrated probabilistic forecast, a Monte-Carlo
price cone, out-of-sample evidence, and a risk / leverage panel.

Run:
    streamlit run mstr_app.py
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import glob
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MSTR · CARN-X", layout="wide", page_icon="📈")

from theme import inject_theme, hero  # visual layer only — no logic
import charts as C  # themed chart builders — visual only

inject_theme()

# ---------------------------------------------------------------------------
# cached data / model layer
# ---------------------------------------------------------------------------

from data_layer import (
    DataConfig,
    build_panel,
    primary_close,
    live_quote,
    panel_freshness,
)
from features import FeatureConfig
from targets import TargetConfig
from dataset import assemble
from forecast_net import DEFAULT_SEQ_FEATURES
from inference import (
    InferenceConfig,
    ModelStale,
    default_model_path,
    fit_production_model,
    load_production_model,
    monte_carlo_paths,
    predict_distribution,
    save_production_model,
    tail_probability,
)


@st.cache_resource(show_spinner="טוען פאנל שוק…")
def get_panel(cache_bust: int = 0):
    return build_panel(DataConfig())


@st.cache_resource(show_spinner="בונה פיצ'רים ויעדים…")
def load_data(cache_bust: int = 0):
    return assemble(
        seq_feature_names=DEFAULT_SEQ_FEATURES,
        target_cfg=TargetConfig(),
        require_targets=False,  # keep the latest target-less bars for prediction
    )


def refresh_market_data():
    build_panel(DataConfig(), force_refresh=True)
    get_panel.clear()
    load_data.clear()
    load_model_cached.clear()


@st.cache_data(ttl=120, show_spinner=False)
def get_live_quote():
    return live_quote("MSTR")


@st.cache_data(ttl=15, show_spinner=False)
def get_live_quotes(_bucket: int):
    """MSTR + BTC live quotes. `_bucket` = int(time()//12) forces a refresh every
    ~12 s so the live-sync fragment always shows a fresh tick."""
    from data_layer import live_quotes

    return live_quotes(("MSTR", "BTC-USD"))


@st.cache_resource(show_spinner=False)
def load_model_cached(_data, model_mtime: float):
    """Loads the on-disk production model. model_mtime keys the cache so a
    freshly-trained model is picked up."""
    return load_production_model(_data, default_model_path())


def get_model_or_stop():
    """Returns the loaded model, or renders a clear message + st.stop()."""
    if not model_exists():
        st.error("אין מודל מאומן. עבור ל'הגדרות' → 'אמן מודל עכשיו'.")
        st.stop()
    try:
        return load_model_cached(data, model_mtime())
    except ModelStale as e:
        st.warning(f"המודל השמור לא תואם את מערך הפיצ'רים הנוכחי — צריך לאמן מחדש.\n\n{e}")
        st.info("עבור ל'הגדרות' ולחץ 'אמן מודל עכשיו'.")
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"טעינת המודל נכשלה: {type(e).__name__}: {e}")
        st.stop()


def model_exists() -> bool:
    return os.path.exists(default_model_path())


def model_mtime() -> float:
    return os.path.getmtime(default_model_path()) if model_exists() else 0.0


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="display:flex;align-items:center;gap:.6rem;margin:.2rem 0 .1rem;">
      <div style="width:34px;height:34px;border-radius:11px;
                  background:linear-gradient(135deg,#38E8FF,#A855F7 55%,#F472B6);
                  box-shadow:0 8px 20px -6px rgba(168,85,247,.7);
                  display:flex;align-items:center;justify-content:center;
                  font-weight:800;color:#061024;font-family:'Space Grotesk',sans-serif;">CX</div>
      <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.15rem;
                  background:linear-gradient(100deg,#38E8FF,#A855F7);-webkit-background-clip:text;
                  background-clip:text;-webkit-text-fill-color:transparent;">MSTR · CARN-X</div>
    </div>
    <div style="color:#9AA5D6;font-size:.8rem;margin-bottom:.4rem;">
      מודל הסתברותי לניתוח מניית MicroStrategy
    </div>
    """,
    unsafe_allow_html=True,
)

section = st.sidebar.radio(
    "מסך",
    [
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
    ],
)

data = load_data()
panel = get_panel()
close = primary_close(panel)

as_of = data.index.max()
last_price = float(close.iloc[-1])
prev_close = float(close.iloc[-2])
rets = np.log(close).diff()
fresh = panel_freshness(panel)
lq = get_live_quote()

st.sidebar.markdown("---")
st.sidebar.metric(
    f"סגירת MSTR ({as_of.date()})",
    f"${last_price:,.2f}",
    f"{(last_price / prev_close - 1) * 100:+.2f}% מהסגירה הקודמת",
)
if lq.price:
    st.sidebar.metric(
        "מחיר חי (Yahoo)",
        f"${lq.price:,.2f}",
        f"{(lq.price / last_price - 1) * 100:+.2f}% מהסגירה",
        delta_color="off",
    )
    st.sidebar.caption(f"עודכן {lq.as_of} · {lq.source}")
if fresh["is_current"]:
    st.sidebar.success(f"הפאנל מעודכן לסשן האחרון ({fresh['expected_last_session']})")
else:
    st.sidebar.error(
        f"⚠ הפאנל מפגר ב-{fresh['sessions_behind']} סשנים "
        f"(עד {fresh['panel_last_date']}, צפוי {fresh['expected_last_session']}). "
        "לחץ 'רענן נתוני שוק' בהגדרות."
    )
st.sidebar.caption(f"{len(data.index)} ימי מסחר בפאנל")
if model_exists():
    st.sidebar.caption(f"מודל: {datetime.fromtimestamp(model_mtime()):%Y-%m-%d %H:%M}")
else:
    st.sidebar.warning("אין מודל מאומן — עבור ל'הגדרות'")

# --- live data sync --------------------------------------------------------
st.sidebar.markdown("---")
_sync_on = st.sidebar.toggle(
    "סנכרון נתונים חי",
    value=st.session_state.get("sync_on", True),
    key="sync_on",
    help="מושך מחיר MSTR + BTC חי מ-Yahoo ומרענן אוטומטית",
)
if _sync_on:
    _iv = st.sidebar.segmented_control(
        "תדירות רענון",
        ["15 שנ'", "30 שנ'", "60 שנ'"],
        default="30 שנ'",
        key="sync_iv",
        label_visibility="collapsed",
    )
    st.session_state["sync_every"] = {"15 שנ'": 15, "30 שנ'": 30, "60 שנ'": 60}.get(_iv, 30)
else:
    st.session_state["sync_every"] = None

_THEME_DARK = str(getattr(getattr(st.context, "theme", None), "type", "dark")) != "light"

_MKT_LABEL = {
    "regular": ("שוק פתוח", "#34E5B0"),
    "pre": ("טרום-מסחר", "#FBBF24"),
    "post": ("אחרי-מסחר", "#FBBF24"),
    "closed": ("שוק סגור", "#8B93B8"),
    "holiday": ("חג — שוק סגור", "#8B93B8"),
    "unknown": ("—", "#8B93B8"),
}


@st.fragment(run_every=st.session_state.get("sync_every"))
def _live_ticker():
    live = st.session_state.get("sync_on", True)
    try:
        qs = get_live_quotes(int(time.time() // 12) if live else 0)
    except Exception:  # noqa: BLE001 -- a live-data hiccup must never break the page
        st.caption("סנכרון חי לא זמין כרגע (Yahoo). המחירים במסכים הם מהפאנל היומי.")
        return
    spark_map = {
        "MSTR": close.tail(40),
        "BTC-USD": panel["btc_close"].dropna().tail(40) if "btc_close" in panel else None,
    }
    row = st.container(horizontal=True, gap="medium")
    for tk, name in (("MSTR", "MicroStrategy"), ("BTC-USD", "Bitcoin")):
        q = qs.get(tk)
        if not q or q.price is None:
            continue
        with row.container(border=True):
            spark = spark_map.get(tk)
            st.metric(
                f"{name}  ·  {tk.replace('-USD', '')}",
                f"${q.price:,.2f}",
                f"{q.change_pct:+.2f}%" if q.change_pct is not None else None,
                chart_data=list(spark.values) if spark is not None and len(spark) else None,
                chart_type="area",
            )
            _lbl, _col = _MKT_LABEL.get(q.market_state, _MKT_LABEL["unknown"])
            _rng = (
                rf"טווח יום \${q.day_low:,.0f}–\${q.day_high:,.0f}"
                if q.day_low and q.day_high
                else ""
            )
            _vol = f" · מחזור {q.volume / 1e6:,.1f}M" if q.volume else ""
            st.markdown(
                f"<span style='color:{_col};font-weight:600'>● {_lbl}</span>"
                f"<span style='color:#8B93B8'> &nbsp; {_rng}{_vol}</span>",
                unsafe_allow_html=True,
            )
    _age = max(0, int(time.time() - (qs.get("MSTR").epoch if qs.get("MSTR") else time.time())))
    st.caption(
        (
            f"סונכרן לפני {_age}s · רענון כל {st.session_state['sync_every']}s"
            if live and st.session_state.get("sync_every")
            else "סנכרון חי כבוי — המחירים הם מהטעינה האחרונה"
        )
        + f" · Yahoo · המודל תמיד רץ על סגירות יומיות מלאות (עד {as_of.date()})"
    )


_live_ticker()


# ===========================================================================
# 1. OVERVIEW
# ===========================================================================
if section == "סקירה":
    hero("סקירת מצב · MSTR", "מחיר, תנודתיות, drawdown ו-MSTR מול ביטקוין במבט אחד", "Overview")

    win = rets.dropna()
    ann_vol = win.tail(20).std() * np.sqrt(252)
    dd = float(close.iloc[-1] / close.cummax().iloc[-1] - 1)
    r20 = np.expm1(rets.tail(20).sum())
    r60 = np.expm1(rets.tail(60).sum())

    c = st.columns(5)
    c[0].metric("מחיר", f"${last_price:,.2f}")
    c[1].metric("תשואה 20 ימים", f"{r20:+.1%}")
    c[2].metric("תשואה 60 ימים", f"{r60:+.1%}")
    c[3].metric("תנודתיות (20d, שנתי)", f"{ann_vol:.0%}")
    c[4].metric("מרחק משיא", f"{dd:.1%}")

    # --- interactive charting terminal (crosshair · zoom · indicators · RSI/MACD) ---
    st.subheader("MSTR — טרמינל גרפים")
    from tv_chart import price_chart_html

    _oc = ["mstr_open", "mstr_high", "mstr_low", "mstr_close", "mstr_volume"]
    ohlc = panel[_oc].reindex(close.index).dropna().reset_index()
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    st.iframe(
        price_chart_html(
            ohlc,
            title="MSTR",
            subtitle="יומי · מתואם-פיצול",
            dark=_THEME_DARK,
            overlays=["EMA21", "SMA50", "SMA200"],
            panes=["rsi"],
            events=[
                {
                    "time": "2024-08-07",
                    "position": "aboveBar",
                    "color": "#FBBF24",
                    "shape": "arrowDown",
                    "text": "10:1 split",
                }
            ],
            price_lines=[
                {"price": float(close.tail(252).max()), "title": "52w high", "color": "#8B93B8"},
                {"price": float(close.tail(252).min()), "title": "52w low", "color": "#8B93B8"},
            ],
        ),
        height=560,
    )
    st.caption(
        "כפתורי הטרמינל: סוג גרף · סקאלה · אינדיקטורים (EMA/SMA/BB/VWAP/RSI/MACD) · טווח. "
        "גלגל=זום · גרירה=הזזה · צלב-כוונת עם קריאת OHLC חיה. הכל מיידי — בלי רענון."
    )

    gcol, dcol = st.columns([1, 1])
    with gcol:
        st.altair_chart(
            C.gauge(
                1.0 + dd,
                lo=0.0,
                hi=1.0,
                title="מחיר ביחס לשיא כל הזמנים",
                label=f"{(1.0 + dd):.0%} מהשיא  ·  drawdown {dd:.0%}",
                zones=((0.0, 0.4, C.RED), (0.4, 0.75, C.AMBER), (0.75, 1.0, C.GREEN)),
            ),
            width="stretch",
        )
    with dcol:
        last60 = np.sign(rets.tail(60).dropna())
        st.altair_chart(
            C.donut(
                {"ימים ירוקים": int((last60 > 0).sum()), "ימים אדומים": int((last60 < 0).sum())},
                title="ימי עלייה מול ירידה (60 ימים)",
                center=f"{(last60 > 0).mean():.0%}",
                colors=[C.GREEN, C.RED],
            ),
            width="stretch",
        )

    st.subheader("MSTR מול BTC (מנורמל, שנה)")
    import altair as _alt

    btc = panel["btc_close"].reindex(close.index).ffill()
    _w = 252 if len(close) > 252 else len(close)
    _mstr_w, _btc_w = close.tail(_w), btc.tail(_w)
    norm = pd.DataFrame(
        {
            "date": pd.to_datetime(_mstr_w.index),
            "MSTR": (_mstr_w / _mstr_w.iloc[0]).to_numpy(),
            "BTC": (_btc_w / _btc_w.iloc[0]).to_numpy(),
        }
    ).melt("date", var_name="נכס", value_name="מכפיל")
    st.altair_chart(
        _alt.Chart(norm)
        .mark_line(strokeWidth=2)
        .encode(
            x=_alt.X("date:T", title=None),
            y=_alt.Y("מכפיל:Q", scale=_alt.Scale(zero=False), title="מכפיל (בסיס = שנה אחורה)"),
            color=_alt.Color(
                "נכס:N",
                scale=_alt.Scale(range=[C.CYAN, C.AMBER]),
                legend=_alt.Legend(orient="top", title=None),
            ),
        )
        .properties(height=260),
        width="stretch",
    )

    colA, colB = st.columns(2)
    with colA:
        st.subheader("תשואות יומיות (שנה אחרונה)")
        st.altair_chart(
            C.diverging_bars(
                pd.Series(np.expm1(rets.tail(252).values), index=close.index[-252:]), height=230
            ),
            width="stretch",
        )
    with colB:
        st.subheader("Drawdown")
        ddser = (close / close.cummax() - 1).tail(504)
        st.altair_chart(
            C.area_gradient(
                ddser, height=230, color=C.RED, signed=True, percent=True, y_title="drawdown"
            ),
            width="stretch",
        )

    nav_df = st.session_state.get("nav_df")
    if nav_df is not None:
        from nav_premium import NAVPremiumCalculator

        calc = NAVPremiumCalculator()
        snap = calc.latest_snapshot(calc.compute(nav_df))
        st.subheader("NAV Premium")
        d = st.columns(4)
        d[0].metric("פרמיה ל-NAV", f"{snap.premium:+.1%}")
        d[1].metric("mNAV ratio", f"{snap.mnav_ratio:.2f}")
        d[2].metric("z-score", f"{snap.premium_zscore:+.2f}")
        d[3].metric("משטר", snap.regime_label)


# ===========================================================================
# 1b. PROBABILITY CALCULATOR  (sensors -> classical distributions)
# ===========================================================================
elif section == "מחשבון הסתברויות":
    import math as _math
    import altair as alt
    from scipy import stats
    from prob_models import (
        market_inputs,
        add_trading_days,
        gbm_paths,
        _geo_rate_from_touch,
        next_day_expected_price,
    )
    from hybrid_engine import continuous_support

    hero(
        "מחשבון הסתברויות",
        "עמוד נפרד לכל התפלגות. שני קלטים בלבד — מחיר יעד ואופק — וההתפלגות "
        "עונה על השאלה 'האם MSTR תגיע לשם, ומתי', ופורשת את ההסתברות על ציר הזמן.",
        "Probability",
    )

    # ------------------------------------------------------------------ the 2 inputs
    ic = st.columns([2, 2, 1])
    target = ic[0].number_input(
        "מחיר יעד ($)",
        min_value=1.0,
        value=float(round(last_price * 1.10, 2)),
        step=1.0,
        help="השער שאתה בודק אם המחיר יגיע אליו",
    )
    horizon = ic[1].slider(
        "אופק — ציר הזמן (ימי מסחר)", 5, 504, 63, help="על פני כמה ימי מסחר קדימה נפרוש את ההסתברות"
    )
    _tgt_date = add_trading_days(pd.Timestamp.now(), int(horizon)).date()
    ic[2].metric("תאריך יעד", f"{_tgt_date:%d/%m/%y}")

    up = target >= last_price
    move_pct = (target / last_price - 1) * 100
    st.caption(
        f"S₀ = ${last_price:,.2f}  ·  יעד ${target:,.2f} ({move_pct:+.1f}%)  ·  "
        f"{horizon} ימי מסחר → {_tgt_date}"
    )

    days = np.arange(1, int(horizon) + 1)

    @st.cache_data(ttl=300, show_spinner="מריץ סימולציה…")
    def _pcalc_engine(tgt: float, h: int, s0_key: float):
        mi = market_inputs(close)
        paths = gbm_paths(mi, int(h), n_paths=40000)
        cross = (paths[:, 1:] >= tgt) if tgt >= paths[0, 0] else (paths[:, 1:] <= tgt)
        ever_by = np.maximum.accumulate(cross, axis=1)
        p_touch_t = ever_by.mean(axis=0)
        first = np.argmax(cross, axis=1) + 1
        hit = cross.any(axis=1)
        w = close.astype(float).dropna().tail(mi.window_days + 1).to_numpy()
        n_cross = int(np.sum(np.sign(w[1:] - tgt) != np.sign(w[:-1] - tgt)))
        return dict(
            mu=float(mi.mu_daily),
            sig=float(mi.sigma_daily),
            s0=float(mi.last_price),
            lo=float(mi.hist_low),
            hi=float(mi.hist_high),
            window=int(mi.window_days),
            p_touch_t=p_touch_t.tolist(),
            p_touch_h=float(p_touch_t[-1]),
            exp_first=float(first[hit].mean()) if hit.any() else float("nan"),
            cross_rate=n_cross / max(len(w) - 1, 1),
            n_cross=n_cross,
            med_end=float(np.median(paths[:, -1])),
            p_close_end=float(
                (paths[:, -1] >= tgt).mean()
                if tgt >= mi.last_price
                else (paths[:, -1] <= tgt).mean()
            ),
        )

    E = _pcalc_engine(float(target), int(horizon), float(last_price))
    p_day = _geo_rate_from_touch(E["p_touch_h"], int(horizon))

    # ------------------------------------------------------------------ headline
    st.markdown("---")
    st.markdown(
        f"### 🎯 האם MSTR {'יגיע ל' if up else 'יירד ל'}-${target:,.0f} תוך {horizon} ימים?"
    )
    g = st.columns(4)
    g[0].metric(
        "P(נוגע בשער)",
        f"{E['p_touch_h']:.0%}",
        help="Monte-Carlo first-passage · זנבות Student-t · התשובה הריאלית",
    )
    g[1].metric("P(נסגר מעבר לשער)", f"{E['p_close_end']:.0%}")
    g[2].metric("מחיר חציוני ביום היעד", f"${E['med_end']:,.0f}")
    g[3].metric(
        "יום צפוי לנגיעה ראשונה",
        "—" if not np.isfinite(E["exp_first"]) else f"~{E['exp_first']:.0f}",
    )
    st.progress(min(E["p_touch_h"], 1.0))
    st.caption(
        "זו התשובה הריאלית (Monte-Carlo). למטה — מה **כל התפלגות קלאסית** אומרת "
        "על אותה שאלה, ואיך ההסתברות נפרשת על ציר הזמן."
    )
    st.markdown("---")

    # ------------------------------------------------------------------ helpers
    def _moments(mean, var):
        ok = isinstance(var, (int, float)) and _math.isfinite(var) and var >= 0
        sd = _math.sqrt(var) if ok else float("nan")
        m = st.columns(3)
        m[0].metric("תוחלת (E)", f"{mean:,.4g}" if _math.isfinite(mean) else "—")
        m[1].metric(
            "שונות (Var)",
            f"{var:,.4g}" if (isinstance(var, (int, float)) and _math.isfinite(var)) else "—",
        )
        m[2].metric("סטיית תקן (SD)", f"{sd:,.4g}" if _math.isfinite(sd) else "—")

    def _curve(y, title, ylab="הסתברות"):
        d = pd.DataFrame({"יום": days, "p": np.asarray(y, float)[: len(days)]})
        area = (
            alt.Chart(d)
            .mark_area(
                line={"strokeWidth": 2, "color": "#38E8FF"},
                color="#38E8FF33",
                interpolate="monotone",
            )
            .encode(
                x=alt.X("יום:Q", title="ימי מסחר קדימה"),
                y=alt.Y("p:Q", title=ylab, axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("יום:Q"), alt.Tooltip("p:Q", format=".1%")],
            )
        )
        rule = (
            alt.Chart(pd.DataFrame({"יום": [int(horizon)]}))
            .mark_rule(strokeDash=[4, 4], color="#FBBF24")
            .encode(x="יום:Q")
        )
        st.altair_chart((area + rule).properties(height=240, title=title), width="stretch")

    DIST = st.segmented_control(
        "בחר התפלגות",
        [
            "בינומי",
            "גיאומטרי",
            "פואסון",
            "נורמלי",
            "אחידה בדידה",
            "אחידה רציפה",
            "יום המסחר הקרוב",
            "מתקדם",
        ],
        default="בינומי",
        label_visibility="collapsed",
    )

    # ============================== בינומי
    if DIST == "בינומי":
        st.subheader("התפלגות בינומית")
        st.caption(
            f"מודל: כל יום מסחר = ניסוי ברנולי — 'האם המחיר נגע ביעד היום'. "
            f"ההסתברות היומית p = {p_day:.2%} (מכוילת מ-Monte-Carlo כך שהמצטבר "
            f"שווה ל-P האמיתי לנגיעה)."
        )
        n = int(horizon)
        exp_hits = n * p_day
        _moments(exp_hits, n * p_day * (1 - p_day))
        a = st.columns(3)
        a[0].metric(f"P(נגיעה ≥ פעם אחת ב-{n} ימים)", f"{1 - (1 - p_day) ** n:.1%}")
        a[1].metric("P(נגיעה ≥ פעמיים)", f"{float(stats.binom.sf(1, n, p_day)):.1%}")
        a[2].metric("מספר נגיעות צפוי", f"{exp_hits:.1f}")
        _curve([1 - (1 - p_day) ** int(t) for t in days], "P(נגיעה ≥ פעם אחת עד יום t)")
        st.caption(
            "⚠ הבינומי מניח ימים בלתי-תלויים — הנחה שמופרת בשוק (volatility "
            "clustering, מומנטום). לכן p מגיע מ-Monte-Carlo ולא מספירה נאיבית."
        )

    # ============================== גיאומטרי
    elif DIST == "גיאומטרי":
        st.subheader("התפלגות גיאומטרית")
        st.caption(f"מודל: כמה ימי מסחר עד ה**נגיעה הראשונה** ביעד. p יומי = {p_day:.2%}.")
        exp_day = 1.0 / p_day
        _moments(exp_day, (1 - p_day) / p_day**2)
        a = st.columns(3)
        a[0].metric(f"P(נגיעה ראשונה תוך {horizon} ימים)", f"{E['p_touch_h']:.1%}")
        a[1].metric("יום צפוי לנגיעה ראשונה", f"{min(exp_day, 99999):.0f}")
        a[2].metric("P(נגיעה כבר בשבוע הראשון)", f"{1 - (1 - p_day) ** 5:.1%}")
        _curve(
            [1 - (1 - p_day) ** int(t) for t in days],
            "P(נגיעה ראשונה עד יום t)  —  פונקציית התפלגות מצטברת",
        )
        pmf = pd.DataFrame({"יום": days, "p": [p_day * (1 - p_day) ** (int(t) - 1) for t in days]})
        st.altair_chart(
            alt.Chart(pmf)
            .mark_bar(color="#A855F7")
            .encode(
                x=alt.X("יום:Q", title="ימי מסחר"),
                y=alt.Y("p:Q", title="P(הנגיעה הראשונה בדיוק ביום t)", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("יום:Q"), alt.Tooltip("p:Q", format=".2%")],
            )
            .properties(height=200, title="PMF — מתי בדיוק תהיה הנגיעה הראשונה"),
            width="stretch",
        )

    # ============================== פואסון
    elif DIST == "פואסון":
        st.subheader("התפלגות פואסון")
        lam = max(E["cross_rate"] * int(horizon), 1e-9)
        st.caption(
            f"מודל: מספר ה**חציות** של רמת ${target:,.0f} (מלמעלה או מלמטה) ב-{horizon} "
            f"ימים. קצב היסטורי = {E['n_cross']} חציות ב-{E['window']} ימים  →  λ = {lam:.2f}."
        )
        _moments(lam, lam)
        a = st.columns(3)
        a[0].metric("מספר חציות צפוי (λ)", f"{lam:.2f}")
        a[1].metric("P(≥ חצייה אחת)", f"{1 - _math.exp(-lam):.1%}")
        a[2].metric("P(≥ 3 חציות)", f"{float(stats.poisson.sf(2, lam)):.1%}")
        _curve(
            [1 - _math.exp(-E["cross_rate"] * int(t)) for t in days],
            "P(≥ חצייה אחת של רמת היעד עד יום t)",
        )
        kk = np.arange(0, max(int(lam * 3) + 4, 8))
        st.altair_chart(
            alt.Chart(pd.DataFrame({"k": kk, "p": stats.poisson.pmf(kk, lam)}))
            .mark_bar(color="#A855F7")
            .encode(
                x=alt.X("k:Q", title="מספר חציות"),
                y=alt.Y("p:Q", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("k:Q"), alt.Tooltip("p:Q", format=".2%")],
            )
            .properties(height=200, title=f"P(בדיוק k חציות ב-{horizon} ימים)"),
            width="stretch",
        )

    # ============================== נורמלי
    elif DIST == "נורמלי":
        st.subheader("התפלגות נורמלית — תשואת h ימים")
        st.caption(
            "מודל: תשואת הלוג ל-h ימים ~ נורמלית עם תוחלת μ·h ושונות σ²·h. "
            "עונה על **המחיר בסוף האופק** — לא על נגיעה בדרך."
        )
        mu, sig = E["mu"], E["sig"]
        m = last_price * _math.exp(mu * horizon + 0.5 * (sig**2) * horizon)
        v = m**2 * (_math.exp((sig**2) * horizon) - 1)
        _moments(m, v)

        def _pab(t):
            zt = (_math.log(target / last_price) - mu * t) / (sig * _math.sqrt(t) + 1e-12)
            return float(1 - stats.norm.cdf(zt))

        a = st.columns(3)
        a[0].metric(f"P(מחיר ≥ ${target:,.0f} ביום {horizon})", f"{_pab(int(horizon)):.1%}")
        a[1].metric(f"P(מחיר ≤ ${target:,.0f})", f"{1 - _pab(int(horizon)):.1%}")
        a[2].metric("מחיר חציוני צפוי", f"${last_price * _math.exp(mu * horizon):,.0f}")
        _curve([_pab(int(t)) for t in days], "P(המחיר בסוף יום t מעל היעד)")
        st.caption(
            "נורמלי על תשואות = לוג-נורמלי על המחיר. מתעלם מזנבות שמנים (kurtosis) "
            "שקיימים ב-MSTR — למספר המדויק ראה 'P(נוגע בשער)' למעלה."
        )

    # ============================== אחידה בדידה
    elif DIST == "אחידה בדידה":
        st.subheader("התפלגות אחידה בדידה — פריור חסר-מידע")
        st.caption(
            "מודל: אם הנגיעה תקרה — כל יום מ-1 עד N שווה-סיכוי. אין שום מידע מהשוק. "
            "הבסיס להשוואה: כמה ה'גיאומטרי' באמת מוסיף מעל ניחוש עיוור."
        )
        n = int(horizon)
        _moments((n + 1) / 2, (n**2 - 1) / 12)
        a = st.columns(3)
        a[0].metric("P(קורה בכל יום ספציפי)", f"{1 / n:.2%}")
        a[1].metric(f"P(קורה עד אמצע האופק, יום {n // 2})", f"{(n // 2) / n:.0%}")
        a[2].metric("יום 'צפוי' (תוחלת)", f"{(n + 1) / 2:.0f}")
        _curve([int(t) / n for t in days], "P(קורה עד יום t)  —  קו ליניארי (אין מידע)")

    # ============================== אחידה רציפה
    elif DIST == "אחידה רציפה":
        st.subheader("התפלגות אחידה רציפה — פריור על המחיר")
        lo, hi = E["lo"], E["hi"]
        st.caption(
            f"מודל: המחיר שווה-סיכוי בכל מקום בטווח ההיסטורי ${lo:,.0f}–${hi:,.0f}. "
            "מתעלם מדריפט ומזנבות שמנים. בסיס-השוואה בלבד."
        )
        span = max(hi - lo, 1e-9)
        _moments((lo + hi) / 2, span**2 / 12)
        p_reach = (
            max(0.0, min(1.0, (hi - target) / span))
            if up
            else max(0.0, min(1.0, (target - lo) / span))
        )
        a = st.columns(2)
        a[0].metric(f"P(מחיר {'≥' if up else '≤'} ${target:,.0f})", f"{p_reach:.0%}")
        a[1].metric("מחיר 'צפוי' (אמצע הטווח)", f"${(lo + hi) / 2:,.0f}")
        xs = np.linspace(lo, hi, 140)
        pv = np.clip((hi - xs) / span, 0, 1)
        st.altair_chart(
            (
                alt.Chart(pd.DataFrame({"מחיר": xs, "p": pv}))
                .mark_line(color="#38E8FF")
                .encode(
                    x=alt.X("מחיר:Q", scale=alt.Scale(zero=False)),
                    y=alt.Y("p:Q", title="P(המחיר לפחות X)", axis=alt.Axis(format="%")),
                )
                + alt.Chart(pd.DataFrame({"מחיר": [target]}))
                .mark_rule(strokeDash=[4, 4], color="#FBBF24")
                .encode(x="מחיר:Q")
            ).properties(height=240, title="P(המחיר ≥ X) לפי אחיד רציף"),
            width="stretch",
        )

    # ============================== יום המסחר הקרוב
    elif DIST == "יום המסחר הקרוב":
        nd = next_day_expected_price(market_inputs(close))
        st.subheader("💡 מחיר צפוי ליום המסחר הקרוב")
        cc = st.columns([2, 1, 1, 1, 1])
        cc[0].metric(
            "מחיר צפוי מחר", f"${nd['expected_price']:,.2f}", f"{nd['expected_change_pct']:+.2f}%"
        )
        cc[1].metric("drift יומי", f"{nd['drift_daily_pct']:+.3f}%")
        cc[2].metric("תנודתיות יומית", f"{nd['vol_daily_pct']:.2f}%")
        cc[3].metric("skew", f"{nd['skew']:+.2f}")
        cc[4].metric("עודף kurtosis", f"{nd['excess_kurtosis']:+.2f}")
        st.caption(
            f"טווח 80% למחר ${nd['band80_low']:,.2f}–${nd['band80_high']:,.2f}  ·  "
            f"מבוסס {nd['based_on_days']} ימי מסחר  ·  "
            f"תנודתיות שנתית {nd['vol_annual_pct']:.0f}%"
        )

    # ============================== מתקדם
    else:
        import math as __m
        from prob_models import (
            full_report,
            CUSTOM_MODELS,
            compute_custom,
            custom_model_keys,
            custom_sensors,
        )

        st.caption(
            "כל 25+ המודלים הקלאסיים, מפורמטים אוטומטית מ-{מחיר יעד, אופק} עם "
            "ברירות מחדל שקטות. למטה — בונה שאילתות חופשי (סנסור לכל מודל, לבחירתך)."
        )
        rep = full_report(
            close,
            target_price=float(target),
            horizon_days=int(horizon),
            k_count=1,
            day_n=max(1, int(horizon) // 2),
            r_successes=2,
            move_threshold=0.05,
            sample_days=min(20, int(horizon)),
            loss_pct=0.10,
            extreme_thr=0.10,
            pattern="UUD",
            price_band=continuous_support(last_price, float(target)),
            active=None,
        )

        def _fmt(v):
            return f"{v:.4g}" if isinstance(v, (int, float)) and __m.isfinite(v) else "—"

        def _render(res):
            mm = st.columns(3)
            mm[0].metric("תוחלת", _fmt(res.mean))
            mm[1].metric("שונות", _fmt(res.variance))
            mm[2].metric("סטיית תקן", _fmt(res.std))
            for label, val in res.answers.items():
                if isinstance(val, (int, float)) and 0.0 <= val <= 1.0:
                    st.progress(float(val), text=f"{label}  =  {val:.1%}")
                elif isinstance(val, (int, float)) and __m.isfinite(val):
                    st.write(f"**{label}** = {val:,.4g}")
                else:
                    st.write(f"**{label}** = —")
            st.caption(
                " · ".join(f"{k}={_fmt(v)}" for k, v in res.params.items())
                + (f"   —   {res.note}" if res.note else "")
            )

        tabs = st.tabs(
            [
                "בדידות",
                "רציפות",
                "ערכי קיצון",
                "קומבינטוריקה",
                "תלות בזמן",
                "היסק סטטיסטי",
                "טבלה מלאה",
                "🛠️ בונה שאילתות",
            ]
        )
        for tab, fam in zip(
            tabs[:6], ["discrete", "continuous", "extreme", "combinatorics", "timedep", "inference"]
        ):
            with tab:
                for res in rep.by_family(fam):
                    with st.expander(res.hebrew, expanded=True):
                        _render(res)
        with tabs[6]:
            st.dataframe(
                pd.DataFrame([r.row() for r in rep.results]).round(4),
                hide_index=True,
                width="stretch",
            )
        with tabs[7]:
            for mkey in custom_model_keys():
                meta = CUSTOM_MODELS[mkey]
                with st.expander(
                    ("⭐ " if meta.get("flagship") else "") + meta["hebrew"],
                    expanded=bool(meta.get("flagship")),
                ):
                    sensors = custom_sensors(mkey, close)
                    spec = {}
                    cols = st.columns(min(len(sensors), 3) or 1)
                    for i, sen in enumerate(sensors):
                        cc2 = cols[i % len(cols)]
                        wkey = f"cs_{mkey}_{sen.name}"
                        if sen.kind == "text":
                            spec[sen.name] = cc2.text_input(
                                sen.label, str(sen.default or "UUD"), max_chars=16, key=wkey
                            )
                        elif sen.kind in ("price", "pct"):
                            spec[sen.name] = cc2.slider(
                                sen.label,
                                float(sen.lo),
                                float(sen.hi),
                                float(sen.default),
                                float(sen.step),
                                help=sen.help or None,
                                key=wkey,
                            )
                        else:
                            spec[sen.name] = cc2.slider(
                                sen.label,
                                int(sen.lo),
                                int(sen.hi),
                                int(sen.default),
                                int(max(sen.step, 1)),
                                help=sen.help or None,
                                key=wkey,
                            )
                    try:
                        _render(compute_custom(close, mkey, spec))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"שגיאה בחישוב: {type(e).__name__}: {e}")


# ===========================================================================
# 2. DIAGNOSIS
# ===========================================================================
elif section == "אבחון סטטיסטי":
    hero(
        "אבחון סטטיסטי · DiagnosisLayer",
        "אפיון המשטר הנוכחי: תנודתיות, זנבות, מגמה ונקודות שבירה",
        "L1 · Diagnosis",
    )
    lookback = st.slider("חלון אבחון (ימי מסחר)", 60, 500, 250)

    from diagnosis_layer import DiagnosisLayer

    diag = DiagnosisLayer().diagnose(close.tail(lookback).values)

    c = st.columns(4)
    c[0].metric("תנודתיות היסטורית", f"{diag.volatility.historical_vol:.0%}")
    c[1].metric("Max Drawdown", f"{diag.extremes.max_drawdown:.1%}")
    c[2].metric("Skewness", f"{diag.basic.skewness:+.2f}")
    c[3].metric("Kurtosis", f"{diag.basic.kurtosis:+.2f}")

    c = st.columns(4)
    c[0].metric("התפלגות מיטבית", diag.distribution.best_fit_name)
    c[1].metric("נורמלי? (JB)", "לא" if not diag.distribution.is_normal_jb else "כן")
    c[2].metric(
        "מחזור דומיננטי",
        f"{diag.cyclicity.dominant_period:.0f}" if diag.cyclicity.dominant_period else "—",
    )
    c[3].metric("חריגים (IQR)", f"{diag.extremes.n_outliers_iqr}")

    g = st.columns(3)
    with g[0]:
        st.altair_chart(
            C.gauge(
                float(diag.volatility.historical_vol),
                lo=0.0,
                hi=1.5,
                title="תנודתיות שנתית",
                label=f"{diag.volatility.historical_vol:.0%}",
                zones=((0.0, 0.4, C.GREEN), (0.4, 0.8, C.AMBER), (0.8, 1.5, C.RED)),
            ),
            width="stretch",
        )
    with g[1]:
        st.altair_chart(
            C.gauge(
                abs(float(diag.extremes.max_drawdown)),
                lo=0.0,
                hi=1.0,
                title="Max Drawdown",
                label=f"{diag.extremes.max_drawdown:.0%}",
                zones=((0.0, 0.3, C.GREEN), (0.3, 0.6, C.AMBER), (0.6, 1.0, C.RED)),
            ),
            width="stretch",
        )
    with g[2]:
        _nout = int(diag.extremes.n_outliers_iqr)
        st.altair_chart(
            C.donut(
                {"ימים רגילים": max(lookback - _nout, 0), "ימי קיצון": _nout},
                title="ימי קיצון (IQR)",
                center=f"{_nout}",
                colors=[C.CYAN, C.RED],
            ),
            width="stretch",
        )

    st.subheader("המלצות המנוע")
    for r in diag.recommendations:
        st.markdown(f"- {r}")

    if diag.volatility.garch_volatility is not None:
        st.subheader("תנודתיות GARCH(1,1) מותנית")
        _gv = np.asarray(diag.volatility.garch_volatility, float).ravel()
        _n = min(len(_gv), len(close))
        st.altair_chart(
            C.area_gradient(
                pd.Series(_gv[-_n:], index=close.index[-_n:]),
                height=240,
                color=C.VIOLET,
                y_title="σ מותנית",
            ),
            width="stretch",
        )


# ===========================================================================
# 3. PROBABILISTIC FORECAST
# ===========================================================================
elif section == "תחזית הסתברותית":
    hero(
        "תחזית הסתברותית",
        "ההתפלגות החזאית המלאה (Student-t) לאופקים 1 / 5 / 20 ימים",
        "L7 · Forecast",
    )
    st.caption(
        "המודל פולט **התפלגות** של תשואת MSTR העתידית, לא נקודה. "
        "בבדיקות out-of-sample דיוק הכיוון ≈ 50% (מובהקות אפסית) — "
        "**המספרים האמינים הם רוחב ההתפלגות, הסתברויות הזנב והכיול**, לא הממוצע."
    )

    pm = get_model_or_stop()
    dist = predict_distribution(pm, data)
    st.caption(f"מודל אומן {pm.trained_at} · {pm.train_rows} שורות · דאטה עד {pm.data_date_max}")
    if str(pm.data_date_max) < str(as_of.date()):
        st.warning(
            f"המודל אומן על דאטה עד {pm.data_date_max}, אבל הפאנל כבר מגיע ל-{as_of.date()}. "
            "אמן מחדש בהגדרות כדי לחזות מהמצב העדכני."
        )
    st.info(
        f"החיזוי מבוסס על סגירת {as_of.date()} = ${last_price:,.2f}"
        + (f"  ·  מחיר חי כעת ≈ ${lq.price:,.2f}" if lq.price else "")
    )

    # --- ridgeline: the three predictive Student-t densities (display only) ---
    from scipy.stats import t as _tdist

    _dens = {}
    for _, _r in dist.iterrows():
        _lo, _hi = _r.mu_raw - 4 * _r.sigma_raw, _r.mu_raw + 4 * _r.sigma_raw
        _xs = np.linspace(_lo, _hi, 220)
        _ys = _tdist.pdf((_xs - _r.mu_raw) / _r.sigma_raw, df=_r.nu) / _r.sigma_raw
        _dens[f"{int(_r.horizon)} ימים"] = (np.expm1(_xs), _ys)
    st.altair_chart(
        C.ridgeline(
            _dens, title="התפלגות התשואה החזויה לפי אופק", x_title="תשואה", percent=True, height=110
        ),
        width="stretch",
    )

    for _, row in dist.iterrows():
        h = int(row.horizon)
        st.subheader(f"אופק {h} ימי מסחר")
        cc = st.columns(5)
        cc[0].metric("תשואה צפויה", f"{row.expected_move_pct:+.2f}%")
        cc[1].metric("P(עלייה)", f"{row.p_up:.0%}")
        cc[2].metric("טווח 90% תחתון", f"{row.ret_q05_pct:+.1f}%")
        cc[3].metric("טווח 90% עליון", f"{row.ret_q95_pct:+.1f}%")
        cc[4].metric("ν (זנבות)", f"{row.nu:.1f}")

        vcol, tcol = st.columns([1, 2])
        with vcol:
            st.altair_chart(
                C.gauge(
                    float(row.p_up),
                    lo=0.0,
                    hi=1.0,
                    title="P(עלייה)",
                    label=f"{row.p_up:.0%}",
                    zones=((0.0, 0.45, C.RED), (0.45, 0.55, C.AMBER), (0.55, 1.0, C.GREEN)),
                ),
                width="stretch",
            )
        with tcol:
            st.altair_chart(
                C.bullet(
                    float(row.expected_move_pct),
                    0.0,
                    lo=float(row.ret_q05_pct),
                    hi=float(row.ret_q95_pct),
                    title="תשואה צפויה מול טווח 90% (%)",
                    bands=[
                        (float(row.ret_q05_pct), float(row.ret_q25_pct), C.RED),
                        (float(row.ret_q25_pct), float(row.ret_q75_pct), C.AMBER),
                        (float(row.ret_q75_pct), float(row.ret_q95_pct), C.GREEN),
                    ],
                    fmt="+.1f",
                ),
                width="stretch",
            )

        band = pd.DataFrame(
            {
                "אחוזון": ["5%", "25%", "חציון", "75%", "95%"],
                "תשואה %": [
                    row.ret_q05_pct,
                    row.ret_q25_pct,
                    row.expected_move_pct,
                    row.ret_q75_pct,
                    row.ret_q95_pct,
                ],
                "מחיר יעד": [
                    last_price * (1 + x / 100)
                    for x in [
                        row.ret_q05_pct,
                        row.ret_q25_pct,
                        row.expected_move_pct,
                        row.ret_q75_pct,
                        row.ret_q95_pct,
                    ]
                ],
            }
        )
        st.dataframe(band.round(2), hide_index=True, width="stretch")

        thr = st.select_slider(
            f"הסתברות לירידה של לפחות … (h={h})",
            options=[-30, -25, -20, -15, -10, -7, -5, -3],
            value=-10,
            key=f"thr{h}",
        )
        p = tail_probability(row, float(thr))
        st.progress(min(p, 1.0), text=f"P(תשואה ≤ {thr}%)  =  {p:.1%}")
        st.markdown("---")


# ===========================================================================
# 1c. TRADING TERMINAL  (institutional chart + model forecast overlay)
# ===========================================================================
elif section == "טרמינל מסחר":
    from tv_chart import price_chart_html, forecast_cone
    from prob_models import add_trading_days

    hero(
        "טרמינל מסחר · CARN-X",
        "גרף מוסדי מלא — אינדיקטורים, RSI/MACD, והחרוט ההסתברותי של המודל משורטט קדימה מהנר האחרון",
        "Terminal",
    )

    _ASSETS = {
        "MSTR": "mstr",
        "Bitcoin": "btc",
        "IBIT (BTC ETF)": "ibit",
        "MSTU (2x)": "mstu",
        "MSTZ (-2x)": "mstz",
    }
    tc = st.columns([1.4, 1, 1.1, 1.3, 1.2])
    sym_name = tc[0].selectbox("נכס", list(_ASSETS), index=0)
    sym = _ASSETS[sym_name]
    tf = tc[1].segmented_control(
        "מסגרת זמן", ["יומי", "שבועי", "חודשי"], default="יומי", label_visibility="collapsed"
    )
    ctype = tc[2].segmented_control(
        "סוג גרף", ["נרות", "HA", "קו", "שטח"], default="נרות", label_visibility="collapsed"
    )
    cmp_name = tc[3].selectbox("השוואה (מנורמל)", ["—", "Bitcoin", "S&P 500", "Nasdaq", "זהב"])
    show_fc = tc[4].toggle(
        "חרוט המודל", value=(sym == "mstr"), help="חרוט Monte-Carlo ניטרלי-סיכון קדימה על הגרף"
    )

    _oc = [f"{sym}_{x}" for x in ("open", "high", "low", "close", "volume")]
    _raw = panel[_oc].dropna()
    _raw.columns = ["open", "high", "low", "close", "volume"]
    if tf != "יומי":
        _rule = {"שבועי": "W", "חודשי": "ME"}[tf]
        _raw = (
            _raw.resample(_rule)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
    ohlc = _raw.reset_index()
    ohlc.columns = ["date", "open", "high", "low", "close", "volume"]

    _CMP = {
        "Bitcoin": "btc_close",
        "S&P 500": "spx_close",
        "Nasdaq": "ndx_close",
        "זהב": "gld_close",
    }
    compare = None
    if cmp_name in _CMP and _CMP[cmp_name] in panel:
        compare = {"title": cmp_name, "series": panel[_CMP[cmp_name]].dropna()}

    _cone, _mc = None, None
    fc_h = 63
    if show_fc and sym == "mstr" and model_exists():
        fc_h = st.select_slider("אופק החרוט (ימי מסחר)", [21, 42, 63, 126, 252], value=63)
        try:
            _pm = get_model_or_stop()

            @st.cache_data(ttl=900, show_spinner="מחשב חרוט…")
            def _term_cone(h: int, mm: float):
                mc = monte_carlo_paths(
                    _pm, data, horizon=int(h), n_paths=30000, drift_mode="risk_neutral"
                )
                return forecast_cone(mc, data.index.max(), add_trading_days), {
                    "e": mc.expected_price,
                    "p50": mc.median_price,
                    "p_up": mc.prob_up,
                    "s0": mc.last_price,
                }

            _cone, _mc = _term_cone(fc_h, model_mtime())
        except Exception as e:  # noqa: BLE001
            st.caption(f"החרוט לא זמין: {type(e).__name__}")

    _lp = float(ohlc["close"].iloc[-1])
    _tgt = st.number_input(
        "קו מחיר יעד", value=float(round(_lp * 1.1, 2)), step=1.0, help="מצויר כקו אופקי על הגרף"
    )
    _plines = [
        {"price": float(ohlc["close"].tail(252).max()), "title": "52w high", "color": "#8B93B8"},
        {"price": float(ohlc["close"].tail(252).min()), "title": "52w low", "color": "#8B93B8"},
        {"price": _tgt, "title": "יעד", "color": "#FBBF24"},
    ]
    _ev = None
    if sym == "mstr":
        _ev = [
            {
                "time": "2024-08-07",
                "position": "aboveBar",
                "color": "#FBBF24",
                "shape": "arrowDown",
                "text": "10:1 split",
            }
        ]
    elif sym in ("btc", "ibit"):
        from btc_cycle import HALVINGS

        _ev = [
            {
                "time": pd.Timestamp(h).strftime("%Y-%m-%d"),
                "position": "belowBar",
                "color": "#FBBF24",
                "shape": "arrowUp",
                "text": "halving",
            }
            for h in HALVINGS
            if ohlc["date"].min() <= pd.Timestamp(h) <= ohlc["date"].max()
        ]

    st.iframe(
        price_chart_html(
            ohlc,
            title=sym_name.split(" ")[0],
            subtitle=f"{tf} · CARN-X",
            dark=_THEME_DARK,
            chart_type={"נרות": "candles", "HA": "heikin", "קו": "line", "שטח": "area"}[ctype],
            scale="log" if sym in ("btc", "ibit") else "linear",
            overlays=["EMA21", "SMA50", "SMA200"],
            panes=["rsi"],
            forecast=_cone,
            compare=compare,
            events=_ev,
            price_lines=_plines,
        ),
        height=680,
    )

    # --- data window (always visible, Python-computed) ---
    _c = ohlc["close"].to_numpy(float)
    _hi = ohlc["high"].to_numpy(float)
    _lo = ohlc["low"].to_numpy(float)
    _rsi_v = None
    if len(_c) > 15:
        _dd = np.diff(_c)
        _ag = pd.Series(np.clip(_dd, 0, None)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
        _al = pd.Series(np.clip(-_dd, 0, None)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
        _rsi_v = 100 - 100 / (1 + _ag / (_al or 1e-9))
    _tr = np.maximum(_hi[1:] - _lo[1:], np.abs(_hi[1:] - _c[:-1])) if len(_c) > 1 else np.array([])
    _atr = float(pd.Series(_tr).rolling(14).mean().iloc[-1]) if len(_tr) > 14 else np.nan
    _sma200 = float(pd.Series(_c).rolling(200).mean().iloc[-1]) if len(_c) > 200 else np.nan
    _rv = (
        float(pd.Series(np.log(_c)).diff().tail(20).std() * np.sqrt(252))
        if len(_c) > 21
        else np.nan
    )

    dw = st.columns(6)
    dw[0].metric(
        "אחרון", f"${_lp:,.2f}", f"{(_lp / _c[-2] - 1) * 100:+.2f}%" if len(_c) > 1 else None
    )
    dw[1].metric(
        "טווח 52ש'", f"${ohlc['low'].tail(252).min():,.0f}–{ohlc['high'].tail(252).max():,.0f}"
    )
    dw[2].metric("RSI(14)", f"{_rsi_v:.0f}" if _rsi_v is not None else "—")
    dw[3].metric("ATR(14)", f"${_atr:,.2f}" if np.isfinite(_atr) else "—")
    dw[4].metric(
        "מרחק מ-SMA200", f"{(_lp / _sma200 - 1) * 100:+.1f}%" if np.isfinite(_sma200) else "—"
    )
    dw[5].metric("תנודתיות 20ד' (שנתי)", f"{_rv:.0%}" if np.isfinite(_rv) else "—")

    if _mc:
        st.divider()
        fw = st.columns(4)
        fw[0].metric(
            "תוחלת המודל בסוף החרוט", f"${_mc['e']:,.2f}", f"{_mc['e'] / _mc['s0'] - 1:+.1%}"
        )
        fw[1].metric("חציון", f"${_mc['p50']:,.2f}", f"{_mc['p50'] / _mc['s0'] - 1:+.1%}")
        fw[2].metric("P(עלייה) לאורך החרוט", f"{_mc['p_up']:.0%}")
        if model_exists() and _tgt >= _lp:
            _pt = float(
                monte_carlo_paths(
                    _pm,
                    data,
                    horizon=fc_h,
                    n_paths=20000,
                    drift_mode="risk_neutral",
                    target_price=float(_tgt),
                ).p_touch_up
            )
            fw[3].metric(f"P(נגיעה ${_tgt:,.0f})", f"{_pt:.0%}")
        else:
            fw[3].metric("P(נגיעה)", "יעד מתחת למחיר" if _tgt < _lp else "—")
        st.caption(
            "החרוט ניטרלי-סיכון (μ = r). לתרחישים אחרים (μ מותאם / היסטורי / מחזור BTC) "
            "— מסך Monte Carlo. הכיוון עצמו אינו נחזה — ראה 'ראיות Walk-Forward'."
        )


# ===========================================================================
# 4. MONTE CARLO
# ===========================================================================
elif section == "Monte Carlo":
    import altair as alt
    from prob_models import add_trading_days, trading_days_between

    hero(
        "Monte Carlo · חרוט מחיר",
        "סימולציית מסלולים עם זנבות Student-t, antithetic variates ובורר מצב drift",
        "Monte Carlo",
    )
    pm = get_model_or_stop()

    st.caption(
        f"S₀ = מחיר הסגירה המתואם-פיצול האחרון: **${last_price:,.2f}**  "
        f"(auto_adjust=True, {data.index.max().date()})"
    )

    # --- simulation engine selector --------------------------------------
    ENG_HYBRID = "מבני-נוירוני היברידי  ·  BTC × BTC-per-share × mNAV × רפלקסיביות"
    ENG_STAT = "סטטיסטי טהור  ·  GARCH + Student-t SDE"
    engine = st.radio(
        "מנוע הסימולציה",
        [ENG_HYBRID, ENG_STAT],
        index=0,
        help="ההיברידי מדמה את מה ש-MSTR *הוא* — תביעה ממונפת ורפלקסיבית "
        "על ביטקוין — ומנטרל את ה-volatility drag על מסלולי שוק שורי "
        "מחזוריים. הסטטיסטי הוא lognormal קלאסי סביב תחזית הרשת.",
    )
    is_hybrid = engine == ENG_HYBRID

    # --- horizon: slider <-> target-date bi-directional sync ---------------
    HMIN, HMAX, HDEF = 5, 504, 155
    _today = pd.Timestamp.now()
    if "mc_h" not in st.session_state:
        st.session_state.mc_h = HDEF
        st.session_state.mc_date = add_trading_days(_today, HDEF).date()

    def _sync_from_date():
        st.session_state.mc_h = int(
            np.clip(trading_days_between(_today, st.session_state.mc_date), HMIN, HMAX)
        )

    def _sync_from_slider():
        st.session_state.mc_date = add_trading_days(_today, st.session_state.mc_h).date()

    _preset_days = {"1M": 21, "3M": 63, "6M": 126, "אפריל 2027": 155, "1Y": 252, "2Y": 504}
    _pick = st.pills(
        "טווח מהיר",
        list(_preset_days),
        selection_mode="single",
        default=None,
        key="mc_preset",
        label_visibility="collapsed",
    )
    if _pick and st.session_state.mc_h != _preset_days[_pick]:
        d = _preset_days[_pick]
        st.session_state.mc_h = d
        st.session_state.mc_date = add_trading_days(_today, d).date()
        st.rerun()

    c = st.columns([2, 1.4, 1.2, 1.2])
    horizon = c[0].slider(
        "אופק (ימי מסחר)",
        HMIN,
        HMAX,
        key="mc_h",
        on_change=_sync_from_slider,
        help="עד 504 ימים ≈ שנתיים מסחר. חגי NYSE מנוכים.",
    )
    c[1].date_input(
        "תאריך יעד",
        key="mc_date",
        min_value=(_today + pd.Timedelta(days=1)).date(),
        on_change=_sync_from_date,
    )
    n_paths = c[2].select_slider("מסלולים", [5000, 10000, 20000, 50000, 100000], value=50000)
    target = c[3].number_input("מחיר יעד לבדיקה", value=float(round(last_price * 1.1, 2)), step=1.0)

    st.divider()
    if is_hybrid:
        # --- Hybrid structural-neural scenario controls -----------------
        import hybrid_engine as HYB

        btc_series = panel["btc_close"].reindex(close.index).ffill().dropna()
        btc_now = float(btc_series.iloc[-1])

        st.markdown(
            "**הנחות התרחיש** (BTC ×  mNAV  ×  אגרסיביות ATM). המחיר נבנה "
            "מ-`(BTC-per-share · BTC − Debt-per-share) · mNAV` בכל צעד."
        )
        b = st.columns(3)
        btc_exp = b[0].slider(
            "BTC צפוי באופק ($)",
            int(btc_now * 0.5),
            400_000,
            int(round(btc_now * 1.3 / 5000) * 5000),
            5_000,
            key="hyb_btc_exp",
        )
        btc_lo = b[1].slider(
            "BTC — קצה נמוך (~P10)",
            int(btc_now * 0.35),
            int(btc_exp),
            int(round(btc_exp * 0.65 / 5000) * 5000),
            5_000,
            key="hyb_btc_lo",
        )
        btc_hi = b[2].slider(
            "BTC — קצה גבוה (~P90)",
            int(btc_exp),
            600_000,
            int(round(btc_exp * 1.7 / 5000) * 5000),
            5_000,
            key="hyb_btc_hi",
        )
        m = st.columns(3)
        mnav_exp = m[0].slider("mNAV צפוי (מכפיל)", 0.8, 2.5, 1.30, 0.05, key="hyb_mnav")
        mnav_rng = m[1].slider("טווח mNAV", 0.6, 3.0, (0.9, 2.2), 0.05, key="hyb_mnav_rng")
        accel = m[2].slider(
            "אגרסיביות ATM / accretion (BTC-yield שנתי)",
            0.0,
            0.20,
            0.08,
            0.01,
            format="%.0f%%",
            key="hyb_accel",
        )
        exp_prior = st.slider(
            "הטיית שעון ה-halving לכיוון expansion",
            0.0,
            1.0,
            0.35,
            0.05,
            key="hyb_expprior",
            help="גבוה בחלון שאחרי ה-halving (2025 → סוף 2026).",
        )

        with st.expander(
            "יסודות מאזן Strategy (הערכות — עדכן מול ה-10Q האחרון)",
            icon=":material/account_balance:",
        ):
            fc = st.columns(4)
            f_btc = fc[0].number_input("החזקות BTC", value=632_000, step=1_000, key="hyb_f_btc")
            f_debt = fc[1].number_input("חוב נטו ($mld)", value=7.2, step=0.1, key="hyb_f_debt")
            f_sh = fc[2].number_input("מניות מדוללות ($mln)", value=284.0, step=1.0, key="hyb_f_sh")
            f_fee = fc[3].number_input(
                "עמלת הנפקה", value=0.003, step=0.001, format="%.3f", key="hyb_f_fee"
            )
        fund = HYB.StrategyFundamentals(
            btc_holdings=float(f_btc),
            net_debt_usd=float(f_debt) * 1e9,
            diluted_shares=float(f_sh) * 1e6,
            issuance_fee=float(f_fee),
        )
        s0_struct = fund.structural_price(btc_now, mnav_exp)
        st.caption(
            f"S₀ מבני = ({fund.btc_per_share():.5f} · ${btc_now:,.0f} − "
            f"${fund.debt_per_share():,.2f}) · {mnav_exp:.2f} = **${s0_struct:,.0f}**"
            f"  ·  מול מחיר שוק ${last_price:,.0f}"
        )

        log_y = st.toggle("ציר Y לוגריתמי", value=False, key="mc_logy")
        with st.spinner("מריץ מנוע היברידי (רג'ימים · jump-diffusion · OU · רפלקסיביות)…"):
            hres = HYB.run_scenario(
                btc_series,
                fund,
                m0_nav=mnav_exp,
                horizon=int(horizon),
                n_paths=int(n_paths),
                btc_expected=float(btc_exp),
                btc_low=float(btc_lo),
                btc_high=float(btc_hi),
                mnav_low=float(mnav_rng[0]),
                mnav_expected=float(mnav_exp),
                mnav_high=float(mnav_rng[1]),
                accretion_yield=float(accel),
                expansion_prior=float(exp_prior),
                target_price=float(target),
                s0_mstr_market=float(last_price),
            )
        mc = hres.to_mc_result()
        if abs(s0_struct / last_price - 1.0) > 0.12:
            st.warning(
                f"פער מבני–שוק: S₀ מבני ${s0_struct:,.0f} מול שוק "
                f"${last_price:,.0f} ({s0_struct / last_price - 1:+.0%}) — "
                "השוק מתמחר mNAV שונה מההנחה. החרוט מעוגן למחיר השוק."
            )
        c4 = st.columns(4)
        c4[0].metric("רג'ים נוכחי", hres.regime_now)
        c4[1].metric("BTC חציוני באופק", f"${np.median(hres.btc_terminal):,.0f}")
        c4[2].metric("mNAV חציוני", f"{np.median(hres.mnav_terminal):.2f}")
        c4[3].metric(
            "BTC-per-share",
            f"{hres.bps_start:.5f} → {np.median(hres.bps_terminal):.5f}",
            f"{np.median(hres.bps_terminal) / hres.bps_start - 1:+.1%}",
        )
        st.caption(
            "תמהיל רג'ימים לאורך המסלול: "
            + " · ".join(f"{k} {v:.0%}" for k, v in hres.regime_mix.items())
        )
    else:
        # --- drift (mu) controller  (statistical engine) ---------------
        d1, d2 = st.columns([1.4, 2])
        drift_label = d1.radio(
            "סחיפה (μ)",
            ["ניטרלי-סיכון (r)", "תרחיש / מותאם", "היסטורי טהור", "תצוגת המודל"],
            help="μ היסטורי שלילי + drag תנודתיות. בחר בסיס סחיפה מפורש.",
        )
        DMAP = {
            "ניטרלי-סיכון (r)": "risk_neutral",
            "תרחיש / מותאם": "custom",
            "היסטורי טהור": "historical",
            "תצוגת המודל": "model",
        }
        drift_mode = DMAP[drift_label]
        rf = 0.045
        custom_mu = 0.0
        if drift_mode == "risk_neutral":
            rf = d2.slider("ריבית חסרת-סיכון r (שנתי)", 0.0, 0.10, 0.045, 0.005, format="%.3f")
            d2.caption("בנצ'מרק פיננסי סטנדרטי: E[Sₜ] = S₀·e^{rT} (תכונת מרטינגייל).")
        elif drift_mode == "custom":
            custom_mu = d2.slider("סחיפה שנתית μ", -0.50, 1.50, 0.30, 0.05, format="%.2f")
            d2.caption("סימולציית תרחיש (למשל שוק שורי 2027). לא תחזית — הנחה.")
        elif drift_mode == "historical":
            d2.warning(
                "μ מחושב מ-504 ימי המסחר האחרונים. תנודתיות גבוהה יוצרת "
                "**median drag** חמור (−½σ²), ולכן חציון נמוך בהרבה מהתוחלת."
            )
        else:
            d2.caption("הסחיפה שהמודל הנוירוני עצמו חוזה. לרוב קרובה לאפס ורועשת.")

        log_y = st.toggle("ציר Y לוגריתמי", value=False, key="mc_logy")
        with st.spinner("מריץ סימולציה (antithetic + Student-t)…"):
            mc = monte_carlo_paths(
                pm,
                data,
                horizon=int(horizon),
                n_paths=int(n_paths),
                drift_mode=drift_mode,
                custom_drift_annual=custom_mu,
                risk_free_rate=rf,
                target_price=float(target),
            )

    # --- headline metrics ------------------------------------------------
    cc = st.columns(4)
    cc[0].metric(
        "תוחלת מחיר E[Sₜ]",
        f"${mc.expected_price:,.2f}",
        f"{mc.expected_price / last_price - 1:+.1%}",
    )
    cc[1].metric(
        "חציון Q50", f"${mc.median_price:,.2f}", f"{mc.median_price / last_price - 1:+.1%}"
    )
    cc[2].metric("P(עלייה)", f"{mc.prob_up:.0%}")
    cc[3].metric(
        "μ שנתי בשימוש", f"{mc.mu_annual:+.1%}", f"σ {mc.sigma_annual:.0%} · ν {mc.nu:.1f}"
    )

    cc = st.columns(4)
    cc[0].metric("VaR 5%", f"${mc.var_5_price:,.2f}")
    cc[1].metric("CVaR 5% (ES)", f"${mc.es_5_price:,.2f}")
    cc[2].metric("VaR 1%", f"${mc.var_1_price:,.2f}")
    cc[3].metric("Max Drawdown חציוני", f"{mc.max_drawdown_p50:.0%}")

    # --- target probabilities ------------------------------------------
    st.subheader(f"הסתברויות מול יעד ${target:,.2f}")
    tp = st.columns(3)
    p_touch = mc.p_touch_up if target >= last_price else mc.p_touch_down
    tp[0].metric(
        "P(נגיעה בשער עד היעד)", f"{p_touch:.1%}", help="First-passage: P(∃t: Sₜ חוצה את היעד)"
    )
    tp[1].metric(
        "P(סגירה מעבר ליעד)",
        f"{mc.p_close_above:.1%}" if target >= last_price else f"{1 - mc.p_close_above:.1%}",
    )
    tp[2].metric(
        "זמן פגיעה צפוי (ימים)",
        "—" if not np.isfinite(mc.expected_hit_day) else f"{mc.expected_hit_day:.0f}",
    )
    st.progress(min(p_touch, 1.0), text=f"הסתברות נגיעה בשער: {p_touch:.1%}")

    gcol, dcol = st.columns([1, 1])
    with gcol:
        st.altair_chart(
            C.gauge(float(p_touch), lo=0.0, hi=1.0, title="P(נגיעה בשער)", label=f"{p_touch:.0%}"),
            width="stretch",
        )
    with dcol:
        _t = mc.terminal_prices
        _lo_b = float((_t < last_price).mean())
        _hi_b = float((_t >= max(target, last_price)).mean())
        _mid_b = max(1.0 - _lo_b - _hi_b, 0.0)
        st.altair_chart(
            C.donut(
                {"מתחת ל-S₀": _lo_b, "בין S₀ ליעד": _mid_b, "מעל היעד": _hi_b},
                title="פילוח המחיר הסופי",
                center=f"{_hi_b:.0%}",
                colors=[C.RED, C.AMBER, C.GREEN],
            ),
            width="stretch",
        )

    # --- fan chart (dynamic axes) --------------------------------------
    x = np.arange(int(horizon) + 1)
    P = mc.percentiles
    cone = pd.DataFrame(
        {"day": x, **{k: P[k] for k in ["p01", "p05", "p25", "p50", "p75", "p95", "p99"]}}
    )
    yscale = alt.Scale(type="log" if log_y else "linear", zero=False, nice=False)

    def _y(f, **kw):
        return alt.Y(f, scale=yscale, **kw)

    base = alt.Chart(cone).encode(x=alt.X("day:Q", title="ימי מסחר קדימה"))
    b99 = base.mark_area(opacity=0.10, color=C.CYAN).encode(
        y=_y("p01:Q", title="מחיר MSTR"), y2="p99:Q"
    )
    b90 = base.mark_area(opacity=0.18, color=C.CYAN).encode(y=_y("p05:Q"), y2="p95:Q")
    b50 = base.mark_area(opacity=0.32, color=C.CYAN).encode(y=_y("p25:Q"), y2="p75:Q")
    median = base.mark_line(strokeWidth=2.5, color="#EAF2FF").encode(y=_y("p50:Q"))
    rules = (
        alt.Chart(pd.DataFrame({"y": [last_price, target], "label": ["מחיר נוכחי", "יעד"]}))
        .mark_rule(strokeDash=[4, 4], color=C.AMBER)
        .encode(
            y=_y("y:Q"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(range=[C.AMBER, C.MAGENTA]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    n_show = min(40, mc.paths.shape[0])
    samp = pd.DataFrame(
        mc.paths[np.random.default_rng(0).choice(mc.paths.shape[0], n_show, replace=False)].T
    )
    samp["day"] = x
    traces = (
        alt.Chart(samp.melt("day", var_name="p", value_name="price"))
        .mark_line(opacity=0.14, strokeWidth=0.5, color=C.VIOLET)
        .encode(x="day:Q", y=_y("price:Q"), detail="p:N")
    )
    st.altair_chart(
        alt.layer(b99, b90, b50, traces, median, rules)
        .resolve_scale(y="shared")
        .properties(height=420),
        width="stretch",
    )

    # --- terminal distribution: histogram + KDE -----------------------
    st.subheader("התפלגות המחיר הסופי")
    term = mc.terminal_prices
    lo, hi = np.percentile(term, [0.5, 99.5])
    term_c = term[(term >= lo) & (term <= hi)]
    counts, edges = np.histogram(term_c, bins=60, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(term_c)
    grid = np.linspace(lo, hi, 200)
    hist_df = pd.DataFrame({"price": centers, "density": counts})
    kde_df = pd.DataFrame({"price": grid, "density": kde(grid)})
    bars = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.45, color=C.VIOLET)
        .encode(
            x=alt.X("price:Q", title="מחיר סופי", scale=alt.Scale(zero=False)),
            y=alt.Y("density:Q", title="צפיפות"),
        )
    )
    curve = (
        alt.Chart(kde_df)
        .mark_line(color=C.CYAN, strokeWidth=2.5)
        .encode(x="price:Q", y="density:Q")
    )
    vlines = (
        alt.Chart(
            pd.DataFrame(
                {
                    "price": [last_price, mc.expected_price, mc.median_price, target],
                    "label": ["S₀", "תוחלת", "חציון", "יעד"],
                }
            )
        )
        .mark_rule(strokeDash=[3, 3])
        .encode(
            x="price:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(range=[C.TEXT, C.GREEN, C.AMBER, C.MAGENTA]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    st.altair_chart((bars + curve + vlines).properties(height=320), width="stretch")

    with st.expander("מומנטים ואחוזונים מלאים"):
        pct_tbl = pd.DataFrame(
            {
                "אחוזון": ["P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99"],
                "מחיר": [
                    P[k][-1]
                    for k in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
                ],
            }
        )
        pct_tbl["מחיר"] = pct_tbl["מחיר"].map(lambda v: f"${v:,.2f}")
        st.dataframe(pct_tbl, hide_index=True, width="stretch")
        st.write(
            f"Skew {mc.terminal_skew:+.2f} · Excess-Kurtosis "
            f"{mc.terminal_kurt:+.2f} · {mc.n_paths:,} מסלולים "
            f"(antithetic) · ν={mc.nu:.2f} · Δt=1/252"
        )


# ===========================================================================
# 4a. MATHEMATICAL STRUCTURES  (analytic companion -- not the neural path)
# ===========================================================================
elif section == "מבנים מתמטיים":
    import math

    import altair as alt

    import math_structures as MS

    hero(
        "מבנים מתמטיים · MSTR",
        "אותה היסטוריה, דרך ענפי המתמטיקה: משולש פסקל, קטלאן, אלגברה לינארית, "
        "משוואות דיפרנציאליות וחשבון סטוכסטי — כל טענה נבדקת מול הנתונים",
        "Analytic companion",
    )
    st.caption(
        "השכבה הזו **אינה** נוגעת ברשת הנוירונים. היא לוקחת את סדרת הסגירות של "
        "MSTR ומבטאת את הסטטיסטיקה שלה במונחים מתמטיים מדויקים — ובודקת מול "
        "צורה סגורה / מונטה-קרלו / התדירות ההיסטורית האמיתית."
    )

    lb = st.slider("חלון היסטוריה (ימי מסחר)", 250, 1200, 504, 21, key="ms_lb")
    hz = st.select_slider("אופק (ימי מסחר)", [5, 10, 20, 40, 60], value=20, key="ms_hz")

    t_pascal, t_catalan, t_lin, t_pde, t_sde, t_path, t_fib, t_puz, t_const = st.tabs(
        [
            "בינום ניוטון · פסקל",
            "קטלאן · נסיגת מחיר",
            "אלגברה לינארית · DMD",
            "מד״ח · Fokker–Planck",
            "חדו״א סטוכסטי · SDE",
            "סטטיסטיקת מסלול",
            "פיבונאצ׳י (מבחן כן)",
            "חידות אסטרטגיות",
            "הקבועים המפורסמים",
        ]
    )

    # ---- Pascal / CRR binomial lattice --------------------------------------
    with t_pascal:
        steps = st.slider("צעדי סריג", 6, 60, min(int(hz), 40), key="ms_steps")
        lat = MS.crr_lattice(close, horizon_days=int(hz), steps=int(steps), lookback=int(lb))
        st.markdown(
            "עץ **Cox–Ross–Rubinstein**: ההסתברות להגיע לצומת הסופי `k` אחרי `n` "
            r"צעדים היא בדיוק $\binom{n}{k}\,p^k(1-p)^{n-k}$ — שורה `n` של משולש "
            "פסקל, משוקללת. כשמספר הצעדים גדל העץ מתכנס ל-Black–Scholes ומופיעים "
            "$e$ (ריבית דריבית רציפה) ו-$\\pi$ (נרמול גאוסיאני)."
        )
        c = st.columns(4)
        c[0].metric("u (צעד למעלה)", f"{lat.u:.4f}")
        c[1].metric("p* (ניטרלי-סיכון)", f"{lat.p_riskneutral:.3f}")
        c[2].metric("E[S_T] ניטרלי-סיכון", f"${lat.expected_terminal_rn:,.1f}")
        c[3].metric(
            "S₀·e^{rT} (גבול רציף)",
            f"${lat.bs_lognormal_mean:,.1f}",
            f"{lat.expected_terminal_rn / lat.bs_lognormal_mean - 1:+.1e}",
        )

        tri = MS.pascal_triangle(min(int(steps) + 1, 22))
        tdf = pd.DataFrame(
            [(n, k, float(tri[n, k])) for n in range(tri.shape[0]) for k in range(n + 1)],
            columns=["n", "k", "coef"],
        )
        st.altair_chart(
            C.heatmap(
                tdf,
                x="k",
                y="n",
                val="coef",
                scheme="magma",
                title="משולש פסקל — מקדמי הבינום",
                height=280,
            ),
            width="stretch",
        )

        term = pd.DataFrame(
            {
                "price": lat.terminal_prices,
                "p_real": lat.terminal_probs_rw,
                "p_rn": lat.terminal_probs_rn,
            }
        )
        st.altair_chart(
            C.stem(
                term,
                x="price",
                y="p_real",
                percent=True,
                title=f"התפלגות המחיר בעוד {int(hz)} ימים — שורת פסקל המשוקללת (הסתברות אמיתית)",
                x_title="מחיר ($)",
                color=C.CYAN,
            ),
            width="stretch",
        )
        if lat.empirical_quantile_overlay:
            st.caption(
                "צ׳ק היסטורי — חמישוני המחיר בפועל לאותו אופק: "
                + " · ".join(f"{k}: ${v:,.0f}" for k, v in lat.empirical_quantile_overlay.items())
            )

        conv = MS.binomial_to_normal_convergence(p=lat.p_realworld)
        cdf = pd.DataFrame({"n": conv.n_grid, "TV": conv.total_variation})
        st.altair_chart(
            C.area_gradient(
                pd.Series(cdf["TV"].values, index=cdf["n"].values),
                color=C.VIOLET,
                height=200,
                y_title="‖Binom − Normal‖",
            ),
            width="stretch",
        )
        st.caption(f"e: {conv.e_appears_as}  ·  π: {conv.pi_appears_as}")

        # --- three views of the same forecast: Pascal lattice · history · model ---
        st.markdown(
            "**שלוש זוויות על אותה תחזית** — הסריג הקומבינטורי, ההיסטוריה, "
            "והרשת הנוירונית — במרחב לוג-התשואה:"
        )
        s0 = lat.s0
        xg = np.linspace(
            np.log(lat.terminal_prices[0] / s0), np.log(lat.terminal_prices[-1] / s0), 240
        )
        from scipy.stats import gaussian_kde as _gkde

        lays = []
        # (1) CRR lattice terminal density (Pascal row, smoothed for display)
        lat_lr = np.log(lat.terminal_prices / s0)
        lat_kde = _gkde(lat_lr, weights=lat.terminal_probs_rw, bw_method=0.35)
        lays.append(pd.DataFrame({"logret": xg, "density": lat_kde(xg), "מקור": "סריג פסקל (CRR)"}))
        # (2) empirical h-day log-returns over the lookback
        _r = np.log(close.astype(float)).diff().dropna().to_numpy()[-int(lb) :]
        if len(_r) > int(hz) + 30:
            _hr = pd.Series(_r).rolling(int(hz)).sum().dropna().to_numpy()
            lays.append(pd.DataFrame({"logret": xg, "density": _gkde(_hr)(xg), "מקור": "היסטורי"}))
        # (3) the neural model's Student-t predictive density (nearest horizon)
        if model_exists():
            try:
                _pm = load_model_cached(data, model_mtime())
                _dist = predict_distribution(_pm, data)
                _mh = int(min(_dist.horizon, key=lambda h: abs(h - int(hz))))
                _row = _dist[_dist.horizon == _mh].iloc[0]
                from scipy.stats import t as _tt

                _d = _tt.pdf((xg - _row.mu_raw) / _row.sigma_raw, df=_row.nu) / _row.sigma_raw
                lays.append(
                    pd.DataFrame({"logret": xg, "density": _d, "מקור": f"מודל CARN-X (h={_mh})"})
                )
            except Exception:
                pass
        cmp_df = pd.concat(lays, ignore_index=True)
        st.altair_chart(
            alt.Chart(cmp_df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("logret:Q", title="לוג-תשואה"),
                y=alt.Y("density:Q", title="צפיפות"),
                color=alt.Color(
                    "מקור:N",
                    scale=alt.Scale(range=[C.CYAN, C.AMBER, C.MAGENTA]),
                    legend=alt.Legend(orient="top", title=None),
                ),
            )
            .properties(height=280),
            width="stretch",
        )
        st.caption(
            "אם שלוש העקומות דומות — הסריג הפשוט תופס את מה שהמודל והשוק "
            "אומרים. פערים = היכן שהזנבות/הא-סימטריה חשובים."
        )

    # ---- Catalan / drawdown survival --------------------------------------
    with t_catalan:
        floor = st.slider("רצפת drawdown", 0.05, 0.50, 0.20, 0.01, format="%.0f%%", key="ms_floor")
        ds = MS.drawdown_survival(
            close, floor_pct=float(floor), horizon_days=int(hz), lookback=int(lb)
        )
        st.markdown(
            "מסלול מחיר שלא שובר רצפה הוא **מסלול סריג לא-שלילי** — נספר במדויק "
            r"על-ידי מספרי קטלאן $C_n=\frac{1}{n+1}\binom{2n}{n}$ (עקרון השיקוף / "
            "בעיית הקלפי). כאן: ההסתברות ש-MSTR *לא* תרד יותר מהרצפה תוך האופק."
        )
        c = st.columns(4)
        c[0].metric("קומבינטורי (סימטרי)", f"{ds.p_survive_combinatorial:.1%}")
        c[1].metric("עם דריפט (שיקוף)", f"{ds.p_survive_drifted:.1%}")
        c[2].metric("מונטה-קרלו (bootstrap)", f"{ds.p_survive_montecarlo:.1%}")
        c[3].metric(
            "תדירות היסטורית",
            f"{ds.p_survive_empirical:.1%}" if ds.p_survive_empirical is not None else "—",
        )
        st.altair_chart(
            C.hbar_ranked(
                pd.DataFrame(
                    {
                        "method": ["קומבינטורי (סימטרי)", "עם דריפט", "מונטה-קרלו", "היסטורי"],
                        "p": [
                            ds.p_survive_combinatorial,
                            ds.p_survive_drifted,
                            ds.p_survive_montecarlo,
                            ds.p_survive_empirical or 0.0,
                        ],
                    }
                ),
                cat="method",
                val="p",
                percent=True,
                descending=False,
                title="הסתברות הישרדות — ארבע שיטות",
            ),
            width="stretch",
        )
        st.caption(ds.note + f"  ·  C_{int(hz) // 2} = {ds.catalan_number:,}")

    # ---- Linear algebra / DMD -------------------------------------------
    with t_lin:
        rk = st.slider("דרגת DMD", 4, 16, 10, key="ms_rank")
        dl = st.slider("עומק השהיה (Hankel)", 10, 60, 30, key="ms_delay")
        dm = MS.dmd_spectrum(
            close, rank=int(rk), delay=int(dl), lookback=int(lb), horizon_days=int(hz)
        )
        st.markdown(
            "**Dynamic Mode Decomposition** — האופרטור הליניארי $A$ הטוב ביותר "
            "עם $x_{t+1}\\approx A\\,x_t$ על שיכון-השהיה של לוג-המחיר (מנוכה מגמה). "
            "הערכים העצמיים שלו הם מודי קופמן: $|\\lambda|<1$ דועך (חזרה לממוצע), "
            "$|\\lambda|\\approx1$ מתמיד, $|\\lambda|>1$ מתפוצץ מקומית; "
            "$\\arg\\lambda$ נותן מחזור בימי מסחר."
        )
        c = st.columns(4)
        c[0].metric("רדיוס ספקטרלי ρ", f"{dm.spectral_radius:.3f}")
        c[1].metric("R² שחזור", f"{dm.reconstruction_r2:.2f}")
        c[2].metric(
            "מחזור דומיננטי",
            f"{dm.dominant_period_days:.0f}d" if math.isfinite(dm.dominant_period_days) else "—",
        )
        c[3].metric(f"תחזית {int(hz)}d (מודלי)", f"{dm.forecast[-1]:+.1%}")
        edf = pd.DataFrame(
            {"re": dm.eigenvalues.real, "im": dm.eigenvalues.imag, "amp": dm.mode_amplitude}
        )
        st.altair_chart(
            C.complex_plane(
                edf, title="ערכים עצמיים על המישור המרוכב (גודל = משרעת המוד)", height=340
            ),
            width="stretch",
        )
        per = dm.period_days
        pdf = pd.DataFrame(
            {"period_d": per[np.isfinite(per)], "amp": dm.mode_amplitude[np.isfinite(per)]}
        ).sort_values("period_d")
        if len(pdf):
            st.altair_chart(
                C.stem(
                    pdf,
                    x="period_d",
                    y="amp",
                    title="מחזורים אוסצילטוריים שזוהו",
                    x_title="מחזור (ימים)",
                    color=C.MAGENTA,
                ),
                width="stretch",
            )
        st.caption(
            dm.note + "  ·  התחזית המודלית היא **אקסטרפולציה ליניארית** של "
            "המודים — לא עברה walk-forward, אל תסחור לפיה. הערך הוא בזיהוי "
            "המבנה (מחזורים, יציבות), לא במספר."
        )

    # ---- PDE / Fokker-Planck ------------------------------------------
    with t_pde:
        lv = st.toggle("תנודתיות תלוית-מצב (Nadaraya–Watson מכווץ)", value=True, key="ms_lv")
        fp = MS.fokker_planck_density(
            close, horizon_days=int(hz), lookback=int(lb), local_vol=bool(lv)
        )
        st.markdown(
            "צפיפות לוג-התשואה הקדמית $p(x,t)$ מפותחת בזמן לפי משוואת "
            "**קולמוגורוב הקדמית / Fokker–Planck**:"
        )
        st.latex(
            r"\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}"
            r"\big[a(x)\,p\big] + \frac12\frac{\partial^2}{\partial x^2}"
            r"\big[b(x)^2\,p\big]"
        )
        c = st.columns(4)
        c[0].metric(f"E[logret] בעוד {int(hz)}d", f"{fp.terminal_mean_logret:+.4f}")
        c[1].metric("סטיית תקן", f"{fp.terminal_std_logret:.4f}")
        c[2].metric("שימור מסה |1−∫p|", f"{fp.mass_drift:.1e}")
        c[3].metric("בדיקת גאוס (מק׳ קבוע)", f"{fp.analytic_gaussian_check:.1e}")
        k = max(1, len(fp.x_grid) // 90)  # keep the row count sane
        cols = np.arange(0, len(fp.x_grid), k)
        surf = pd.DataFrame(
            [
                (int(fp.t_grid[i]), float(fp.x_grid[j]), float(fp.density[i, j]))
                for i in range(len(fp.t_grid))
                for j in cols
            ],
            columns=["day", "logret", "p"],
        )
        st.altair_chart(
            C.surface_heat(
                surf,
                x="day",
                y="logret",
                val="p",
                scheme="inferno",
                title="התפתחות הצפיפות p(x, t) — פתרון נומרי של המד״ח",
                x_title="יום",
                y_title="לוג-תשואה",
            ),
            width="stretch",
        )
        st.caption(
            fp.note + ("  ·  תנודתיות תלוית-מצב" if fp.local_vol_used else "  ·  מקדמים קבועים")
        )

    # ---- SDE / jump-diffusion ----------------------------------------
    with t_sde:
        jd = MS.fit_jump_diffusion(close, lookback=min(int(lb) + 252, 900))
        st.markdown(
            "מודל **Merton jump-diffusion** מותאם ב-EM: "
            r"$dS/S = \mu\,dt + \sigma\,dW + (J-1)\,dN$,  $N\sim\text{Poisson}(\lambda)$. "
            "התשואות היומיות הן תערובת של גרעין צר (דיפוזיה) וזנב רחב (קפיצות)."
        )
        c = st.columns(4)
        c[0].metric("μ (דריפט/שנה)", f"{jd.mu_drift_annual:+.1%}")
        c[1].metric("σ דיפוזיה/שנה", f"{jd.sigma_diffusion_annual:.0%}")
        c[2].metric("λ קפיצות/שנה", f"{jd.jump_intensity_annual:.1f}")
        c[3].metric("מובהקות (LR)", f"p={jd.lr_pvalue:.1e}")
        st.markdown(
            f"**השלד הדטרמיניסטי (ה-ODE):**  `{jd.deterministic_skeleton}`  "
            f"— צמיחה מעריכית בקצב הדריפט; ה-SDE מוסיף רעש יומי וקפיצות פואסון."
        )
        st.caption(
            jd.note
            + ("  ·  ימי קפיצה אחרונים: " + ", ".join(jd.jump_days[-6:]) if jd.jump_days else "")
        )

    # ---- Path statistics --------------------------------------------
    with t_path:
        ps = MS.path_statistics(close, lookback=int(lb))
        st.markdown(
            "מבחני קומבינטוריקה של מסלול: **Wald–Wolfowitz** (האם רצף העליות/"
            "ירידות אקראי?), **חוק הארקסינוס של לוי** (שבר הזמן מעל ההתחלה), "
            "והרצף הרצוף הארוך ביותר מול Binomial."
        )
        c = st.columns(4)
        c[0].metric("P(יום עולה)", f"{ps.up_day_prob:.1%}", f"מבחן בינום p={ps.up_day_binom_p:.2f}")
        c[1].metric("Runs test", f"z={ps.runs_z:+.2f}", f"p={ps.runs_pvalue:.3f}")
        c[2].metric("רצף עליות / ירידות ארוך", f"{ps.longest_up_run} / {ps.longest_down_run}")
        c[3].metric(
            "שבר הזמן מעל ההתחלה",
            f"{ps.frac_time_at_high:.1%}",
            f"arcsine p={ps.arcsine_pvalue:.2f}",
        )
        st.caption(ps.note)

    # ---- Fibonacci honest test ------------------------------------
    with t_fib:
        st.markdown(
            "רמות פיבונאצ׳י הן **קונבנציית תרשים**, לא סיגנל מותאם. המבחן הכן: "
            "האם נקודות מפנה מתקבצות ליד 0.382 / 0.5 / 0.618 יותר מאשר null מעורבל?"
        )
        fl = MS.fibonacci_levels(close, swing_lookback=min(int(lb), 160))
        ft = MS.fibonacci_pivot_test(close, lookback=min(int(lb) * 2, 1200))
        c = st.columns(4)
        c[0].metric("מפנים ליד רמה", f"{ft.n_near_fib}/{ft.n_pivots}")
        c[1].metric("שיעור פגיעה", f"{ft.hit_rate:.1%}", f"null {ft.null_hit_rate:.1%}")
        c[2].metric("p (פרמוטציה)", f"{ft.p_value:.3f}")
        c[3].metric("רמה קרובה כעת", fl.nearest_level, f"{fl.distance_pct:.1%}")
        st.altair_chart(
            C.line_with_levels(
                close.tail(min(int(lb), 160)),
                fl.levels,
                title=f"MSTR עם רמות פיבונאצ׳י ({fl.direction})",
            ),
            width="stretch",
        )
        (st.success if ft.p_value < 0.05 else st.info)(ft.verdict)

    # ---- Strategic puzzles ---------------------------------------
    with t_puz:
        pm = MS.strategic_puzzle_map(close, lookback=int(lb))
        st.markdown("החידות שהזכרת הן האריתמטיקה של שכבת ההחלטה:")
        c = st.columns(3)
        c[0].metric(
            "ביצים — בדיקות probe",
            f"{pm.egg_drop_probes}",
            f"למקד תמיכה ל-{pm.egg_drop_tol_pct:.0%}",
        )
        c[1].metric("משקולות — אופק אופטימלי", f"{pm.ternary_best_horizon}d", "יחס סיגנל/רעש מרבי")
        c[2].metric(
            "מדרגות — רמות מחיר",
            f"{pm.staircase_reachable_states}",
            f"מ-{pm.staircase_distinct_paths:,} מסלולים",
        )
        bz = pm.bezout_alignment
        st.markdown(
            f"**שעוני חול / Bézout (GCD):** מחזור BTC ≈ {bz['btc_cycle_days']} ימים, "
            f"מחזור DMD דומיננטי של MSTR ≈ {bz['mstr_dominant_period_days']} ימים  →  "
            f"gcd = {bz['gcd_days']} ימים, מחזור פעימה ≈ "
            f"{bz['beat_period_days']:,} ימים."
            if bz["beat_period_days"]
            else f"**Bézout:** מחזור BTC ≈ {bz['btc_cycle_days']} ימים."
        )
        tdf = pd.DataFrame(
            {
                "horizon": list(pm.ternary_horizon_tstat),
                "tstat": list(pm.ternary_horizon_tstat.values()),
            }
        )
        st.altair_chart(
            C.hbar_ranked(
                tdf, cat="horizon", val="tstat", title="עוצמת סיגנל הדריפט לפי אופק (t-stat)"
            ),
            width="stretch",
        )
        st.caption(pm.note)

    # ---- Constants honesty panel ------------------------------
    with t_const:
        st.markdown("**היכן הקבועים המפורסמים באמת מופיעים — ואיפה לא:**")
        st.dataframe(pd.DataFrame(MS.math_constants_in_finance()), width="stretch", hide_index=True)
        st.caption(
            "φ, מספרים ראשוניים ו-π *כדפוס בגרף* אינם נושאים מידע חיזוי — "
            "ראה את מבחן הפיבונאצ׳י. e ו-π כן מופיעים מבנית: e בגבול הרציף של "
            "עץ הבינום, π בנרמול הגאוסיאני של כל רווח-סמך."
        )


# ===========================================================================
# 4b. BTC 4-YEAR CYCLE -> MSTR  (structural long-horizon scenario)
# ===========================================================================
elif section == "מחזור ביטקוין → MSTR":
    import altair as alt
    from btc_cycle import (
        HALVINGS,
        fit_cycle_template,
        load_btc_history,
        macro_state,
        mstr_path_from_btc,
        project_btc,
    )

    hero(
        "מחזור הביטקוין (4 שנים) → MSTR",
        "תרחיש מבני ארוך-טווח: מחזור ה-halving של BTC ממונף דרך MSTR",
        "BTC Cycle",
    )
    st.caption(
        'MSTR הוא מבנית החזקת ביטקוין ממונפת. הנתיב הרב-חודשי שלה נשלט ע"י BTC, '
        "והנתיב הרב-שנתי של BTC אורגן היסטורית סביב מחזור ה-halving (~4 שנים). "
        "זה **תרחיש מבני**, נפרד מהמודל הנוירוני קצר-הטווח — יש רק 2–3 מחזורים "
        "קודמים והמחזור הנוכחי חלש משניהם, אז **הרוחב הוא המסר, לא החציון**."
    )

    @st.cache_resource(show_spinner="טוען היסטוריית BTC…")
    def _btc_hist(bust: int = 0):
        try:
            s = load_btc_history()
        except Exception:  # noqa: BLE001
            s = pd.Series(dtype=float)
        if s is None or len(s) < 400:  # yfinance throttled -> fall back to the cached panel
            s = panel["btc_close"].dropna()
            s = s[s > 0]
        return s, fit_cycle_template(s)

    btc_hist, tmpl = _btc_hist()
    if len(btc_hist) < 400:
        st.error(
            "היסטוריית BTC לא זמינה (Yahoo מגביל קצב) — נסה שוב מאוחר יותר או "
            "רענן נתוני שוק בהגדרות."
        )
        st.stop()
    ms = macro_state(panel)

    h_now = max(h for h in HALVINGS if h <= btc_hist.index[-1])
    age_now = (btc_hist.index[-1] - h_now).days
    c = st.columns(4)
    c[0].metric("BTC", f"${btc_hist.iloc[-1]:,.0f}")
    c[1].metric("ימים מאז ה-halving", f"{age_now}", f"halving הבא ~{HALVINGS[-1].year + 4}")
    c[2].metric("שלב במחזור", f"{age_now / 1400:.0%}")
    c[3].metric(
        "מדד מאקרו (risk)",
        f"{ms.risk_score:+.2f}",
        "risk-off" if ms.risk_score < -0.2 else "risk-on" if ms.risk_score > 0.2 else "ניטרלי",
        delta_color="off",
    )

    from tv_chart import price_chart_html as _tv

    _bc = ["btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]
    _bohlc = panel[_bc].dropna().tail(2600).reset_index()
    _bohlc.columns = ["date", "open", "high", "low", "close", "volume"]
    _hmarks = [
        {
            "time": h.strftime("%Y-%m-%d"),
            "position": "belowBar",
            "color": "#FBBF24",
            "shape": "arrowUp",
            "text": "halving",
        }
        for h in HALVINGS
        if _bohlc["date"].min() <= pd.Timestamp(h) <= _bohlc["date"].max()
    ]
    st.subheader("Bitcoin — טרמינל גרפים (סקאלה לוגריתמית)")
    st.iframe(
        _tv(
            _bohlc,
            title="BTC",
            subtitle="יומי · חצאי-מחיר מסומנים",
            dark=_THEME_DARK,
            scale="log",
            overlays=["SMA50", "SMA200"],
            panes=["rsi"],
            events=_hmarks,
        ),
        height=520,
    )
    gcol = st.columns([1, 2])[0]
    with gcol:
        st.altair_chart(
            C.gauge(
                min(age_now / 1400.0, 1.0),
                lo=0.0,
                hi=1.0,
                title="שלב במחזור ה-halving",
                label=f"{age_now / 1400:.0%}",
                zones=((0.0, 0.4, C.CYAN), (0.4, 0.75, C.AMBER), (0.75, 1.0, C.RED)),
            ),
            width="stretch",
        )

    cc = st.columns(3)
    tgt = cc[0].date_input("תאריך יעד", value=pd.Timestamp("2027-04-30")).isoformat()
    w_cur = cc[1].slider("משקל למחזור הנוכחי מול התבנית ההיסטורית", 0.0, 1.0, 0.6, 0.05)
    use_macro = cc[2].checkbox("שקלל מצב מאקרו/מיקרו", value=True)

    try:
        proj = project_btc(
            btc_hist, tgt, tmpl, current_cycle_weight=w_cur, macro=ms if use_macro else None
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()
    mproj = mstr_path_from_btc(
        panel["mstr_close"].dropna(), panel["btc_close"].dropna(), proj, beta_window=252
    )

    st.subheader(f"תחזית ל-{tgt}  ({proj.horizon_days} ימים)")
    a, b = st.columns(2)
    with a:
        st.markdown("**Bitcoin**")
        st.metric(
            "חציון",
            f"${proj.median_price:,.0f}",
            f"{np.expm1(proj.median_logret) * 100:+.0f}% מהיום",
        )
        st.caption(f"טווח 80%: ${proj.low_price:,.0f} – ${proj.high_price:,.0f}")
    with b:
        st.markdown("**MSTR**")
        st.metric(
            "חציון",
            f"${mproj.median_price:,.0f}",
            f"{(mproj.median_price / mproj.last_mstr - 1) * 100:+.0f}% מהיום",
        )
        st.caption(
            f"טווח 80%: ${mproj.low_price:,.0f} – ${mproj.high_price:,.0f}  ·  "
            f"beta ל-BTC {mproj.beta:.2f}"
        )
    st.caption(proj.template_used)
    if use_macro:
        for n in ms.notes:
            st.caption(f"· {n}")

    # --- cone chart (MSTR) ---
    p = mproj.daily_path.copy()
    hist_m = panel["mstr_close"].dropna().tail(240).reset_index()
    hist_m.columns = ["date", "price"]
    base = alt.Chart(p).encode(x=alt.X("date:T", title=None))
    band = base.mark_area(opacity=0.20, color=C.MAGENTA).encode(
        y=alt.Y("p_low:Q", title="MSTR ($)", scale=alt.Scale(zero=False)), y2="p_high:Q"
    )
    med = base.mark_line(color=C.MAGENTA, strokeDash=[5, 3], strokeWidth=2).encode(y="p_median:Q")
    histline = (
        alt.Chart(hist_m)
        .mark_line(color="#EAF2FF", strokeWidth=1.8)
        .encode(x="date:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)))
    )
    st.altair_chart((histline + band + med).resolve_scale(y="shared"), width="stretch")

    # --- the aligned historical cycles ---
    st.subheader("שלושת המחזורים הקודמים, מיושרים לפי ימים מה-halving")
    rows = []
    for k, path in tmpl.per_cycle.items():
        for age, lr in zip(tmpl.ages, path):
            if np.isfinite(lr):
                rows.append({"מחזור": k, "יום": age, "מכפיל": float(np.exp(lr))})
    cyc_df = pd.DataFrame(rows)
    chart = (
        alt.Chart(cyc_df)
        .mark_line()
        .encode(
            x=alt.X("יום:Q", title="ימים מה-halving"),
            y=alt.Y("מכפיל:Q", scale=alt.Scale(type="log"), title="מכפיל ממחיר ה-halving"),
            color="מחזור:N",
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"יום": [age_now]}))
        .mark_rule(color="grey", strokeDash=[4, 4])
        .encode(x="יום:Q")
    )
    st.altair_chart(chart + rule, width="stretch")
    st.caption("הקו האפור = היכן שאנחנו במחזור הנוכחי. שים לב שהמשרעת קטנה כל מחזור.")

    st.markdown("---")
    st.subheader('תרחיש מבני: "אם BTC יגיע ל-X"')
    st.caption(
        "כאן אתה קובע את הנחת ה-BTC. המערכת מחשבת את מחיר MSTR המשתמע "
        "(דרך beta ופרמיית mNAV) **ואת ההסתברות שה-BTC אכן יגיע לשם** לפי המודל המחזורי."
    )
    sc = st.columns(3)
    btc_now = float(btc_hist.iloc[-1])
    btc_target = sc[0].slider(
        "יעד BTC ($)", int(btc_now * 0.4), 400_000, int(round(btc_now * 1.5 / 1000) * 1000), 5_000
    )
    mnav_prem = sc[1].slider(
        "פרמיית mNAV (מכפיל)",
        0.7,
        2.5,
        1.3,
        0.05,
        help="1.0 = בשווי נכסים · 1.5–2.0 = הטווח ההיסטורי בשוק שוורי",
    )
    from btc_cycle import simulate_btc_paths as _sbp

    _bpaths = _sbp(
        btc_hist,
        proj.horizon_days,
        tmpl,
        n_paths=20000,
        current_cycle_weight=w_cur,
        macro=ms if use_macro else None,
    )
    p_btc_reach = float((_bpaths.max(axis=1) >= btc_target).mean())
    implied_mstr = mproj.last_mstr * (btc_target / btc_now) ** mproj.beta * (mnav_prem / 1.0)
    d = sc[2].columns(1)[0]
    st.markdown(
        f"**אם** BTC יגיע ל-${btc_target:,.0f} (×{btc_target / btc_now:.1f} מהיום) "
        f"**ופרמיית mNAV** = {mnav_prem:.2f}x  →  **MSTR ≈ ${implied_mstr:,.0f}** "
        f"(×{implied_mstr / mproj.last_mstr:.1f} מהיום)"
    )
    st.progress(
        min(p_btc_reach, 1.0),
        text=f"P(BTC נוגע ב-${btc_target:,.0f} עד {tgt}) לפי המודל המחזורי = {p_btc_reach:.0%}",
    )
    st.caption(
        f"beta MSTR↔BTC = {mproj.beta:.2f}. שים לב: התרחיש (המחיר המשתמע) הוא **הנחה שלך**; "
        f"רק ההסתברות למעלה מגיעה מהמודל."
    )

    st.info(
        "מגבלות: 2–3 מחזורים בלבד; המחזור הנוכחי רץ ב-~30% מהמשרעת ההיסטורית; "
        "תיאוריית ה-4-שנים עשויה להיחלש עם כניסת ה-ETFים. אל תתייחס לחציון כאל יעד."
    )


# ===========================================================================
# 5. WALK-FORWARD EVIDENCE
# ===========================================================================
elif section == "ראיות Walk-Forward":
    hero(
        "ראיות out-of-sample",
        "כיול, דיוק כיווני ו-skill מול baselines על נתונים שהמודל לא ראה",
        "L8 · Evidence",
    )
    runs = sorted(glob.glob("runs/*/oos_predictions.parquet"), key=os.path.getmtime, reverse=True)
    if not runs:
        st.warning("אין ריצות. הרץ:  `.venv/bin/python run_evaluation.py`")
        st.stop()

    choice = st.selectbox("ריצה", runs, format_func=lambda p: p.split("/")[1])
    oos = pd.read_parquet(choice)
    from evaluation import evaluate

    res = evaluate(oos)
    st.caption(f"{res['n_oos_rows']} תחזיות OOS · {res['date_range'][0]} → {res['date_range'][1]}")

    _hstat = {str(h): b["statistical"] for h, b in res["horizons"].items()}
    hb = st.columns(2)
    with hb[0]:
        st.altair_chart(
            C.hbar_ranked(
                pd.DataFrame(
                    {"אופק": list(_hstat), "דיוק": [v["dir_acc_mean"] for v in _hstat.values()]}
                ),
                cat="אופק",
                val="דיוק",
                percent=True,
                height=170,
                title="דיוק כיוון לפי אופק",
                descending=False,
            ),
            width="stretch",
        )
    with hb[1]:
        st.altair_chart(
            C.hbar_ranked(
                pd.DataFrame(
                    {"אופק": list(_hstat), "כיסוי": [v["coverage_90"] for v in _hstat.values()]}
                ),
                cat="אופק",
                val="כיסוי",
                percent=True,
                height=170,
                title="כיסוי טווח 90% (יעד 0.90)",
                descending=False,
            ),
            width="stretch",
        )

    for h, block in res["horizons"].items():
        s = block["statistical"]
        st.subheader(f"אופק {h} ימים")
        cc = st.columns(5)
        cc[0].metric("דיוק כיוון", f"{s['dir_acc_mean']:.1%}", f"p={s['dir_binom_p']:.2f}")
        cc[1].metric("Skill מול RW (RMSE)", f"{s['rmse_skill']:+.1%}")
        cc[2].metric("PIT KS (כיול)", f"{s['pit_ks']:.3f}")
        cc[3].metric("כיסוי 90%", f"{s['coverage_90']:.1%}")
        cc[4].metric("Student-t NLL", f"{s['nll']:.3f}")
        if "economic" in block:
            e = block["economic"]["strategy"]
            st.caption(
                f"אסטרטגיה ממומשת (עם עלויות): Sharpe {e.get('sharpe', float('nan')):.2f} · "
                f"MaxDD {e.get('max_drawdown', float('nan')):.1%} · "
                f"תנודתיות {e.get('ann_vol', float('nan')):.0%}"
            )

    st.subheader("כיול — היסטוגרמת PIT (h=1)")
    d1 = oos[oos.horizon == oos.horizon.min()].dropna(subset=["y_raw", "mu_raw", "sigma_raw", "nu"])
    from scipy.stats import t as _t

    pit = _t.cdf(
        (d1.y_raw.to_numpy() - d1.mu_raw.to_numpy()) / d1.sigma_raw.to_numpy(), df=d1.nu.to_numpy()
    )
    pit = pit[np.isfinite(pit)]
    counts = np.histogram(pit, bins=10, range=(0, 1))[0]
    st.altair_chart(C.reliability_bars(counts, height=280), width="stretch")
    st.caption("התפלגות אחידה = כיול מושלם. הקו הכתום = השכיחות הצפויה (10%).")


# ===========================================================================
# 6. RISK & LEVERAGE
# ===========================================================================
elif section == "סיכון ומינוף":
    hero(
        "סיכון ומינוף · ייעוץ בלבד",
        "גודל פוזיציה מומלץ לפי הביטחון וסיכון הקיצון — לא הוראת מסחר",
        "Risk",
    )
    pm = get_model_or_stop()
    dist = predict_distribution(pm, data)
    from diagnosis_layer import DiagnosisLayer
    from leverage_factorization import LeverageFactorization

    diag = DiagnosisLayer().diagnose(close.tail(250).values)
    row1 = dist[dist.horizon == dist.horizon.min()].iloc[0]

    c = st.columns(3)
    equity = c[0].number_input("הון בחשבון ($)", value=50_000, step=5_000)
    conf = c[1].slider(
        "ביטחון (confidence)", 0.0, 1.0, float(row1.p_up if row1.p_up > 0.5 else 1 - row1.p_up)
    )
    sev = c[2].slider("חומרת אירוע קיצון", 0.0, 1.0, 0.2)

    lf = LeverageFactorization()
    rec = lf.recommend(
        position_scale=1.0,
        confidence=conf,
        volatility_profile=diag.volatility,
        extreme_severity=sev,
        nav_premium_regime_code=0.0,
        position_notional_usd=float(equity),
    )
    cc = st.columns(4)
    cc[0].metric("מינוף מומלץ", f"{rec.recommended_leverage_ratio:.2f}x")
    cc[1].metric("תקרת מינוף", f"{rec.max_leverage_ceiling:.2f}x")
    cc[2].metric("מרחק ל-margin call", f"{rec.margin_call_distance_pct:.1%}")
    cc[3].metric("עלות מימון שנתית", f"{rec.financing_cost_annual_pct_of_equity:.2%}")
    _ceil = max(float(rec.max_leverage_ceiling), 0.1)
    lg = st.columns([1, 2])
    with lg[0]:
        st.altair_chart(
            C.gauge(
                float(rec.recommended_leverage_ratio),
                lo=0.0,
                hi=_ceil,
                title="מינוף מומלץ מול תקרה",
                label=f"{rec.recommended_leverage_ratio:.2f}x",
                zones=(
                    (0.0, _ceil * 0.5, C.GREEN),
                    (_ceil * 0.5, _ceil * 0.8, C.AMBER),
                    (_ceil * 0.8, _ceil, C.RED),
                ),
            ),
            width="stretch",
        )
    with lg[1]:
        st.altair_chart(
            C.bullet(
                float(rec.recommended_leverage_ratio),
                _ceil,
                lo=0.0,
                hi=_ceil * 1.1,
                title="מינוף (קו לבן = תקרה)",
                fmt=".2f",
                bands=[
                    (0.0, _ceil * 0.5, C.GREEN),
                    (_ceil * 0.5, _ceil * 0.8, C.AMBER),
                    (_ceil * 0.8, _ceil * 1.1, C.RED),
                ],
            ),
            width="stretch",
        )
    for n in rec.notes:
        st.markdown(f"- {n}")

    st.subheader("גודל פוזיציה — Kelly חסום")
    p_up = float(row1.p_up)
    edge = p_up - 0.5
    # payoff ratio b from the predictive distribution's own asymmetry (q75 gain vs q25 loss)
    win = abs(float(row1.ret_q75_pct))
    loss = max(abs(float(row1.ret_q25_pct)), 1e-6)
    b = win / loss
    kelly_full = p_up - (1.0 - p_up) / b  # f* = p - q/b
    kelly = float(np.clip(0.5 * kelly_full, -0.35, 0.35))  # half-Kelly, capped
    notional = equity * abs(kelly) * rec.recommended_leverage_ratio
    st.metric(
        "חשיפה מוצעת",
        f"${notional:,.0f}",
        f"{'לונג' if kelly > 0 else 'שורט / הימנעות'} · "
        f"{abs(kelly):.0%} מההון × {rec.recommended_leverage_ratio:.2f}x",
    )
    st.caption(
        f"חצי-Kelly חסום ל-±35%. יחס תמורה b = {b:.2f} (q75 {win:.1f}% מול q25 {loss:.1f}%), "
        f"P(עלייה) = {p_up:.0%}. **הכיוון (edge) לא מובהק סטטיסטית** — ראה מסך הראיות; "
        "התייחס לזה כאל תרגיל sizing, לא כאיתות."
    )


# ===========================================================================
# 7. SETTINGS
# ===========================================================================
elif section == "הגדרות":
    hero("הגדרות", "אימון מודל הייצור, רענון נתוני שוק ובדיקות תקינות", "Settings")

    st.subheader("נתונים")
    if st.button("רענן נתוני שוק (משוך מ-Yahoo)"):
        with st.spinner("מושך נתונים…"):
            refresh_market_data()
        st.success("הנתונים רועננו.")
        st.rerun()

    nav_file = st.file_uploader("קובץ NAV Premium (CSV)", type="csv")
    if nav_file is not None:
        from nav_premium import load_nav_csv

        tmp = "./data_cache/_nav_upload.csv"
        with open(tmp, "wb") as f:
            f.write(nav_file.getvalue())
        st.session_state.nav_df = load_nav_csv(tmp)
        st.success(f"נטענו {len(st.session_state.nav_df)} שורות NAV.")

    st.markdown("---")
    st.subheader("אימון מודל")
    c = st.columns(3)
    ens = c[0].slider("Ensemble (seeds)", 1, 7, 3)
    ep = c[1].slider("Epochs", 15, 90, 40)
    experts_on = c[2].checkbox("מודולי Gate (Fourier/Recurrent/Extreme)", value=True)

    st.caption("אימון על כל ההיסטוריה עד היום. ~1–4 דקות ל-CPU.")
    if st.button("אמן מודל עכשיו", type="primary"):
        icfg = InferenceConfig(
            ensemble=ens,
            seeds=tuple(range(ens)),
            epochs=ep,
            use_experts=experts_on,
            num_threads=4,
        )
        bar = st.progress(0.0, text="מתחיל…")

        def _p(i, n):
            bar.progress(min(i / n, 1.0), text=f"מאמן מודל {i}/{n}")

        train_data = load_data()
        try:
            pm = fit_production_model(train_data, icfg, progress=_p)
            path = save_production_model(pm, n_tab_features=train_data.X.shape[1])
        except Exception as exc:  # surface the real error instead of a raw traceback card
            st.error(f"האימון נכשל: {type(exc).__name__}: {exc}")
            st.stop()
        load_model_cached.clear()
        bar.progress(1.0, text="הושלם")
        st.success(f"המודל נשמר: {path}  ·  דאטה עד {pm.data_date_max}")
        st.rerun()

    if model_exists():
        st.info(f"מודל קיים מ-{datetime.fromtimestamp(model_mtime()):%Y-%m-%d %H:%M}")

    st.markdown("---")
    st.subheader("בדיקות תקינות")
    if st.button("הרץ test_integrity"):
        import subprocess
        import sys as _sys

        with st.spinner("מריץ בדיקות…"):
            out = subprocess.run(
                [_sys.executable, "test_integrity.py"],  # same interpreter as the app
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
        body = (out.stdout or "") + ("\n" + out.stderr if out.stderr else "")
        (st.success if out.returncode == 0 else st.error)(
            "כל הבדיקות עברו ✓" if out.returncode == 0 else "יש כשלים ✗"
        )
        st.code(body[-4000:])
