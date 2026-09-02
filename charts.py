"""
CARN-X  --  chart library  (visualisation only)
==============================================
A palette of themed Altair chart builders used across ``mstr_app.py`` so every
screen can show a *variety* of chart types (candlestick, donut, gauge, gradient
area, ranked bars, bullet, reliability, ridgeline, heatmap) instead of the three
default Streamlit charts.

**Nothing here computes model quantities.**  Every function takes data that the
app already produced and only decides how to draw it.  Cosmetic layer, no logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Dict, Optional

import altair as alt
import numpy as np
import pandas as pd

# palette (kept in sync with theme.py) ------------------------------------------
CYAN = "#38E8FF"
VIOLET = "#A855F7"
MAGENTA = "#F472B6"
LIME = "#A3E635"
AMBER = "#FBBF24"
RED = "#FB7185"
GREEN = "#34E5B0"
INK_GRID = "rgba(150,170,255,0.10)"
TEXT = "#C4CBEF"
CATEGORICAL = [CYAN, VIOLET, MAGENTA, LIME, AMBER, "#22D3EE", "#818CF8", RED]

_UP = GREEN
_DOWN = RED


def _axis(percent: bool = False, **kw):
    """Altair 6 rejects ``format=None`` -- build the Axis without it unless asked."""
    if percent:
        kw["format"] = "%"
    return alt.Axis(**kw) if kw else alt.Undefined


def _legend(percent: bool = False, **kw):
    if percent:
        kw["format"] = "%"
    return alt.Legend(**kw)


def _grad(c_top: str, c_bot: str, opa_top: float = 0.55, opa_bot: float = 0.02):
    return alt.Gradient(
        gradient="linear",
        x1=1,
        x2=1,
        y1=1,
        y2=0,
        stops=[alt.GradientStop(color=c_bot, offset=0), alt.GradientStop(color=c_top, offset=1)],
    )


# --------------------------------------------------------------------------- #
# 1. candlestick  (+ optional volume)
# --------------------------------------------------------------------------- #
def candlestick(
    df: pd.DataFrame,
    *,
    date: str = "date",
    o: str = "open",
    h: str = "high",
    l: str = "low",
    c: str = "close",
    volume: str | None = None,
    title: str = "",
    height: int = 360,
    y_title: str = "מחיר ($)",
    log_y: bool = False,
) -> alt.LayerChart:
    """Japanese candlestick chart from an OHLC(V) frame."""
    need = {date, o, h, l, c}
    d = (
        df.copy()
        if need <= set(df.columns)
        else pd.DataFrame({date: pd.to_datetime([]), o: [], h: [], l: [], c: []})
    )
    d = d.dropna(subset=[o, h, l, c])
    d[date] = pd.to_datetime(d[date])
    d["_dir"] = np.where(d[c].to_numpy() >= d[o].to_numpy(), "up", "down")
    color = alt.Color(
        "_dir:N", scale=alt.Scale(domain=["up", "down"], range=[_UP, _DOWN]), legend=None
    )
    yscale = alt.Scale(type="log" if log_y else "linear", zero=False, nice=False)
    x = alt.X(f"{date}:T", title=None, axis=alt.Axis(labelColor=TEXT, format="%b %y"))

    base = alt.Chart(d)
    wick = base.mark_rule().encode(
        x=x, y=alt.Y(f"{l}:Q", scale=yscale, title=y_title), y2=f"{h}:Q", color=color
    )
    body = base.mark_bar(size=5).encode(
        x=x,
        y=alt.Y(f"{o}:Q", scale=yscale),
        y2=f"{c}:Q",
        color=color,
        tooltip=[
            alt.Tooltip(f"{date}:T", title="תאריך"),
            alt.Tooltip(f"{o}:Q", title="פתיחה", format=",.2f"),
            alt.Tooltip(f"{h}:Q", title="גבוה", format=",.2f"),
            alt.Tooltip(f"{l}:Q", title="נמוך", format=",.2f"),
            alt.Tooltip(f"{c}:Q", title="סגירה", format=",.2f"),
        ],
    )
    price = (wick + body).properties(height=height, title=title)

    if volume and volume in d.columns:
        vol = (
            alt.Chart(d)
            .mark_bar(opacity=0.5)
            .encode(
                x=x, y=alt.Y(f"{volume}:Q", title="מחזור", axis=alt.Axis(labels=False)), color=color
            )
            .properties(height=max(70, height // 5))
        )
        return alt.vconcat(price, vol, spacing=4).resolve_scale(x="shared")
    return price


# --------------------------------------------------------------------------- #
# 2. donut / pie
# --------------------------------------------------------------------------- #
def donut(
    parts: Mapping[str, float],
    *,
    title: str = "",
    height: int = 260,
    center: str = "",
    inner: int = 62,
    colors: Sequence[str] | None = None,
    as_pie: bool = False,
) -> alt.LayerChart:
    d = pd.DataFrame(
        {
            "label": [str(k) for k in parts],
            "value": [float(v) if np.isfinite(v) else 0.0 for v in parts.values()],
        }
    )
    d = d[d["value"] > 0]
    if d.empty:  # nothing to slice -> neutral ring
        d = pd.DataFrame({"label": ["—"], "value": [1.0]})
    rng = list(colors)[: len(d)] if colors else CATEGORICAL[: len(d)]
    if len(rng) < len(d):
        rng = (rng * (len(d) // max(len(rng), 1) + 1))[: len(d)]
    arc = (
        alt.Chart(d)
        .mark_arc(
            innerRadius=0 if as_pie else inner,
            outerRadius=105,
            cornerRadius=3,
            stroke="#0B1020",
            strokeWidth=2,
        )
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(range=rng),
                legend=alt.Legend(orient="bottom", title=None, labelColor=TEXT),
            ),
            order=alt.Order("value:Q", sort="descending"),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("value:Q", format=",.1f")],
        )
        .properties(height=height, title=title)
    )
    if center and not as_pie:
        txt = (
            alt.Chart(pd.DataFrame({"t": [center]}))
            .mark_text(fontSize=20, fontWeight="bold", color="#EAF2FF")
            .encode(text="t:N")
        )
        return arc + txt
    return arc + alt.Chart(pd.DataFrame({"t": [""]})).mark_text().encode(text="t:N")


# --------------------------------------------------------------------------- #
# 3. gauge  (semicircular)
# --------------------------------------------------------------------------- #
def gauge(
    value: float,
    *,
    lo: float = 0.0,
    hi: float = 1.0,
    title: str = "",
    label: str | None = None,
    height: int = 200,
    zones: Sequence[tuple] = ((0.0, 0.5, RED), (0.5, 0.8, AMBER), (0.8, 1.0, GREEN)),
) -> alt.LayerChart:
    """A 270-degree gauge.  ``value`` is clamped to [lo, hi]; degenerate or
    non-finite inputs render an empty dial rather than raising."""
    span = 4.712389  # 270 deg in radians
    start = -2.356194  # -135 deg
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        hi, lo = (lo + 1.0), lo  # keep drawing, just neutral
    v = float(value) if np.isfinite(value) else lo
    frac = (float(np.clip(v, lo, hi)) - lo) / (hi - lo)

    def _t(x):
        return start + span * (float(np.clip(x, lo, hi)) - lo) / (hi - lo)

    zdf = pd.DataFrame([{"t1": _t(z0), "t2": _t(z1), "c": col} for z0, z1, col in zones if z1 > z0])
    if zdf.empty:
        zdf = pd.DataFrame([{"t1": start, "t2": start + span, "c": "#334155"}])
    track = (
        alt.Chart(zdf)
        .mark_arc(innerRadius=58, outerRadius=88)
        .encode(
            theta=alt.Theta("t1:Q", scale=None),
            theta2="t2:Q",
            color=alt.Color("c:N", scale=None, legend=None),
            opacity=alt.value(0.35),
        )
    )
    val = (
        alt.Chart(pd.DataFrame({"t1": [start], "t2": [start + span * frac]}))
        .mark_arc(innerRadius=58, outerRadius=88, cornerRadius=6)
        .encode(theta=alt.Theta("t1:Q", scale=None), theta2="t2:Q", color=alt.value(CYAN))
    )
    disp = label if label is not None else (f"{v:.0%}" if hi <= 1.0 else f"{v:,.2f}")
    txt = (
        alt.Chart(pd.DataFrame({"t": [disp]}))
        .mark_text(fontSize=26, fontWeight="bold", color="#EAF2FF", dy=6)
        .encode(text="t:N")
    )
    return (track + val + txt).properties(height=height, title=title)


# --------------------------------------------------------------------------- #
# 4. gradient area  (levels or a signed series)
# --------------------------------------------------------------------------- #
def area_gradient(
    s: pd.Series,
    *,
    title: str = "",
    height: int = 240,
    color: str = CYAN,
    x_title: str | None = None,
    y_title: str = "",
    signed: bool = False,
    percent: bool = False,
) -> alt.Chart:
    dt = _is_dt(s.index)
    d = pd.DataFrame(
        {
            "x": pd.to_datetime(s.index) if dt else np.arange(len(s)),
            "y": pd.to_numeric(pd.Series(np.asarray(s)), errors="coerce"),
        }
    )
    xtype = "T" if dt else "Q"
    enc_y = alt.Y("y:Q", title=y_title, axis=_axis(percent))
    line_c = color if not signed else MAGENTA
    return (
        alt.Chart(d)
        .mark_area(
            line={"color": line_c, "strokeWidth": 2},
            color=_grad(color, color),
            interpolate="monotone",
        )
        .encode(
            x=alt.X(f"x:{xtype}", title=x_title),
            y=enc_y,
            tooltip=[alt.Tooltip(f"x:{xtype}", title=""), alt.Tooltip("y:Q", format=".3f")],
        )
        .properties(height=height, title=title)
    )


# --------------------------------------------------------------------------- #
# 5. diverging bars  (up / down coloured)
# --------------------------------------------------------------------------- #
def diverging_bars(
    s: pd.Series,
    *,
    title: str = "",
    height: int = 220,
    percent: bool = True,
    x_title: str | None = None,
) -> alt.Chart:
    d = pd.DataFrame(
        {"x": pd.to_datetime(s.index) if _is_dt(s.index) else np.arange(len(s)), "y": s.values}
    )
    d["dir"] = np.where(d["y"] >= 0, "up", "down")
    xtype = "T" if _is_dt(s.index) else "Q"
    return (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(f"x:{xtype}", title=x_title),
            y=alt.Y("y:Q", title=None, axis=_axis(percent)),
            color=alt.Color(
                "dir:N", scale=alt.Scale(domain=["up", "down"], range=[_UP, _DOWN]), legend=None
            ),
            tooltip=[alt.Tooltip("y:Q", format=".2%" if percent else ".3f")],
        )
        .properties(height=height, title=title)
    )


# --------------------------------------------------------------------------- #
# 6. ranked horizontal bars
# --------------------------------------------------------------------------- #
def hbar_ranked(
    df: pd.DataFrame,
    *,
    cat: str,
    val: str,
    title: str = "",
    height: int = 300,
    percent: bool = False,
    color: str = VIOLET,
    descending: bool = True,
) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=5, color=color)
        .encode(
            x=alt.X(f"{val}:Q", title=None, axis=_axis(percent)),
            y=alt.Y(f"{cat}:N", sort="-x" if descending else "x", title=None),
            color=alt.Color(f"{val}:Q", scale=alt.Scale(scheme="viridis"), legend=None),
            tooltip=[
                alt.Tooltip(f"{cat}:N", title=""),
                alt.Tooltip(f"{val}:Q", format=".1%" if percent else ".3f"),
            ],
        )
        .properties(height=height, title=title)
    )


# --------------------------------------------------------------------------- #
# 7. bullet  (value vs target with qualitative bands)
# --------------------------------------------------------------------------- #
def bullet(
    value: float,
    target: float,
    *,
    lo: float,
    hi: float,
    title: str = "",
    bands: Sequence[tuple] = (),
    height: int = 90,
    fmt: str = ".2f",
) -> alt.LayerChart:
    # widen the domain so the measure bar and target tick are always visible,
    # and repair a degenerate [lo, hi]
    vals = [v for v in (lo, hi, value, target) if np.isfinite(v)] or [0.0, 1.0]
    dlo, dhi = min(vals), max(vals)
    if dhi <= dlo:
        dhi = dlo + 1.0
    pad = (dhi - dlo) * 0.04
    sc = alt.Scale(domain=[dlo - pad, dhi + pad])
    bnd = (
        pd.DataFrame(
            [
                {"s": b0, "e": b1, "c": col}
                for b0, b1, col in bands
                if np.isfinite(b0) and np.isfinite(b1) and b1 > b0
            ]
        )
        if bands
        else pd.DataFrame({"s": [dlo], "e": [dhi], "c": ["#223"]})
    )
    if bnd.empty:
        bnd = pd.DataFrame({"s": [dlo], "e": [dhi], "c": ["#223"]})
    band = (
        alt.Chart(bnd)
        .mark_bar(height=26)
        .encode(
            x=alt.X("s:Q", scale=sc, title=None),
            x2="e:Q",
            color=alt.Color("c:N", scale=None, legend=None),
            opacity=alt.value(0.30),
        )
    )
    meas = (
        alt.Chart(pd.DataFrame({"v": [value if np.isfinite(value) else dlo]}))
        .mark_bar(height=10, color=CYAN)
        .encode(x=alt.X("v:Q", scale=sc))
    )
    tgt = (
        alt.Chart(pd.DataFrame({"v": [target if np.isfinite(target) else dlo]}))
        .mark_tick(thickness=3, size=34, color="#EAF2FF")
        .encode(x=alt.X("v:Q", scale=sc))
    )
    return (band + meas + tgt).properties(height=height, title=title)


# --------------------------------------------------------------------------- #
# 8. reliability / PIT calibration
# --------------------------------------------------------------------------- #
def reliability_bars(
    deciles: Sequence[float],
    *,
    title: str = "כיול — היסטוגרמת PIT",
    height: int = 260,
) -> alt.LayerChart:
    arr = np.asarray(list(deciles), float)
    if arr.size == 0:
        arr = np.zeros(10)
    n = arr.size
    total = float(np.nansum(arr)) or 1.0
    d = pd.DataFrame(
        {"bin": [f"{i / n:.1f}" for i in range(n)], "freq": np.nan_to_num(arr) / total}
    )
    bars = (
        alt.Chart(d)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("bin:N", title="PIT"),
            y=alt.Y("freq:Q", title="שכיחות", axis=alt.Axis(format="%")),
            color=alt.Color("freq:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
        )
    )
    ref = (
        alt.Chart(pd.DataFrame({"y": [1.0 / n]}))
        .mark_rule(color=AMBER, strokeDash=[4, 4], strokeWidth=2)
        .encode(y="y:Q")
    )
    return (bars + ref).properties(height=height, title=title)


# --------------------------------------------------------------------------- #
# 9. ridgeline  (several densities stacked)
# --------------------------------------------------------------------------- #
def ridgeline(
    densities: Mapping[str, tuple],
    *,
    title: str = "",
    height: int = 90,
    x_title: str = "",
    percent: bool = False,
) -> alt.FacetChart:
    frames = []
    for name, (xs, ys) in densities.items():
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        if xs.size and xs.size == ys.size:
            frames.append(pd.DataFrame({"grp": str(name), "x": xs, "y": ys}))
    if not frames:
        frames = [pd.DataFrame({"grp": ["—"], "x": [0.0], "y": [0.0]})]
    d = pd.concat(frames, ignore_index=True)
    return (
        alt.Chart(d)
        .mark_area(
            interpolate="monotone",
            fillOpacity=0.55,
            line={"strokeWidth": 1.5},
        )
        .encode(
            x=alt.X("x:Q", title=x_title, axis=_axis(percent)),
            y=alt.Y("y:Q", title=None, axis=None, scale=alt.Scale(range=[height, -6])),
            color=alt.Color("grp:N", scale=alt.Scale(range=CATEGORICAL), legend=None),
            row=alt.Row(
                "grp:N",
                title=None,
                header=alt.Header(labelColor=TEXT, labelAngle=0, labelAlign="right"),
            ),
        )
        .properties(height=height, title=title, bounds="flush")
    )


# --------------------------------------------------------------------------- #
# 10. heatmap
# --------------------------------------------------------------------------- #
def heatmap(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    val: str,
    title: str = "",
    height: int = 260,
    scheme: str = "magma",
    percent: bool = False,
) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(f"{x}:O", title=None),
            y=alt.Y(f"{y}:O", title=None),
            color=alt.Color(
                f"{val}:Q", scale=alt.Scale(scheme=scheme), legend=_legend(percent, title=None)
            ),
            tooltip=[x, y, alt.Tooltip(f"{val}:Q", format=".1%" if percent else ".3f")],
        )
        .properties(height=height, title=title)
    )


# --------------------------------------------------------------------------- #
# 11. probability fan  (percentile bands over a horizon)
# --------------------------------------------------------------------------- #
def fan(
    df: pd.DataFrame,
    *,
    x: str,
    bands: Sequence[tuple],
    median: str,
    title: str = "",
    height: int = 380,
    y_title: str = "מחיר ($)",
    log_y: bool = False,
    rules: Mapping[str, float] | None = None,
) -> alt.LayerChart:
    yscale = alt.Scale(type="log" if log_y else "linear", zero=False, nice=False)
    base = alt.Chart(df).encode(x=alt.X(f"{x}:Q", title=None))
    layers = []
    for lo, hi, opa in bands:
        layers.append(
            base.mark_area(opacity=opa, color=CYAN).encode(
                y=alt.Y(f"{lo}:Q", scale=yscale, title=y_title), y2=f"{hi}:Q"
            )
        )
    layers.append(
        base.mark_line(color="#EAF2FF", strokeWidth=2.5).encode(
            y=alt.Y(f"{median}:Q", scale=yscale)
        )
    )
    if rules:
        rdf = pd.DataFrame({"y": list(rules.values()), "label": list(rules.keys())})
        layers.append(
            alt.Chart(rdf)
            .mark_rule(strokeDash=[5, 4], strokeWidth=1.5)
            .encode(
                y=alt.Y("y:Q", scale=yscale),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(range=[AMBER, MAGENTA, LIME]),
                    legend=alt.Legend(orient="top", title=None, labelColor=TEXT),
                ),
            )
        )
    return alt.layer(*layers).resolve_scale(y="shared").properties(height=height, title=title)


# --------------------------------------------------------------------------- #
# 12. KPI sparkline strip  (small multiples)
# --------------------------------------------------------------------------- #
def spark(s: pd.Series, *, color: str = CYAN, height: int = 60) -> alt.Chart:
    d = pd.DataFrame({"x": np.arange(len(s)), "y": np.asarray(s, float)})
    return (
        alt.Chart(d)
        .mark_area(
            line={"color": color, "strokeWidth": 1.5},
            color=_grad(color, color),
            interpolate="monotone",
        )
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False, nice=False)),
        )
        .properties(height=height)
    )


# --------------------------------------------------------------------------- #
def _is_dt(idx) -> bool:
    try:
        return np.issubdtype(np.asarray(idx).dtype, np.datetime64) or isinstance(
            idx, pd.DatetimeIndex
        )
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 13. mathematical-structures views  (visual only -- math_structures.py owns the maths)
# --------------------------------------------------------------------------- #
def complex_plane(
    df: pd.DataFrame,
    *,
    re: str = "re",
    im: str = "im",
    size: str = "amp",
    title: str = "",
    height: int = 320,
) -> alt.LayerChart:
    """Eigenvalues on the complex plane with the unit circle drawn in."""
    th = np.linspace(0, 2 * np.pi, 240)
    circ = pd.DataFrame({"x": np.cos(th), "y": np.sin(th)})
    ring = (
        alt.Chart(circ)
        .mark_line(color=INK_GRID, strokeWidth=1)
        .encode(x=alt.X("x:Q", title="Re λ"), y=alt.Y("y:Q", title="Im λ"))
    )
    axes = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK_GRID).encode(
        x="z:Q"
    ) + alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK_GRID).encode(y="z:Q")
    pts = (
        alt.Chart(df)
        .mark_circle(opacity=0.85)
        .encode(
            x=f"{re}:Q",
            y=f"{im}:Q",
            size=alt.Size(f"{size}:Q", legend=None, scale=alt.Scale(range=[30, 500]))
            if size in df
            else alt.value(90),
            color=alt.Color(f"{re}:Q", scale=alt.Scale(scheme="turbo"), legend=None),
            tooltip=[alt.Tooltip(f"{re}:Q", format=".3f"), alt.Tooltip(f"{im}:Q", format=".3f")]
            + ([alt.Tooltip(f"{size}:Q", format=".3f")] if size in df else []),
        )
    )
    return alt.layer(ring, axes, pts).properties(height=height, title=title)


def surface_heat(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    val: str,
    title: str = "",
    height: int = 300,
    scheme: str = "inferno",
    x_title: str = "",
    y_title: str = "",
) -> alt.Chart:
    """Quantitative-axis heat map -- density surfaces, path-probability fields."""
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=120), title=x_title or None),
            y=alt.Y(f"{y}:Q", bin=alt.Bin(maxbins=60), title=y_title or None),
            color=alt.Color(f"{val}:Q", scale=alt.Scale(scheme=scheme), legend=_legend(title=None)),
            tooltip=[
                alt.Tooltip(f"{x}:Q", format=".3f"),
                alt.Tooltip(f"{y}:Q", format=".3f"),
                alt.Tooltip(f"{val}:Q", format=".4f"),
            ],
        )
        .properties(height=height, title=title)
    )


def stem(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str = "",
    height: int = 240,
    color: str = CYAN,
    x_title: str = "",
    y_title: str = "",
    percent: bool = False,
) -> alt.LayerChart:
    """Lollipop / stem plot -- Pascal-row weights, terminal distributions, mode amplitudes."""
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:Q", title=x_title or None),
        y=alt.Y(f"{y}:Q", title=y_title or None, axis=_axis(percent)),
    )
    return alt.layer(
        base.mark_rule(color=color, strokeWidth=1.5, opacity=0.5),
        base.mark_circle(color=color, size=55),
    ).properties(height=height, title=title)


def line_with_levels(
    s: pd.Series,
    levels: Mapping[str, float],
    *,
    title: str = "",
    height: int = 320,
    y_title: str = "מחיר ($)",
) -> alt.LayerChart:
    """A price line with labelled horizontal reference levels (e.g. Fibonacci)."""
    d = pd.DataFrame({"date": pd.to_datetime(s.index), "price": np.asarray(s, float)})
    line = (
        alt.Chart(d)
        .mark_line(color="#EAF2FF", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("price:Q", scale=alt.Scale(zero=False, nice=False), title=y_title),
        )
    )
    ldf = pd.DataFrame({"y": list(levels.values()), "label": [str(k) for k in levels]})
    rules = (
        alt.Chart(ldf)
        .mark_rule(strokeDash=[6, 4], strokeWidth=1.4)
        .encode(
            y="y:Q",
            color=alt.Color(
                "label:N",
                scale=alt.Scale(scheme="plasma"),
                legend=alt.Legend(orient="right", title="רמה"),
            ),
        )
    )
    text = (
        alt.Chart(ldf)
        .mark_text(align="left", dx=4, color=TEXT, fontSize=10)
        .encode(y="y:Q", text=alt.Text("y:Q", format="$,.0f"), x=alt.value(4))
    )
    return alt.layer(line, rules, text).properties(height=height, title=title)
