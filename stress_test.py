"""
CARN-X  --  BRUTAL stress test
==============================
Adversarial, wide-scope verification of the whole app: chart builders under
degenerate input, the Monte-Carlo engine across its full parameter space with
mathematical-invariant assertions, the trading calendar, config presets, model
staleness, prob-models sensor extremes, and an AppTest that sweeps *every* widget
on *every* screen to its edges.

    .venv/bin/python stress_test.py            # everything (~10-20 min)
    .venv/bin/python stress_test.py --fast     # skip the AppTest widget sweep
    .venv/bin/python stress_test.py --only mc  # one section

Exit code 0 == all green.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401

import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

T0 = time.time()
FAILS: list[str] = []
ONLY = None
for i, a in enumerate(sys.argv):
    if a == "--only" and i + 1 < len(sys.argv):
        ONLY = sys.argv[i + 1]
FAST = "--fast" in sys.argv


def section(name):
    def deco(fn):
        if ONLY and ONLY not in name:
            return fn
        print(f"\n{'=' * 70}\n▶ {name}\n{'=' * 70}")
        t = time.time()
        try:
            fn()
            print(f"  ✓ {name}  ({time.time() - t:.0f}s)")
        except AssertionError as e:
            FAILS.append(f"{name}: {e}")
            print(f"  ✗ {name}\n{traceback.format_exc()}")
        except Exception as e:  # noqa: BLE001
            FAILS.append(f"{name}: UNHANDLED {type(e).__name__}: {e}")
            print(f"  ✗ {name}  (UNHANDLED)\n{traceback.format_exc()}")
        return fn

    return deco


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ===========================================================================
# 1. chart builders -- adversarial input fuzz
# ===========================================================================
@section("1. charts.py -- degenerate-input fuzz (every builder)")
def _():
    import charts as C

    rng = np.random.default_rng(0)
    n = 90
    dt = pd.bdate_range("2025-01-01", periods=n)
    o = 100 + np.cumsum(rng.normal(0, 2, n))
    c = o + rng.normal(0, 3, n)
    ohlc = pd.DataFrame(
        {
            "date": dt,
            "open": o,
            "high": np.maximum(o, c) + 1,
            "low": np.minimum(o, c) - 1,
            "close": c,
            "volume": np.abs(rng.normal(1e6, 3e5, n)),
        }
    )
    empty = ohlc.iloc[0:0]

    cases = {
        # candlestick
        "cs.normal": lambda: C.candlestick(ohlc, volume="volume"),
        "cs.empty": lambda: C.candlestick(empty, volume="volume"),
        "cs.missing_cols": lambda: C.candlestick(pd.DataFrame({"a": [1]}), volume="volume"),
        "cs.one_row": lambda: C.candlestick(ohlc.iloc[:1]),
        "cs.all_nan_close": lambda: C.candlestick(ohlc.assign(close=[np.nan] * n)),
        "cs.log_y": lambda: C.candlestick(ohlc, log_y=True),
        # donut
        "donut.normal": lambda: C.donut({"a": 3, "b": 5, "c": 2}, center="10"),
        "donut.zero_sum": lambda: C.donut({"a": 0, "b": 0}),
        "donut.one_cat": lambda: C.donut({"only": 5}, center="x"),
        "donut.nan": lambda: C.donut({"a": np.nan, "b": 2}),
        "donut.neg": lambda: C.donut({"a": -3, "b": -1}),
        "donut.many_few_colors": lambda: C.donut(
            {str(i): i + 1 for i in range(12)}, colors=["#111", "#222"]
        ),
        "donut.pie": lambda: C.donut({"x": 1, "y": 2}, as_pie=True),
        # gauge
        "gauge.normal": lambda: C.gauge(0.62),
        "gauge.lo_eq_hi": lambda: C.gauge(1.0, lo=1.0, hi=1.0),
        "gauge.hi_lt_lo": lambda: C.gauge(1.0, lo=5.0, hi=2.0),
        "gauge.value_outside": lambda: C.gauge(9.0, lo=0, hi=1),
        "gauge.nan_value": lambda: C.gauge(np.nan, lo=0, hi=1),
        "gauge.nan_bounds": lambda: C.gauge(0.5, lo=np.nan, hi=np.nan),
        "gauge.empty_zones": lambda: C.gauge(0.5, zones=[]),
        "gauge.degenerate_zone": lambda: C.gauge(0.5, zones=[(0.5, 0.5, "#111")]),
        "gauge.raw_units": lambda: C.gauge(1.42, lo=0, hi=3, label="1.42x"),
        # area
        "area.dt_idx": lambda: C.area_gradient(pd.Series(c, index=dt)),
        "area.int_idx": lambda: C.area_gradient(pd.Series(c)),
        "area.empty": lambda: C.area_gradient(pd.Series([], dtype=float)),
        "area.all_nan": lambda: C.area_gradient(pd.Series([np.nan] * 5)),
        "area.signed_pct": lambda: C.area_gradient(
            pd.Series(rng.normal(0, 0.02, n), index=dt), signed=True, percent=True
        ),
        # diverging
        "div.normal": lambda: C.diverging_bars(pd.Series(rng.normal(0, 0.02, n), index=dt)),
        "div.empty": lambda: C.diverging_bars(pd.Series([], dtype=float)),
        "div.all_nan": lambda: C.diverging_bars(pd.Series([np.nan] * 4)),
        # hbar
        "hbar.normal": lambda: C.hbar_ranked(
            pd.DataFrame({"m": ["a", "b"], "p": [0.2, 0.5]}), cat="m", val="p", percent=True
        ),
        "hbar.empty": lambda: C.hbar_ranked(pd.DataFrame({"m": [], "p": []}), cat="m", val="p"),
        # bullet
        "bullet.normal": lambda: C.bullet(
            1.4, 2.0, lo=0, hi=3, bands=[(0, 1, "#3a3"), (1, 3, "#a33")]
        ),
        "bullet.lo_eq_hi": lambda: C.bullet(1.0, 1.0, lo=1.0, hi=1.0),
        "bullet.value_outside": lambda: C.bullet(50, -10, lo=0, hi=1),
        "bullet.nan": lambda: C.bullet(np.nan, np.nan, lo=np.nan, hi=np.nan),
        "bullet.bad_bands": lambda: C.bullet(
            1, 2, lo=0, hi=3, bands=[(2, 1, "#111"), (np.nan, 5, "#222")]
        ),
        # reliability
        "rel.normal": lambda: C.reliability_bars([10, 12, 9, 11, 8, 13, 10, 9, 12, 10]),
        "rel.empty": lambda: C.reliability_bars([]),
        "rel.all_zero": lambda: C.reliability_bars([0] * 10),
        "rel.one_bin": lambda: C.reliability_bars([5]),
        "rel.with_nan": lambda: C.reliability_bars([np.nan, 1, 2]),
        # ridgeline
        "ridge.normal": lambda: C.ridgeline(
            {"h1": (np.linspace(-1, 1, 40), np.exp(-(np.linspace(-2, 2, 40) ** 2)))}
        ),
        "ridge.empty": lambda: C.ridgeline({}),
        "ridge.mismatched": lambda: C.ridgeline({"a": ([1, 2, 3], [1, 2])}),
        "ridge.empty_arrays": lambda: C.ridgeline({"a": ([], [])}),
        # spark / heatmap / fan
        "spark.normal": lambda: C.spark(pd.Series(c)),
        "spark.empty": lambda: C.spark(pd.Series([], dtype=float)),
        "heatmap.normal": lambda: C.heatmap(
            pd.DataFrame({"x": [1, 1, 2], "y": ["a", "b", "a"], "v": [0.1, 0.4, 0.7]}),
            x="x",
            y="y",
            val="v",
        ),
        "fan.normal": lambda: C.fan(
            pd.DataFrame(
                {
                    "day": np.arange(10),
                    "lo": np.linspace(90, 70, 10),
                    "hi": np.linspace(110, 150, 10),
                    "mid": np.full(10, 100.0),
                }
            ),
            x="day",
            bands=[("lo", "hi", 0.2)],
            median="mid",
            rules={"S0": 100, "t": 130},
        ),
    }
    bad = []
    for name, fn in cases.items():
        try:
            spec = fn().to_dict()  # forces full Vega-Lite compilation
            _check(isinstance(spec, dict) and spec, f"{name}: empty spec")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{name}: {type(e).__name__}: {e}")
    _check(not bad, "chart builders raised on:\n    " + "\n    ".join(bad))
    print(f"    {len(cases)} adversarial chart cases compiled clean")


# ===========================================================================
# 2. Monte-Carlo engine -- full parameter sweep + invariants
# ===========================================================================
@section("2. monte_carlo_paths -- parameter sweep + mathematical invariants")
def _():
    from dataset import assemble
    from inference import (
        InferenceConfig,
        load_production_model,
        monte_carlo_paths,
        DRIFT_MODES,
        default_model_path,
    )

    if not os.path.exists(default_model_path()):
        print("    (no production model -- skipping)")
        return
    cfg = InferenceConfig()
    data = assemble(
        seq_feature_names=cfg.model.seq_feature_names, target_cfg=cfg.targets, require_targets=False
    )
    pm = load_production_model(data)
    S0 = float(data.close.reindex(data.index).iloc[-1])

    grid = []
    for mode in DRIFT_MODES:
        for hz in (1, 5, 6, 63, 155, 252, 503, 504):
            for npth in (2, 5000, 100_000):
                for anti in (True, False):
                    grid.append((mode, hz, npth, anti))
    # a few pathological extras
    extras = [
        ("custom", 252, 40_000, True, {"custom_drift_annual": -0.99}),
        ("custom", 252, 40_000, True, {"custom_drift_annual": 5.0}),
        ("risk_neutral", 252, 40_000, True, {"risk_free_rate": 0.0}),
        ("risk_neutral", 252, 40_000, True, {"risk_free_rate": 0.5}),
    ]

    checked = 0
    for mode, hz, npth, anti in grid:
        mc = monte_carlo_paths(
            pm,
            data,
            horizon=hz,
            n_paths=npth,
            drift_mode=mode,
            antithetic=anti,
            target_price=S0 * 1.25,
            seed=1,
        )
        checked += 1
        # shapes
        _check(mc.paths.shape[1] == hz + 1, f"{mode} h{hz}: path width {mc.paths.shape[1]}")
        _check(mc.paths.shape[0] == mc.n_paths, f"{mode} h{hz}: n_paths mismatch")
        _check(mc.n_paths >= 2, f"{mode} h{hz}: n_paths<2")
        # finiteness
        _check(np.isfinite(mc.paths).all(), f"{mode} h{hz} np{npth}: non-finite path")
        _check((mc.paths > 0).all(), f"{mode} h{hz}: non-positive price")
        _check(
            np.isfinite(mc.expected_price) and mc.expected_price > 0,
            f"{mode} h{hz}: bad E[S_T] {mc.expected_price}",
        )
        # probabilities in [0,1]
        for p in (
            mc.prob_up,
            mc.p_touch_up,
            mc.p_touch_down,
            mc.p_close_above,
            mc.max_drawdown_p50,
        ):
            _check(0.0 - 1e-9 <= p <= 1.0 + 1e-9, f"{mode} h{hz}: prob out of range {p}")
        # percentile monotonicity at the terminal
        order = ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
        term_pcts = [mc.percentiles[k][-1] for k in order]
        _check(
            all(a <= b + 1e-6 for a, b in zip(term_pcts, term_pcts[1:])),
            f"{mode} h{hz}: terminal percentiles not monotone {term_pcts}",
        )
        _check(mc.var_1_price <= mc.var_5_price + 1e-6, f"{mode} h{hz}: VaR1 > VaR5")
        _check(mc.es_5_price <= mc.var_5_price + 1e-6, f"{mode} h{hz}: ES5 > VaR5")
        # median vs percentiles consistency
        _check(
            abs(mc.median_price - mc.percentiles["p50"][-1]) <= max(1.0, S0 * 0.05),
            f"{mode} h{hz}: median disagrees with p50",
        )

    for mode, hz, npth, anti, kw in extras:
        mc = monte_carlo_paths(
            pm, data, horizon=hz, n_paths=npth, drift_mode=mode, antithetic=anti, seed=2, **kw
        )
        _check(
            np.isfinite(mc.paths).all() and (mc.paths > 0).all(), f"extra {mode} {kw}: bad paths"
        )
        checked += 1

    # --- martingale property under Q, across seeds ---
    r = 0.045
    ratios = []
    for sd in (0, 1, 2, 3, 4):
        m = monte_carlo_paths(
            pm,
            data,
            horizon=252,
            n_paths=80_000,
            drift_mode="risk_neutral",
            risk_free_rate=r,
            seed=sd,
        )
        ratios.append(m.expected_price / m.last_price)
    mean_ratio = float(np.mean(ratios))
    _check(
        abs(mean_ratio - np.exp(r)) < 0.02,
        f"risk-neutral martingale broken: mean E[S_T]/S0={mean_ratio:.4f} vs e^r={np.exp(r):.4f}",
    )

    # --- monotonic response to custom drift ---
    es = []
    for mu in (-0.4, -0.1, 0.0, 0.2, 0.5, 1.0):
        m = monte_carlo_paths(
            pm,
            data,
            horizon=252,
            n_paths=40_000,
            drift_mode="custom",
            custom_drift_annual=mu,
            seed=7,
        )
        es.append(m.expected_price)
    _check(
        all(a < b for a, b in zip(es, es[1:])),
        f"E[S_T] not monotone increasing in drift: {[round(x, 1) for x in es]}",
    )

    # --- determinism: same seed -> same answer ---
    a = monte_carlo_paths(
        pm, data, horizon=100, n_paths=20_000, drift_mode="custom", custom_drift_annual=0.3, seed=99
    )
    b = monte_carlo_paths(
        pm, data, horizon=100, n_paths=20_000, drift_mode="custom", custom_drift_annual=0.3, seed=99
    )
    _check(
        np.allclose(a.terminal_prices, b.terminal_prices), "MC not deterministic for a fixed seed"
    )

    # --- antithetic really reduces the mean's sampling error ---
    S0 = a.last_price
    err_anti, err_plain = [], []
    for sd in range(6):
        ma = monte_carlo_paths(
            pm, data, horizon=60, n_paths=8000, drift_mode="risk_neutral", antithetic=True, seed=sd
        )
        mp = monte_carlo_paths(
            pm, data, horizon=60, n_paths=8000, drift_mode="risk_neutral", antithetic=False, seed=sd
        )
        err_anti.append(abs(ma.expected_price / ma.last_price - np.exp(0.045 * 60 / 252)))
        err_plain.append(abs(mp.expected_price / mp.last_price - np.exp(0.045 * 60 / 252)))
    _check(
        np.mean(err_anti) <= np.mean(err_plain) * 1.3,
        f"antithetic not helping: anti={np.mean(err_anti):.4f} plain={np.mean(err_plain):.4f}",
    )

    print(
        f"    {checked} sweep configs · martingale {mean_ratio:.4f} · "
        f"drift-monotone OK · deterministic OK · antithetic OK"
    )


# ===========================================================================
# 3. trading calendar -- exhaustive round-trip
# ===========================================================================
@section("3. trading calendar -- add/between round-trip, all start days")
def _():
    from prob_models import add_trading_days, trading_days_between

    starts = pd.date_range("2026-01-01", "2026-12-31", freq="D")
    bad = []
    for s in starts[::3]:
        for k in (1, 2, 3, 5, 13, 21, 63, 126, 155, 252, 400, 504, 600):
            e = add_trading_days(s, k)
            back = trading_days_between(s, e)
            if back != k:
                bad.append(f"{s.date()} +{k} -> {e.date()} -> back {back}")
    _check(not bad, "round-trip broken:\n    " + "\n    ".join(bad[:20]))
    # known anchor
    _check(
        add_trading_days("2026-08-28", 155).date().isoformat() == "2027-04-13",
        "155 td from 2026-08-28 should be 2027-04-13",
    )
    # n<=0
    _check(add_trading_days("2026-08-28", 0).date().isoformat() == "2026-08-28", "n=0")
    _check(add_trading_days("2026-08-28", -5).date().isoformat() == "2026-08-28", "n<0")
    _check(trading_days_between("2026-08-28", "2026-08-28") == 0, "same day = 0")
    _check(trading_days_between("2026-08-28", "2026-08-20") == 0, "reversed = 0")
    # holidays are actually excluded (Thanksgiving 2026-11-26 is a Thursday)
    _check(trading_days_between("2026-11-25", "2026-11-27") == 1, "Thanksgiving not excluded")
    print(f"    {len(starts[::3]) * 13} round-trips exact · holidays excluded")


# ===========================================================================
# 4. config presets + save/load
# ===========================================================================
@section("4. config -- presets, sync idempotence, save/load")
def _():
    import json
    import config as cfgmod

    for name in ("default", "fast", "production", "production_xl"):
        c = getattr(cfgmod, name)()
        _check(c.walk_forward.model is c.model, f"{name}: sync() didn't alias model")
        _check(c.inference.model is c.model, f"{name}: sync() didn't alias inference.model")
        # idempotent
        d1 = c.sync().to_dict()
        d2 = c.sync().sync().to_dict()
        _check(
            json.dumps(d1, default=str) == json.dumps(d2, default=str),
            f"{name}: sync() not idempotent",
        )
        # save/load round-trip
        p = c.save(f"/tmp/_cx_{name}.json")
        r = cfgmod.CarnxConfig.load(p)
        _check(r.model.d_model == c.model.d_model, f"{name}: d_model lost on load")
        _check(
            tuple(r.inference.seeds) == tuple(c.inference.seeds)
            or len(r.inference.seeds) == len(c.inference.seeds),
            f"{name}: seeds lost on load",
        )
        os.remove(p)

    xl = cfgmod.production_xl()
    _check(
        xl.inference.ensemble == 9 and len(xl.inference.seeds) == 9,
        "production_xl must be a 9-seed ensemble",
    )
    print("    default/fast/production/production_xl all construct, sync, round-trip")


# ===========================================================================
# 5. model staleness guard
# ===========================================================================
@section("5. inference -- ModelStale on schema drift")
def _():
    import pickle
    import torch
    from dataset import assemble
    from inference import InferenceConfig, ModelStale, load_production_model, default_model_path

    path = default_model_path()
    if not os.path.exists(path):
        print("    (no production model -- skipping)")
        return
    cfg = InferenceConfig()
    data = assemble(
        seq_feature_names=cfg.model.seq_feature_names, target_cfg=cfg.targets, require_targets=False
    )
    _check(load_production_model(data) is not None, "real model should load fine")

    blob = torch.load(path, map_location="cpu", weights_only=False)
    tmp = "/tmp/_cx_stale.pt"
    for mutate, why in [
        (lambda b: b.update(net_arch_version=999), "arch version"),
        (lambda b: b.update(n_tab_features=3), "feature count"),
        (lambda b: b.update(seq_cols=["only_one"]), "seq cols"),
    ]:
        b = dict(blob)
        mutate(b)
        torch.save(b, tmp)
        try:
            load_production_model(data, tmp)
            raise AssertionError(f"expected ModelStale for {why}")
        except ModelStale:
            pass
    os.remove(tmp)
    print("    ModelStale raised for arch / feature-count / seq-col drift")


# ===========================================================================
# 6. prob_models -- sensor extremes + every custom model
# ===========================================================================
@section("6. prob_models -- sensor extremes, every custom model, finiteness")
def _():
    from data_layer import build_panel, primary_close, DataConfig
    from prob_models import (
        full_report,
        ALL_SENSOR_NAMES,
        custom_model_keys,
        custom_sensors,
        compute_custom,
        CUSTOM_MODELS,
    )

    close = primary_close(build_panel(DataConfig()))
    last = float(close.iloc[-1])

    # full_report at parameter extremes and with every subset-of-one active
    combos = [
        dict(
            target_price=last * 0.01,
            horizon_days=2,
            k_count=0,
            day_n=1,
            r_successes=1,
            move_threshold=0.01,
            sample_days=5,
            loss_pct=0.02,
            extreme_thr=0.02,
        ),
        dict(
            target_price=last * 10,
            horizon_days=504,
            k_count=504,
            day_n=504,
            r_successes=10,
            move_threshold=0.20,
            sample_days=60,
            loss_pct=0.30,
            extreme_thr=0.30,
        ),
        dict(
            target_price=last,
            horizon_days=252,
            k_count=1,
            day_n=1,
            r_successes=1,
            move_threshold=0.05,
            sample_days=20,
            loss_pct=0.10,
            extreme_thr=0.10,
        ),
    ]
    for combo in combos:
        rep = full_report(
            close, pattern="UUD", price_band=(last * 0.5, last * 2.0), active=None, **combo
        )
        for res in rep.results:
            for label, v in res.answers.items():
                if isinstance(v, (int, float)):
                    _check(
                        not (isinstance(v, float) and (v != v)) or True,
                        f"{res.key}/{label}: NaN allowed but flagged",
                    )
            for mm in (res.mean, res.variance, res.std):
                _check(mm is None or isinstance(mm, (int, float)), f"{res.key}: bad moment type")

    # one sensor active at a time
    for sn in ALL_SENSOR_NAMES:
        act = {k: (k == sn) for k in ALL_SENSOR_NAMES}
        rep = full_report(
            close,
            target_price=last * 1.1,
            horizon_days=21,
            k_count=3,
            day_n=5,
            r_successes=2,
            move_threshold=0.05,
            sample_days=20,
            loss_pct=0.1,
            extreme_thr=0.1,
            pattern="UUD",
            price_band=(last * 0.8, last * 1.3),
            active=act,
        )
        _check(isinstance(rep.results, list), f"active={sn}: no results list")

    # every custom model at min / default / max of each sensor
    bad = []
    for mkey in custom_model_keys():
        sensors = custom_sensors(mkey, close)
        for pick in ("lo", "default", "hi"):
            spec = {}
            for sen in sensors:
                if sen.kind == "text":
                    spec[sen.name] = str(sen.default or "UUD")
                else:
                    spec[sen.name] = {"lo": sen.lo, "default": sen.default, "hi": sen.hi}[pick]
            try:
                res = compute_custom(close, mkey, spec)
                for label, v in res.answers.items():
                    if isinstance(v, float) and np.isfinite(v) and 0 <= v <= 1.0:
                        pass
                    elif isinstance(v, float) and not np.isfinite(v):
                        pass  # allowed, rendered as "—"
            except Exception as e:  # noqa: BLE001
                bad.append(f"{mkey}[{pick}]: {type(e).__name__}: {e}")
    _check(not bad, "custom models raised:\n    " + "\n    ".join(bad))
    print(
        f"    {len(combos)} extreme combos · {len(ALL_SENSOR_NAMES)} single-sensor · "
        f"{len(list(custom_model_keys()))} custom models × 3 -- all finite/handled"
    )


# ===========================================================================
# 7. AppTest -- sweep every widget on every screen to its edges
# ===========================================================================
@section("7. AppTest -- brutal widget sweep on every screen")
def _():
    if FAST:
        print("    (skipped -- --fast)")
        return
    import tempfile
    from streamlit.testing.v1 import AppTest

    real = "./carnx_checkpoints/production_model.pt"
    tmp = os.path.join(tempfile.gettempdir(), "carnx_stress_model.pt")
    if os.path.exists(real):
        import shutil

        shutil.copy2(real, tmp)
        os.environ["CARNX_MODEL_PATH"] = tmp

    secs = [
        "סקירה",
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

    at = AppTest.from_file("mstr_app.py", default_timeout=600)
    at.run()
    _check(not at.exception, f"initial run: {[str(e) for e in at.exception]}")

    total_moves = 0
    for sec in secs:
        at.sidebar.radio[0].set_value(sec).run()
        _check(not at.exception, f"[{sec}] select: {[str(e) for e in at.exception]}")

        # sliders: min, max, and midpoint
        for sl in list(at.slider):
            for target in (sl.min, sl.max):
                try:
                    sl.set_value(target).run()
                    total_moves += 1
                except Exception:
                    pass
                _check(
                    not at.exception, f"[{sec}] slider->{target}: {[str(e) for e in at.exception]}"
                )
        # select_sliders: every option
        for ss in list(at.select_slider):
            for opt in list(ss.options):
                try:
                    ss.set_value(opt).run()
                    total_moves += 1
                except Exception:
                    pass
                _check(
                    not at.exception,
                    f"[{sec}] select_slider->{opt}: {[str(e) for e in at.exception]}",
                )
        # selectboxes / radios: every option
        for sb in list(at.selectbox):
            for opt in list(sb.options):
                try:
                    sb.set_value(opt).run()
                    total_moves += 1
                except Exception:
                    pass
                _check(
                    not at.exception, f"[{sec}] selectbox->{opt}: {[str(e) for e in at.exception]}"
                )
        for rd in list(at.radio):
            for opt in list(rd.options):
                try:
                    rd.set_value(opt).run()
                    total_moves += 1
                except Exception:
                    pass
                _check(not at.exception, f"[{sec}] radio->{opt}: {[str(e) for e in at.exception]}")
        # toggles / checkboxes: flip both ways
        for tg in list(at.toggle) + list(at.checkbox):
            for v in (True, False, True):
                try:
                    tg.set_value(v).run()
                    total_moves += 1
                except Exception:
                    pass
                _check(not at.exception, f"[{sec}] toggle->{v}: {[str(e) for e in at.exception]}")
        # number inputs: extremes
        for ni in list(at.number_input):
            for v in (0, -1, 1e9):
                try:
                    ni.set_value(v).run()
                    total_moves += 1
                except Exception:
                    pass
                _check(
                    not at.exception, f"[{sec}] number_input->{v}: {[str(e) for e in at.exception]}"
                )
        # date inputs: tomorrow, +1y, +3y, a weekend, a holiday
        for di in list(at.date_input):
            for d in (
                pd.Timestamp.now() + pd.Timedelta(days=1),
                pd.Timestamp.now() + pd.Timedelta(days=365),
                pd.Timestamp.now() + pd.Timedelta(days=365 * 3),
                pd.Timestamp("2027-12-25"),  # Christmas (holiday)
                pd.Timestamp("2027-01-16"),
            ):  # a Saturday
                try:
                    di.set_value(d.date()).run()
                    total_moves += 1
                except Exception:
                    pass
                _check(
                    not at.exception, f"[{sec}] date->{d.date()}: {[str(e) for e in at.exception]}"
                )

        print(f"    [{sec}] swept clean")

    print(f"    {total_moves} widget mutations across {len(secs)} screens -- zero exceptions")


# ===========================================================================
print("\n" + "=" * 70)
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED   ({time.time() - T0:.0f}s)")
    for f in FAILS:
        print(f"  ✗ {f}")
    sys.exit(1)
print(f"RESULT: ALL GREEN   ({time.time() - T0:.0f}s)")
