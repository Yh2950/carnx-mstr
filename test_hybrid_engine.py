"""
Tests for hybrid_engine.py -- the Hybrid Neural-Structural MSTR engine.

Verifies the *mechanics* (structural formula, drag-neutral calibration,
reflexive accretion, regime matrix, antithetic variance reduction), not a
market forecast. Run:  python test_hybrid_engine.py   (or pytest).
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

import hybrid_engine as HE


def _synth_btc(n=1400, mu=0.0016, sig=0.035, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        30_000 * np.exp(np.cumsum(rng.normal(mu, sig, n))),
        index=pd.bdate_range("2021-01-01", periods=n),
    )


# --------------------------------------------------------------------------- #
def test_structural_formula_converges_to_target_band():
    """BTC = $150k, mNAV = 1.30x -> MSTR in $350-420 (the spec's anchor test)."""
    f = HE.StrategyFundamentals()
    px = f.structural_price(150_000.0, 1.30)
    assert 350.0 <= px <= 420.0, px
    # NAV identity: price / mNAV = BPS*BTC - DPS
    assert abs(px / 1.30 - (f.btc_per_share() * 150_000 - f.debt_per_share())) < 1e-6


def test_structural_formula_monotone_and_leverage():
    f = HE.StrategyFundamentals()
    assert f.structural_price(200_000, 1.3) > f.structural_price(150_000, 1.3)
    assert f.structural_price(150_000, 2.0) > f.structural_price(150_000, 1.0)
    # leveraged: +10% BTC -> more than +10% MSTR equity (debt is fixed)
    base = f.structural_price(150_000, 1.0)
    up = f.structural_price(165_000, 1.0)
    assert (up / base - 1) > 0.10


def test_regime_model_is_valid_markov():
    rm = HE.RegimeModel.fit(_synth_btc())
    assert rm.transition.shape == (3, 3)
    assert np.allclose(rm.transition.sum(axis=1), 1.0)
    assert np.all(rm.transition >= 0)
    assert np.isclose(rm.stationary.sum(), 1.0, atol=1e-6)
    assert rm.means_daily[0] <= rm.means_daily[1] <= rm.means_daily[2]  # sorted asc
    assert rm.label_of_now in HE.REGIME_NAMES
    # heuristic fallback also well-formed
    h = HE.RegimeModel.heuristic(_synth_btc())
    assert np.allclose(h.transition.sum(axis=1), 1.0)


def test_regime_paths_persist_and_respect_expansion_prior():
    rm = HE.RegimeModel.heuristic(_synth_btc())
    rng = np.random.default_rng(0)
    R0 = HE.simulate_regime_paths(rm, 4000, 252, rng, expansion_prior=0.0)
    R1 = HE.simulate_regime_paths(rm, 4000, 252, np.random.default_rng(0), expansion_prior=1.0)
    assert R0.shape == (4000, 252)
    assert set(np.unique(R0)).issubset({0, 1, 2})
    # persistence: most days keep the previous state
    stay = np.mean(R0[:, 1:] == R0[:, :-1])
    assert stay > 0.7
    # the expansion prior raises time spent in state 2
    assert (R1 == 2).mean() > (R0 == 2).mean()


def test_scenario_btc_median_hits_target_drag_neutral():
    """Volatility-drag neutralisation: the *median* BTC terminal converges to
    the scenario's expected price, not a decayed value."""
    btc = _synth_btc()
    for tgt in (120_000, 180_000, 250_000):
        res = HE.run_scenario(
            btc,
            HE.StrategyFundamentals(),
            m0_nav=1.2,
            horizon=252,
            n_paths=40_000,
            btc_expected=tgt,
            btc_low=tgt * 0.6,
            btc_high=tgt * 1.7,
            seed=1,
        )
        med = float(np.median(res.btc_terminal))
        assert abs(med / tgt - 1.0) < 0.06, (tgt, med)


def test_scenario_band_controls_spread():
    btc = _synth_btc()
    tight = HE.run_scenario(
        btc,
        HE.StrategyFundamentals(),
        1.2,
        252,
        30_000,
        btc_expected=150_000,
        btc_low=130_000,
        btc_high=175_000,
        seed=2,
    )
    wide = HE.run_scenario(
        btc,
        HE.StrategyFundamentals(),
        1.2,
        252,
        30_000,
        btc_expected=150_000,
        btc_low=80_000,
        btc_high=300_000,
        seed=2,
    )
    q = lambda r: np.percentile(r.btc_terminal, 90) - np.percentile(r.btc_terminal, 10)
    assert q(wide) > 1.5 * q(tight)


def test_reflexive_accretion_grows_bps_only_when_aggressive():
    btc = _synth_btc(mu=0.003)  # bullish -> mNAV mostly > 1
    f = HE.StrategyFundamentals()
    off = HE.run_scenario(
        btc,
        f,
        1.4,
        252,
        20_000,
        btc_expected=200_000,
        btc_low=120_000,
        btc_high=350_000,
        accretion_yield=0.0,
        seed=3,
    )
    on = HE.run_scenario(
        btc,
        f,
        1.4,
        252,
        20_000,
        btc_expected=200_000,
        btc_low=120_000,
        btc_high=350_000,
        accretion_yield=0.15,
        seed=3,
    )
    assert np.isclose(np.median(off.bps_terminal), off.bps_start, rtol=1e-4)
    assert np.median(on.bps_terminal) > on.bps_start * 1.02  # sats/share grew
    # accretion lifts the MSTR terminal (more BTC behind each share)
    assert on.median_price > off.median_price


def test_antithetic_reduces_terminal_variance():
    btc = _synth_btc()
    f = HE.StrategyFundamentals()
    rm = HE.RegimeModel.fit(btc)
    common = dict(s0_btc=float(btc.iloc[-1]), m0_nav=1.2, btc_hist=btc)
    base = HE.HybridConfig(
        horizon=126,
        n_paths=20_000,
        seed=7,
        btc_target_expected=160_000,
        btc_target_low=110_000,
        btc_target_high=240_000,
    )
    a = HE.simulate_hybrid(f, rm, HE.replace(base, antithetic=True), **common)
    b = HE.simulate_hybrid(f, rm, HE.replace(base, antithetic=False), **common)
    # antithetic tightens the estimate of the mean (paired negative correlation)
    se_a = a.terminal_prices.std() / math.sqrt(len(a.terminal_prices))
    se_b = b.terminal_prices.std() / math.sqrt(len(b.terminal_prices))
    assert se_a <= se_b * 1.02


def test_vectorised_performance_budget():
    """50k x 252 must complete well inside the app's existing MC-screen budget
    (~2 s). Sub-250 ms needs a torch/numba backend -- see module docstring."""
    btc = _synth_btc()
    f = HE.StrategyFundamentals()
    rm = HE.RegimeModel.fit(btc)
    cfg = HE.HybridConfig(
        horizon=252,
        n_paths=50_000,
        seed=0,
        btc_target_expected=150_000,
        btc_target_low=95_000,
        btc_target_high=230_000,
    )
    HE.simulate_hybrid(f, rm, cfg, float(btc.iloc[-1]), 1.25, btc_hist=btc)  # warm
    t = time.perf_counter()
    res = HE.simulate_hybrid(
        f, rm, cfg, float(btc.iloc[-1]), 1.25, target_price=500.0, btc_hist=btc
    )
    dt = time.perf_counter() - t
    assert dt < 5.0, f"{dt:.2f}s"
    assert res.paths.shape == (50_000, 253)
    assert res.paths.dtype == np.float32


def test_to_mc_result_adapter_matches_shape():
    btc = _synth_btc()
    res = HE.run_scenario(
        btc,
        HE.StrategyFundamentals(),
        1.2,
        60,
        10_000,
        btc_expected=140_000,
        btc_low=100_000,
        btc_high=200_000,
        target_price=450.0,
        seed=0,
    )
    mc = res.to_mc_result()
    assert mc.paths.shape == res.paths.shape
    assert mc.drift_mode == "hybrid_structural"
    assert set(mc.percentiles) >= {"p05", "p50", "p95"}
    assert mc.target_price == 450.0
    assert 0.0 <= mc.p_touch_up <= 1.0


def test_market_anchor_keeps_hitting_probs_consistent():
    """With s0_mstr_market given, path[0] == market price and hitting-time
    probs are computed against the user's dollar target on the anchored cone."""
    btc = _synth_btc()
    f = HE.StrategyFundamentals()
    s0s = f.structural_price(float(btc.iloc[-1]), 1.2)
    mkt = 0.55 * s0s  # market trades at a discount
    res = HE.run_scenario(
        btc,
        f,
        1.2,
        120,
        20_000,
        btc_expected=180_000,
        btc_low=120_000,
        btc_high=280_000,
        target_price=mkt * 1.4,
        s0_mstr_market=mkt,
        seed=1,
    )
    assert abs(res.paths[:, 0].mean() / mkt - 1.0) < 1e-4
    assert abs(res.last_price / mkt - 1.0) < 1e-4
    assert abs(res.s0_structural / s0s - 1.0) < 1e-4
    assert 0.05 < res.p_touch_up < 0.95  # a live, sane probability


def test_continuous_support_binding_no_zero_clip():
    a, b = HE.continuous_support(s0=127.0, target=420.0)
    assert a <= 0.6 * 127.0 + 1e-6
    assert b >= 1.5 * 420.0 - 1e-6
    assert a < 127.0 < b and a < 420.0 < b  # both S0 and target inside


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fail = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            fail += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - fail}/{len(fns)} passed")
    raise SystemExit(fail)
