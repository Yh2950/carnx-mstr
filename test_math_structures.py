"""
Tests for math_structures.py -- the mathematical-structures companion.

Checks the *maths* (closed forms, conservation laws, known limits), not the
forecasts. Run:  python test_math_structures.py   (or pytest).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import math_structures as ms


def _synthetic(mu_d=0.0006, sig_d=0.03, n=1000, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(mu_d, sig_d, n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2021-01-01", periods=n))


def test_pascal_row_sums_and_symmetry():
    tri = ms.pascal_triangle(20)
    for n in range(20):
        row = tri[n, : n + 1]
        assert abs(row.sum() - 2**n) < 1e-6 * 2**n  # sum = 2^n
        assert np.allclose(row, row[::-1])  # symmetric
        if n >= 2:  # Pascal's rule
            assert (
                abs(tri[n, 1:n].sum() - (tri[n - 1, : n - 1].sum() + tri[n - 1, 1:n].sum())) < 1e-6
            )


def test_crr_is_risk_neutral_and_converges_to_black_scholes():
    close = _synthetic()
    lat = ms.crr_lattice(close, horizon_days=20, steps=60, r_annual=0.05)
    # risk-neutral expectation of terminal price = S0 e^{rT}
    assert abs(lat.expected_terminal_rn / lat.bs_lognormal_mean - 1.0) < 1e-6
    # terminal probabilities are a normalised Pascal row
    assert abs(lat.terminal_probs_rn.sum() - 1.0) < 1e-12
    # more steps -> closer to the log-normal (Black-Scholes) mean
    err = []
    for s in (10, 40, 160):
        L = ms.crr_lattice(close, 20, s, r_annual=0.05)
        err.append(abs(L.expected_terminal_rn / L.bs_lognormal_mean - 1.0))
    assert err[0] >= err[-1] - 1e-9


def test_binomial_to_normal_tv_decreases():
    conv = ms.binomial_to_normal_convergence(p=0.5)
    tv = conv.total_variation
    assert np.all(np.diff(tv) < 1e-9)  # monotone down
    assert tv[-1] < 0.05


def test_catalan_recurrence():
    # C_{n+1} = sum_i C_i C_{n-i}
    C = [ms.catalan(n) for n in range(12)]
    for n in range(11):
        assert C[n + 1] == sum(C[i] * C[n - i] for i in range(n + 1))
    assert C[:6] == [1, 1, 2, 5, 14, 42]


def test_drawdown_survival_bounds_and_monotone():
    close = _synthetic(mu_d=0.0)
    a = ms.drawdown_survival(close, floor_pct=0.10, horizon_days=20, n_mc=20000)
    b = ms.drawdown_survival(close, floor_pct=0.30, horizon_days=20, n_mc=20000)
    for r in (a, b):
        for p in (r.p_survive_combinatorial, r.p_survive_drifted, r.p_survive_montecarlo):
            assert 0.0 <= p <= 1.0
    # a deeper floor is easier to stay above
    assert b.p_survive_montecarlo >= a.p_survive_montecarlo - 0.02
    # MC and empirical should agree to within sampling noise
    assert abs(a.p_survive_montecarlo - a.p_survive_empirical) < 0.15


def test_dmd_recovers_known_period():
    # a genuine cycle in the *log-price level* (period ~40 days) + drift + noise
    rng = np.random.default_rng(1)
    t = np.arange(900)
    level = (
        0.0008 * t + 0.10 * np.sin(2 * np.pi * t / 40.0) + np.cumsum(rng.normal(0, 0.006, t.size))
    )
    px = pd.Series(100 * np.exp(level), index=pd.bdate_range("2021-01-01", periods=t.size))
    dm = ms.dmd_spectrum(px, rank=8, delay=50, lookback=900)
    periods = dm.period_days[np.isfinite(dm.period_days)]
    assert np.any(np.abs(periods - 40.0) < 6.0)  # the 40-day cycle is found
    assert dm.spectral_radius <= 1.05
    assert dm.reconstruction_r2 > 0.1


def test_fokker_planck_conserves_mass_and_matches_gaussian():
    close = _synthetic(mu_d=0.0004, sig_d=0.025)
    fp = ms.fokker_planck_density(close, horizon_days=20, local_vol=False)
    assert fp.mass_drift < 1e-6  # conservation
    assert fp.analytic_gaussian_check < 0.03  # vs exact Gaussian
    # spread grows ~ sqrt(t)
    t = fp.t_grid[1:]
    sd = np.array(
        [
            math.sqrt(np.dot(fp.density[i], (fp.x_grid - 0) ** 2) * (fp.x_grid[1] - fp.x_grid[0]))
            for i in range(1, len(fp.t_grid))
        ]
    )
    ratio = sd / np.sqrt(t)
    assert ratio.std() / ratio.mean() < 0.1  # roughly constant


def test_jump_diffusion_identifiable_and_bounded():
    # data with genuine, separable jumps
    rng = np.random.default_rng(7)
    r = rng.normal(0.0005, 0.015, 800)
    jump_idx = rng.choice(np.arange(800), 15, replace=False)
    r[jump_idx] += rng.choice([-1, 1], 15) * rng.uniform(0.10, 0.20, 15)
    px = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2021-01-01", periods=800))
    jd = ms.fit_jump_diffusion(px, lookback=800)
    assert 0.0 < jd.jump_intensity_annual < 0.15 * ms.TRADING_DAYS_YEAR + 1
    # diffusion scale recovered, jump component clearly wider
    assert abs(jd.sigma_diffusion_annual / math.sqrt(ms.TRADING_DAYS_YEAR) - 0.015) < 0.006
    assert jd.jump_std > 3.0 * jd.sigma_diffusion_annual / math.sqrt(ms.TRADING_DAYS_YEAR)
    assert jd.lr_pvalue < 0.01  # jumps detected


def test_jump_diffusion_null_on_clean_gaussian():
    px = _synthetic(mu_d=0.0003, sig_d=0.02, n=800, seed=5)
    jd = ms.fit_jump_diffusion(px, lookback=800)
    # no real jumps -> the likelihood-ratio test should not scream, and the
    # diffusion scale should match the (only) generating sigma
    assert jd.lr_pvalue > 0.01
    assert abs(jd.sigma_diffusion_annual / math.sqrt(ms.TRADING_DAYS_YEAR) - 0.02) < 0.006


def test_path_statistics_ranges():
    close = _synthetic()
    ps = ms.path_statistics(close)
    assert 0.0 < ps.up_day_prob < 1.0
    assert 0.0 <= ps.up_day_binom_p <= 1.0
    assert 0.0 <= ps.frac_time_at_high <= 1.0
    assert ps.longest_up_run >= 1 and ps.longest_down_run >= 1


def test_fibonacci_null_on_random_walk():
    close = _synthetic(n=1200, seed=7)
    ft = ms.fibonacci_pivot_test(close, lookback=1000)
    # a random walk must not show Fib clustering beyond chance
    assert ft.p_value > 0.05
    lv = ms.fibonacci_levels(close)
    assert lv.swing_low <= lv.current_price or lv.swing_high >= lv.current_price


def test_puzzle_map_sane():
    close = _synthetic()
    pm = ms.strategic_puzzle_map(close)
    assert pm.egg_drop_probes >= 1
    assert pm.ternary_best_horizon in (1, 5, 20)
    assert pm.staircase_reachable_states == 21
    assert pm.bezout_alignment["gcd_days"] >= 1


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
