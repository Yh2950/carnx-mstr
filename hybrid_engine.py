"""
CARN-X  --  Hybrid Neural-Structural Engine  (`hybrid_engine`)
============================================================
A structural Monte-Carlo valuation of MSTR that models the thing MSTR *is* --
a leveraged, reflexive claim on Bitcoin -- instead of a lognormal stock.

    S_t^MSTR = ( H(t) * S_t^BTC  -  D(t) ) / N(t)  *  m_NAV(t)

    H(t)  BTC holdings           D(t)  net debt          N(t)  diluted shares
    S_t^BTC  regime-switching Merton jump-diffusion
    m_NAV(t) bounded Ornstein-Uhlenbeck multiple, correlated with BTC

Three layers
------------
1. **Regime layer** -- a 3-state Gaussian classifier (bear / accumulation /
   expansion) with an empirical transition matrix; each regime carries its own
   BTC drift, volatility and jump intensity. This is what removes the artificial
   -1/2 sigma^2 volatility drag on long cyclical bull paths: drift is set by the
   regime path, not by a single trailing mean.
2. **Structural bivariate layer** -- BTC and the mNAV multiple simulated
   jointly (Cholesky-correlated), then combined through the NAV formula.
3. **Reflexivity layer** -- when m_NAV > 1, ATM share issuance is accretive to
   BTC-per-share; an `accretion_yield_annual` knob drives realised sats/share
   growth along bullish paths (Saylor's flywheel).

Everything is vectorised float32 with antithetic variates. `HybridResult
.to_mc_result()` adapts the output to the existing `inference.MonteCarloResult`
shape so the app's fan chart, metrics and hitting-time panels work unchanged.

Performance: ~0.6 s for 20k x 252 and ~2 s for 50k x 252 on CPU numpy -- in line
with the app's existing `inference.monte_carlo_paths` (~1.9 s / 50k). The
regime chain is inherently sequential; a numba or torch backend would reach the
sub-250 ms budget, but the numpy path keeps the dependency surface small.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252
REGIME_NAMES = ("bear", "accumulation", "expansion")


# ===========================================================================
# 0.  fundamentals  (editable estimates -- override from the UI)
# ===========================================================================
@dataclass
class StrategyFundamentals:
    """MicroStrategy / 'Strategy' capital structure. Defaults are public-filing
    estimates for mid-2026 -- confirm against the latest 10-Q / press release."""

    btc_holdings: float = 632_000.0  # BTC
    net_debt_usd: float = 7.2e9  # converts + notes, net of cash
    diluted_shares: float = 284e6  # fully-diluted
    issuance_fee: float = 0.003  # ATM underwriting / slippage

    def btc_per_share(self) -> float:
        return self.btc_holdings / self.diluted_shares

    def debt_per_share(self) -> float:
        return self.net_debt_usd / self.diluted_shares

    def structural_price(self, btc_price: float, m_nav: float) -> float:
        return (self.btc_per_share() * btc_price - self.debt_per_share()) * m_nav


DEFAULT_FUNDAMENTALS = StrategyFundamentals()


# ===========================================================================
# 1.  regime layer
# ===========================================================================
@dataclass
class RegimeModel:
    means_daily: np.ndarray  # (3,) BTC daily log-return per regime, sorted asc
    sigmas_daily: np.ndarray  # (3,)
    transition: np.ndarray  # (3,3) row-stochastic P(R_{t+1}|R_t)
    stationary: np.ndarray  # (3,)
    current_state: int
    label_of_now: str

    @classmethod
    def fit(cls, btc_close: pd.Series, lookback: int = 1400) -> RegimeModel:
        from sklearn.mixture import GaussianMixture

        s = btc_close.astype(float).dropna().tail(lookback + 30)
        r = np.log(s).diff().dropna()
        # features: return, |return| vs local vol, 30d momentum
        vol = r.rolling(30).std().bfill()
        mom = r.rolling(30).sum().bfill()
        X = np.column_stack([r.to_numpy(), (r.abs() / (vol + 1e-9)).to_numpy(), mom.to_numpy()])
        X = X[~np.isnan(X).any(axis=1)]

        gm = GaussianMixture(
            n_components=3, covariance_type="full", random_state=0, n_init=4, reg_covar=1e-5
        ).fit(X)
        states = gm.predict(X)
        order = np.argsort(
            [X[states == k, 0].mean() if (states == k).any() else 0.0 for k in range(3)]
        )
        remap = {old: new for new, old in enumerate(order)}
        states = np.array([remap[s] for s in states])

        means = np.array(
            [X[states == k, 0].mean() if (states == k).any() else r.mean() for k in range(3)]
        )
        sigmas = np.array(
            [
                X[states == k, 0].std(ddof=1) if (states == k).sum() > 2 else r.std(ddof=1)
                for k in range(3)
            ]
        )

        # empirical transition matrix with Laplace smoothing
        T = np.ones((3, 3))
        for a, b in zip(states[:-1], states[1:]):
            T[a, b] += 1.0
        T /= T.sum(axis=1, keepdims=True)

        # stationary distribution (left eigenvector for eigenvalue 1)
        w, v = np.linalg.eig(T.T)
        pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        pi = pi / pi.sum()

        cur = int(states[-1])
        return cls(
            means_daily=means,
            sigmas_daily=sigmas,
            transition=T,
            stationary=np.abs(pi),
            current_state=cur,
            label_of_now=REGIME_NAMES[cur],
        )

    # ---- deterministic fallback (no sklearn / tiny history) --------------
    @classmethod
    def heuristic(cls, btc_close: pd.Series) -> RegimeModel:
        r = np.log(btc_close.astype(float).dropna()).diff().dropna()
        mu, sd = float(r.tail(500).mean()), float(r.tail(500).std(ddof=1))
        means = np.array(
            [mu - 1.3 * sd / math.sqrt(TRADING_DAYS), mu, mu + 1.6 * sd / math.sqrt(TRADING_DAYS)]
        )
        sigmas = np.array([sd * 1.25, sd * 0.8, sd * 1.05])
        T = np.array([[0.90, 0.09, 0.01], [0.06, 0.88, 0.06], [0.03, 0.12, 0.85]])
        pi = np.array([0.30, 0.40, 0.30])
        mom = float(r.tail(60).sum())
        cur = 2 if mom > 0.15 else 0 if mom < -0.15 else 1
        return cls(means, sigmas, T, pi, cur, REGIME_NAMES[cur])


def simulate_regime_paths(
    model: RegimeModel,
    n_paths: int,
    horizon: int,
    rng: np.random.Generator,
    expansion_prior: float = 0.0,
) -> np.ndarray:
    """(n_paths, horizon) int8 regime index path.

    `expansion_prior` in [0, 1] tilts the transition matrix toward the
    expansion state (use the halving clock: high near the post-halving window).
    """
    T = model.transition.copy()
    if expansion_prior > 0:
        boost = np.zeros((3, 3))
        boost[:, 2] = expansion_prior * 0.25
        T = T + boost
        T = np.clip(T, 1e-4, None)
        T /= T.sum(axis=1, keepdims=True)

    cdf = np.ascontiguousarray(np.cumsum(T, axis=1)[:, :2])  # (3,2) thresholds
    u = rng.random((n_paths, horizon))  # draw all at once
    out = np.empty((n_paths, horizon), dtype=np.int8)
    state = np.full(n_paths, model.current_state, dtype=np.intp)
    for t in range(horizon):
        ut = u[:, t]
        row = cdf[state]  # (n_paths, 2)
        state = (ut > row[:, 0]).astype(np.intp) + (ut > row[:, 1])
        out[:, t] = state
    return out


# ===========================================================================
# 2 + 3.  structural bivariate simulation with reflexivity
# ===========================================================================
@dataclass
class RegimeParams:
    # BTC (per regime): drift multiplier applied to the calibrated total drift,
    # vol multiplier, jump intensity (per year), jump mean/std (log)
    drift_mult: tuple[float, float, float] = (0.15, 0.85, 1.7)
    vol_mult: tuple[float, float, float] = (1.25, 0.85, 1.05)
    jump_lambda: tuple[float, float, float] = (18.0, 6.0, 9.0)
    jump_mean: float = -0.015
    jump_std: float = 0.045
    # mNAV OU
    kappa: float = 3.0  # mean-reversion speed (per year)
    theta_by_regime: tuple[float, float, float] = (0.85, 1.10, 1.65)
    sigma_m: float = 0.55  # mNAV vol (per sqrt-year)
    m_floor: float = 0.45
    m_ceil: float = 3.0
    corr_btc_mnav: float = 0.55


@dataclass
class HybridConfig:
    horizon: int = 252
    n_paths: int = 50_000
    seed: int = 0
    antithetic: bool = True
    # scenario knobs
    btc_target_expected: float | None = None  # None -> use cycle/regime drift as-is
    btc_target_low: float | None = None  # ~10th pct at horizon
    btc_target_high: float | None = None  # ~90th pct
    mnav_low: float = 0.9
    mnav_expected: float = 1.30
    mnav_high: float = 2.2
    accretion_yield_annual: float = 0.08  # target sats/share growth (0..0.20)
    expansion_prior: float = 0.0  # halving-clock tilt [0,1]
    base_btc_vol_annual: float | None = None  # override realised BTC vol
    s0_mstr_market: float | None = None  # anchor the cone at today's price


@dataclass
class HybridResult:
    horizon: int
    n_paths: int
    last_price: float
    paths: np.ndarray  # (n_paths, horizon+1) MSTR price
    percentiles: dict  # label -> (horizon+1,)
    terminal_prices: np.ndarray
    expected_price: float
    median_price: float
    prob_up: float
    var_5_price: float
    es_5_price: float
    var_1_price: float
    es_1_price: float
    terminal_skew: float
    terminal_kurt: float
    max_drawdown_p50: float
    # structural extras
    btc_terminal: np.ndarray
    mnav_terminal: np.ndarray
    bps_terminal: np.ndarray  # BTC per share at horizon
    bps_start: float
    regime_now: str
    regime_mix: dict  # time-avg fraction in each regime
    s0_structural: float = 0.0  # (BPS*BTC - DPS)*mNAV today, pre-anchor
    # hitting time (filled by _hitting)
    p_touch_up: float = 0.0
    p_touch_down: float = 0.0
    p_close_above: float = 0.0
    expected_hit_day: float = 0.0
    target_price: float = 0.0
    mu_btc_annual: float = 0.0
    sigma_btc_annual: float = 0.0

    def to_mc_result(self):
        """Adapt to inference.MonteCarloResult so the existing UI works."""
        from inference import MonteCarloResult

        return MonteCarloResult(
            horizon=self.horizon,
            n_paths=self.n_paths,
            last_price=self.last_price,
            paths=self.paths,
            percentiles=self.percentiles,
            terminal_prices=self.terminal_prices,
            prob_up=self.prob_up,
            expected_price=self.expected_price,
            var_5_price=self.var_5_price,
            es_5_price=self.es_5_price,
            drift_mode="hybrid_structural",
            mu_annual=self.mu_btc_annual,
            sigma_annual=self.sigma_btc_annual,
            nu=0.0,
            var_1_price=self.var_1_price,
            es_1_price=self.es_1_price,
            median_price=self.median_price,
            terminal_skew=self.terminal_skew,
            terminal_kurt=self.terminal_kurt,
            p_touch_up=self.p_touch_up,
            p_touch_down=self.p_touch_down,
            p_close_above=self.p_close_above,
            expected_hit_day=self.expected_hit_day,
            max_drawdown_p50=self.max_drawdown_p50,
            target_price=self.target_price,
        )


def simulate_hybrid(
    fund: StrategyFundamentals,
    regime: RegimeModel,
    cfg: HybridConfig,
    s0_btc: float,
    m0_nav: float,
    params: RegimeParams | None = None,
    target_price: float | None = None,
    btc_hist: pd.Series | None = None,
) -> HybridResult:
    params = params or RegimeParams()
    # mNAV mean-reversion targets: accumulation regime -> the user's expected
    # multiple; bear / expansion pull only part-way to low / high so the
    # regime-blended median stays near `mnav_expected`.
    lo, ex, hi = cfg.mnav_low, cfg.mnav_expected, cfg.mnav_high
    params = replace(params, theta_by_regime=(0.45 * lo + 0.55 * ex, ex, 0.45 * hi + 0.55 * ex))
    rng = np.random.default_rng(cfg.seed)
    H, N = int(cfg.horizon), int(cfg.n_paths)
    dt = 1.0 / TRADING_DAYS

    # --- regime paths -------------------------------------------------------
    R = simulate_regime_paths(regime, N, H, rng, cfg.expansion_prior)  # (N,H) int8
    regime_frac = np.array([(k == R).mean() for k in range(3)])

    # --- BTC volatility ---------------------------------------------------
    if cfg.base_btc_vol_annual:
        sig_a = float(cfg.base_btc_vol_annual)
    elif btc_hist is not None and len(btc_hist) > 60:
        rr = np.log(btc_hist.astype(float).dropna()).diff().dropna().to_numpy()[-500:]
        sig_a = float(np.std(rr, ddof=1) * math.sqrt(TRADING_DAYS))
    else:
        sig_a = float(np.mean(regime.sigmas_daily) * math.sqrt(TRADING_DAYS))
    sig_d = sig_a / math.sqrt(TRADING_DAYS)

    vm_arr = np.asarray(params.vol_mult, np.float32)
    dm_arr = np.asarray(params.drift_mult, np.float32)
    jl_arr = np.asarray(params.jump_lambda, np.float32)
    jcomp_regime = jl_arr * dt * (math.exp(params.jump_mean + 0.5 * params.jump_std**2) - 1.0)

    # --- vol band scaling so [low, high] ~ 10/90 pct at horizon -----------
    vscale = 1.0
    if cfg.btc_target_expected and cfg.btc_target_low and cfg.btc_target_high:
        spread = math.log(cfg.btc_target_high / cfg.btc_target_low)
        base_sd_T = sig_d * math.sqrt(H) * float(np.dot(regime_frac, vm_arr))
        vscale = float(np.clip((spread / (2 * 1.2816)) / max(base_sd_T, 1e-9), 0.4, 2.5))

    # per-regime -> per-step tables (index R exactly once each) ------------
    sv = (sig_d * vscale) * vm_arr[R]  # (N,H) sigma*vm
    dm_R = dm_arr[R]  # (N,H) drift mult
    jcomp_R = jcomp_regime[R]
    lam_R = jl_arr[R] * dt
    theta_R = np.asarray(params.theta_by_regime, np.float32)[R]

    # --- BTC drift: scenario-calibrated (drag-neutral) or regime-native ---
    if cfg.btc_target_expected:
        want_log = math.log(cfg.btc_target_expected / s0_btc)
        # drag added back per step:  1/2(sv)^2 (Ito) + comp (jump mean comp)
        #   - lambda dt * jump_mean  (jumps are mean-negative -> median decays)
        drag_i = (0.5 * sv * sv + jcomp_R - lam_R * params.jump_mean).sum(axis=1)
        g_i = ((want_log + drag_i) / np.maximum(dm_R.sum(axis=1), 1e-6)).astype(np.float32)
        step_drift = g_i[:, None] * dm_R
    else:
        step_drift = np.asarray(regime.means_daily, np.float32)[R] * dm_R / float(np.mean(dm_arr))

    # --- diffusion (antithetic Gaussian) + Merton jumps -----------------
    zc = _antithetic_normal(rng, N, H, cfg.antithetic)  # BTC driver
    zm = _antithetic_normal(rng, N, H, cfg.antithetic)  # mNAV idiosyncratic
    rho = float(np.clip(params.corr_btc_mnav, -0.95, 0.95))
    z_mnav = rho * zc + math.sqrt(1 - rho**2) * zm

    jsize = (
        (rng.random((N, H)) < lam_R)
        * (params.jump_mean + params.jump_std * rng.standard_normal((N, H)))
    ).astype(np.float32)

    btc_steps = step_drift - 0.5 * sv * sv - jcomp_R + sv * zc + jsize
    btc = np.empty((N, H + 1), np.float32)
    btc[:, 0] = s0_btc
    btc[:, 1:] = s0_btc * np.exp(np.cumsum(btc_steps, axis=1, dtype=np.float32))

    # --- mNAV: bounded Ornstein-Uhlenbeck toward theta(R_t)  (vectorised) ---
    from scipy.signal import lfilter

    theta = theta_R
    a = 1.0 - params.kappa * dt  # AR(1) coefficient
    sm_d = params.sigma_m * math.sqrt(dt)
    b = (params.kappa * dt) * theta + sm_d * z_mnav  # (N,H) forcing
    zi = np.full((N, 1), a * m0_nav, np.float32)
    m_body, _ = lfilter([1.0], [1.0, -a], b, axis=1, zi=zi)
    m = np.empty((N, H + 1), np.float32)
    m[:, 0] = m0_nav
    m[:, 1:] = np.clip(m_body, params.m_floor, params.m_ceil)

    # --- reflexive accretion: BPS / DPS via cumulative products ----------
    bps0, dps0 = fund.btc_per_share(), fund.debt_per_share()
    g_yield = float(np.clip(cfg.accretion_yield_annual, 0.0, 0.25)) * dt
    fee = fund.issuance_fee
    mt = m[:, 1:]
    eff = np.clip((mt * (1.0 - fee) - 1.0) / np.maximum(mt, 1e-6), 0.0, 1.0)
    issue_rate = g_yield * (mt > 1.0)
    bps_path = np.empty((N, H + 1), np.float32)
    dps_path = np.empty((N, H + 1), np.float32)
    bps_path[:, 0] = bps0
    dps_path[:, 0] = dps0
    bps_path[:, 1:] = bps0 * np.cumprod(1.0 + g_yield * eff, axis=1, dtype=np.float32)
    dps_path[:, 1:] = dps0 / np.cumprod(1.0 + issue_rate, axis=1, dtype=np.float32)

    # --- structural MSTR price -----------------------------------------
    mstr = ((bps_path * btc - dps_path) * m).astype(np.float32)
    mstr = np.maximum(mstr, 0.01)
    s0_struct = fund.structural_price(s0_btc, m0_nav)
    # anchor the cone at today's market price if given (keeps hitting-time
    # probabilities consistent with the user's dollar target); the
    # structural-vs-market gap is reported separately as `s0_structural`.
    anchor = float(cfg.s0_mstr_market) if cfg.s0_mstr_market else s0_struct
    mstr *= np.float32(anchor / max(s0_struct, 1e-6))
    mstr[:, 0] = anchor

    res = _summarise(mstr, cfg, anchor, target_price)
    res.s0_structural = float(s0_struct)
    res.btc_terminal = btc[:, -1].astype(float)
    res.mnav_terminal = m[:, -1].astype(float)
    res.bps_terminal = bps_path[:, -1].astype(float)
    res.bps_start = float(bps0)
    res.regime_now = regime.label_of_now
    res.regime_mix = {REGIME_NAMES[k]: float(regime_frac[k]) for k in range(3)}
    res.mu_btc_annual = float(np.log(np.median(btc[:, -1]) / s0_btc) / (H / TRADING_DAYS))
    res.sigma_btc_annual = float(sig_a * np.dot(regime_frac, params.vol_mult))
    return res


# ===========================================================================
# helpers
# ===========================================================================
def _antithetic_normal(rng: np.random.Generator, n: int, h: int, anti: bool) -> np.ndarray:
    if anti and n % 2 == 0:
        half = rng.standard_normal((n // 2, h))
        return np.vstack([half, -half]).astype(np.float32)
    return rng.standard_normal((n, h)).astype(np.float32)


_PCTL = {
    "p01": 1,
    "p05": 5,
    "p10": 10,
    "p25": 25,
    "p50": 50,
    "p75": 75,
    "p90": 90,
    "p95": 95,
    "p99": 99,
}


def _summarise(
    paths: np.ndarray, cfg: HybridConfig, s0: float, target: float | None
) -> HybridResult:
    from scipy import stats as _st

    term = paths[:, -1].astype(float)
    qs = list(_PCTL.values())
    stacked = np.percentile(paths, qs, axis=0)  # one partition, all bands
    pct = {k: stacked[i] for i, k in enumerate(_PCTL)}
    var5, var1 = float(pct["p05"][-1]), float(pct["p01"][-1])
    res = HybridResult(
        horizon=cfg.horizon,
        n_paths=paths.shape[0],
        last_price=float(s0),
        paths=paths,
        percentiles=pct,
        terminal_prices=term,
        expected_price=float(term.mean()),
        median_price=float(np.median(term)),
        prob_up=float((term > s0).mean()),
        var_5_price=var5,
        es_5_price=float(term[term <= var5].mean() if (term <= var5).any() else var5),
        var_1_price=var1,
        es_1_price=float(term[term <= var1].mean() if (term <= var1).any() else var1),
        terminal_skew=float(_st.skew(term)),
        terminal_kurt=float(_st.kurtosis(term)),
        max_drawdown_p50=float(np.median(_path_max_drawdown(paths))),
        btc_terminal=np.array([]),
        mnav_terminal=np.array([]),
        bps_terminal=np.array([]),
        bps_start=0.0,
        regime_now="",
        regime_mix={},
    )
    if target:
        _hitting(res, paths, s0, float(target))
    return res


def _path_max_drawdown(paths: np.ndarray) -> np.ndarray:
    run_max = np.maximum.accumulate(paths, axis=1)
    return (paths / run_max - 1.0).min(axis=1)


def _hitting(res: HybridResult, paths: np.ndarray, s0: float, target: float):
    res.target_price = target
    if target >= s0:
        hit = paths.max(axis=1) >= target
        res.p_touch_up = float(hit.mean())
        res.p_close_above = float((paths[:, -1] >= target).mean())
    else:
        hit = paths.min(axis=1) <= target
        res.p_touch_down = float(hit.mean())
        res.p_close_above = float((paths[:, -1] >= target).mean())
    if hit.any():
        cross = (paths >= target) if target >= s0 else (paths <= target)
        day = np.argmax(cross, axis=1).astype(float)
        res.expected_hit_day = float(day[hit].mean())


# ===========================================================================
# convenience: one-call scenario from the app
# ===========================================================================
def run_scenario(
    btc_close: pd.Series,
    fundamentals: StrategyFundamentals,
    m0_nav: float,
    horizon: int,
    n_paths: int = 50_000,
    btc_expected: float | None = None,
    btc_low: float | None = None,
    btc_high: float | None = None,
    mnav_low: float = 0.9,
    mnav_expected: float = 1.30,
    mnav_high: float = 2.2,
    accretion_yield: float = 0.08,
    expansion_prior: float = 0.0,
    target_price: float | None = None,
    s0_mstr_market: float | None = None,
    seed: int = 0,
) -> HybridResult:
    s0_btc = float(btc_close.astype(float).dropna().iloc[-1])
    try:
        rm = RegimeModel.fit(btc_close)
    except Exception:  # noqa: BLE001
        rm = RegimeModel.heuristic(btc_close)
    cfg = HybridConfig(
        horizon=int(horizon),
        n_paths=int(n_paths),
        seed=seed,
        btc_target_expected=btc_expected,
        btc_target_low=btc_low,
        btc_target_high=btc_high,
        mnav_low=mnav_low,
        mnav_expected=mnav_expected,
        mnav_high=mnav_high,
        accretion_yield_annual=accretion_yield,
        expansion_prior=expansion_prior,
        s0_mstr_market=s0_mstr_market,
    )
    return simulate_hybrid(
        fundamentals, rm, cfg, s0_btc, m0_nav, target_price=target_price, btc_hist=btc_close
    )


# ===========================================================================
# continuous-support auto-binding  (fixes P=0 slider-clip bugs)
# ===========================================================================
def continuous_support(
    s0: float, target: float, lo_mult: float = 0.6, hi_mult: float = 1.5
) -> tuple[float, float]:
    a = lo_mult * min(s0, target)
    b = hi_mult * max(s0, target)
    return float(a), float(b)


# ===========================================================================
# self-test
# ===========================================================================
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("CARN-X  Hybrid Neural-Structural Engine  --  self test")
    print("=" * 70)

    f = DEFAULT_FUNDAMENTALS
    px = f.structural_price(150_000.0, 1.30)
    print(
        f"\n[structural]  BTC $150k · mNAV 1.30x · BPS {f.btc_per_share():.5f} "
        f"· DPS ${f.debt_per_share():,.2f}  ->  MSTR ${px:,.2f}   "
        f"(target $350-420: {'OK' if 350 <= px <= 420 else 'OUT'})"
    )

    try:
        from btc_cycle import load_btc_history

        btc = load_btc_history()
    except Exception as e:  # noqa: BLE001
        print(f"(no BTC history: {e}) -- synthesising")
        rng = np.random.default_rng(0)
        btc = pd.Series(
            30000 * np.exp(np.cumsum(rng.normal(0.0015, 0.035, 1400))),
            index=pd.bdate_range("2021-01-01", periods=1400),
        )

    rm = RegimeModel.fit(btc)
    print(
        f"\n[regime]  now = {rm.label_of_now}  ·  daily mu by regime "
        f"{np.round(rm.means_daily * 1e3, 2)} (x1e-3)  ·  "
        f"stationary {np.round(rm.stationary, 2)}"
    )

    t0 = time.perf_counter()
    res = run_scenario(
        btc,
        f,
        m0_nav=1.25,
        horizon=252,
        n_paths=50_000,
        btc_expected=150_000,
        btc_low=95_000,
        btc_high=230_000,
        mnav_expected=1.30,
        accretion_yield=0.08,
        target_price=500.0,
    )
    dt = (time.perf_counter() - t0) * 1e3
    print(f"\n[MC]  50k x 252 in {dt:.0f} ms")
    print(
        f"      MSTR  median ${res.median_price:,.0f}  ·  mean ${res.expected_price:,.0f}"
        f"  ·  80% band ${res.percentiles['p10'][-1]:,.0f}-${res.percentiles['p90'][-1]:,.0f}"
    )
    print(f"      BTC   median ${np.median(res.btc_terminal):,.0f}  (target 150k)")
    print(f"      mNAV  median {np.median(res.mnav_terminal):.2f}")
    print(
        f"      BPS   {res.bps_start:.5f} -> {np.median(res.bps_terminal):.5f}  "
        f"(+{np.median(res.bps_terminal) / res.bps_start - 1:.1%} sats/share)"
    )
    print(f"      regime mix { ({k: round(v, 2) for k, v in res.regime_mix.items()}) }")
    print(f"      P(touch $500) {res.p_touch_up:.1%}  ·  VaR5 ${res.var_5_price:,.0f}")
    print("\nOK")
