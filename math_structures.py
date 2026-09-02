"""
CARN-X  --  Mathematical Structures  (analytic companion, not the neural path)
=============================================================================
"Glue the chart to arithmetic structures and mathematical tools."

Every function here takes the *same* MSTR history the rest of CARN-X uses
(``close: pd.Series`` of daily closes) and re-expresses its statistics through a
named branch of mathematics, then -- where a claim is testable -- checks it
against the empirical record, Monte-Carlo, or a closed form.

What is real, and labelled as such
----------------------------------
  * **Newton's binomial / Pascal's triangle**  ->  the Cox-Ross-Rubinstein
    binomial lattice. Node probabilities are literally ``C(n,k) p^k q^(n-k)``;
    as steps -> inf the tree converges to Black-Scholes and the constants
    ``e`` (continuous compounding) and ``pi`` (Gaussian normaliser) appear.
  * **Combinatorics (Catalan / ballot / reflection principle)**  ->  the
    probability a price path stays above a drawdown floor. Dyck-path counts.
  * **Linear algebra (Koopman / DMD)**  ->  the eigen-decomposition of the
    best linear operator on a delay-embedding of the returns: decaying,
    oscillating and growing modes on the complex plane.
  * **PDEs (Kolmogorov forward / Fokker-Planck)**  ->  the predictive return
    density evolved forward in time; the forecast distribution solves a PDE.
  * **ODEs / stochastic calculus**  ->  a Merton jump-diffusion fitted by MLE;
    its deterministic drift skeleton vs. the noise.
  * **Combinatorial path statistics**  ->  Wald-Wolfowitz runs test, Levy
    arcsine law, longest-run and up-day-count tests on the real series.

What is convention, and labelled as such
----------------------------------------
  * **Fibonacci / golden-ratio retracement levels** -- a charting convention.
    ``fibonacci_pivot_test`` runs the honest experiment: do swing reversals
    cluster near the 0.382 / 0.5 / 0.618 levels more than a shuffled null?
  * ``math_constants_in_finance`` states plainly where ``e``, ``pi``, ``phi``
    and the primes genuinely occur -- and where they do not.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import svd
from scipy.signal import argrelextrema
from scipy.special import comb, gammaln

TRADING_DAYS_YEAR = 252
PHI = (1.0 + math.sqrt(5.0)) / 2.0
DEFAULT_LOOKBACK = 504


# ===========================================================================
# helpers
# ===========================================================================
def _logret(close: pd.Series, lookback: int | None = None) -> np.ndarray:
    s = close.astype(float).dropna()
    if lookback:
        s = s.tail(lookback + 1)
    return np.log(s).diff().dropna().to_numpy()


def _annualised(r: np.ndarray) -> tuple[float, float]:
    """(mu, sigma) per year from daily log-returns."""
    return float(np.mean(r) * TRADING_DAYS_YEAR), float(
        np.std(r, ddof=1) * math.sqrt(TRADING_DAYS_YEAR)
    )


# ===========================================================================
# 1.  NEWTON'S BINOMIAL  /  PASCAL'S TRIANGLE  ->  CRR binomial lattice
# ===========================================================================
@dataclass
class BinomialLattice:
    steps: int
    horizon_days: int
    s0: float
    u: float
    d: float
    p_riskneutral: float
    p_realworld: float
    dt_years: float
    node_prices: np.ndarray  # (steps+1, steps+1) upper-triangular
    node_probs: np.ndarray  # Pascal row weights * p^k q^(n-k)
    terminal_prices: np.ndarray  # (steps+1,)
    terminal_probs_rn: np.ndarray
    terminal_probs_rw: np.ndarray
    expected_terminal_rn: float
    expected_terminal_rw: float
    bs_lognormal_mean: float  # continuous-limit check (uses e)
    empirical_quantile_overlay: dict[str, float] = field(default_factory=dict)


def crr_lattice(
    close: pd.Series,
    horizon_days: int = 20,
    steps: int = 20,
    r_annual: float = 0.045,
    lookback: int = DEFAULT_LOOKBACK,
) -> BinomialLattice:
    """Cox-Ross-Rubinstein recombining tree calibrated to MSTR's realised vol.

    The probability of landing at terminal node ``k`` after ``n`` steps is
    ``C(n,k) p^k (1-p)^(n-k)`` -- row ``n`` of Pascal's triangle, weighted.
    """
    ret = _logret(close, lookback)
    mu_a, sig_a = _annualised(ret)
    s0 = float(close.astype(float).dropna().iloc[-1])

    T = horizon_days / TRADING_DAYS_YEAR
    dt = T / steps
    u = math.exp(sig_a * math.sqrt(dt))
    d = 1.0 / u
    p_rn = (math.exp(r_annual * dt) - d) / (u - d)
    p_rn = float(np.clip(p_rn, 1e-6, 1 - 1e-6))
    p_rw = (math.exp(mu_a * dt) - d) / (u - d)
    p_rw = float(np.clip(p_rw, 1e-6, 1 - 1e-6))

    n = steps
    prices = np.zeros((n + 1, n + 1))
    probs = np.zeros((n + 1, n + 1))
    for lvl in range(n + 1):
        ks = np.arange(lvl + 1)
        prices[lvl, : lvl + 1] = s0 * u**ks * d ** (lvl - ks)
        log_c = gammaln(lvl + 1) - gammaln(ks + 1) - gammaln(lvl - ks + 1)
        probs[lvl, : lvl + 1] = np.exp(log_c + ks * math.log(p_rn) + (lvl - ks) * math.log1p(-p_rn))

    ks = np.arange(n + 1)
    term_prices = s0 * u**ks * d ** (n - ks)
    log_c = gammaln(n + 1) - gammaln(ks + 1) - gammaln(n - ks + 1)
    term_rn = np.exp(log_c + ks * math.log(p_rn) + (n - ks) * math.log1p(-p_rn))
    term_rw = np.exp(log_c + ks * math.log(p_rw) + (n - ks) * math.log1p(-p_rw))

    # continuous-compounding limit (this is where e lives)
    bs_mean = s0 * math.exp(r_annual * T)

    # empirical h-day return quantiles for overlay
    if len(ret) > horizon_days + 20:
        hret = pd.Series(ret).rolling(horizon_days).sum().dropna().to_numpy()
        emp = {
            f"q{int(q * 100)}": float(s0 * math.exp(np.quantile(hret, q)))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        }
    else:
        emp = {}

    return BinomialLattice(
        steps=n,
        horizon_days=horizon_days,
        s0=s0,
        u=u,
        d=d,
        p_riskneutral=p_rn,
        p_realworld=p_rw,
        dt_years=dt,
        node_prices=prices,
        node_probs=probs,
        terminal_prices=term_prices,
        terminal_probs_rn=term_rn / term_rn.sum(),
        terminal_probs_rw=term_rw / term_rw.sum(),
        expected_terminal_rn=float(np.dot(term_prices, term_rn)),
        expected_terminal_rw=float(np.dot(term_prices, term_rw)),
        bs_lognormal_mean=bs_mean,
        empirical_quantile_overlay=emp,
    )


@dataclass
class BinomialNormalConvergence:
    n_grid: np.ndarray
    total_variation: np.ndarray  # ||Binomial_n - Normal|| vs n
    demoivre_note: str
    pi_appears_as: str
    e_appears_as: str


def binomial_to_normal_convergence(
    p: float = 0.52, n_grid: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320)
) -> BinomialNormalConvergence:
    """de Moivre-Laplace: the up-day count Binomial(n, p) -> Normal(np, np(1-p)).
    The Gaussian limit is where sqrt(2*pi) enters finance."""
    tv = []
    for n in n_grid:
        k = np.arange(n + 1)
        pmf = stats.binom.pmf(k, n, p)
        mu, sd = n * p, math.sqrt(n * p * (1 - p))
        approx = stats.norm.pdf(k, mu, sd)
        tv.append(0.5 * float(np.sum(np.abs(pmf - approx))))
    return BinomialNormalConvergence(
        n_grid=np.array(n_grid),
        total_variation=np.array(tv),
        demoivre_note="Binomial(n,p) -> Normal(np, np(1-p)) as n grows; the "
        "CRR tree inherits this, converging to Black-Scholes.",
        pi_appears_as="the 1/sqrt(2*pi*sigma^2) normaliser of the limiting "
        "Gaussian return density.",
        e_appears_as="the n->inf limit (1 + r*T/n)^n -> e^{rT}: continuous "
        "compounding, the tree's growth factor.",
    )


def pascal_triangle(rows: int = 16) -> np.ndarray:
    """Pascal's triangle as a lower-triangular matrix of binomial coefficients."""
    tri = np.zeros((rows, rows), dtype=float)
    for n in range(rows):
        tri[n, : n + 1] = comb(n, np.arange(n + 1), exact=False)
    return tri


# ===========================================================================
# 2.  COMBINATORICS  ->  Catalan numbers / ballot problem / drawdown survival
# ===========================================================================
def catalan(n: int) -> int:
    return int(round(comb(2 * n, n, exact=False) / (n + 1)))


@dataclass
class DrawdownSurvival:
    horizon_days: int
    floor_pct: float
    p_survive_combinatorial: float  # driftless symmetric walk, reflection principle
    p_survive_drifted: float  # with fitted per-step drift/vol, reflection
    p_survive_montecarlo: float
    p_survive_empirical: float | None
    catalan_number: int
    dyck_paths_total: float
    note: str


def drawdown_survival(
    close: pd.Series,
    floor_pct: float = 0.20,
    horizon_days: int = 20,
    lookback: int = DEFAULT_LOOKBACK,
    n_mc: int = 40_000,
    seed: int = 0,
) -> DrawdownSurvival:
    """P(the path never falls more than `floor_pct` below its start within h days).

    Combinatorial core: a +/-1 lattice walk that never crosses -b is counted by
    the reflection principle (Catalan numbers when b -> 0 and h even).
    """
    ret = _logret(close, lookback)
    mu, sig = float(np.mean(ret)), float(np.std(ret, ddof=1))
    b_lr = -math.log1p(-floor_pct)  # log-return distance to floor (>0)
    h = horizon_days

    # --- combinatorial: symmetric +/-1 lattice walk, step size = sigma ---
    # reflection principle (exact):  P(min_{j<=h} S_j <= -m)
    #   = P(S_h <= -m) + P(S_h <= -(m+1)),  with S_h = 2*Binom(h, 1/2) - h.
    m = max(1, int(round(b_lr / sig)))
    p_hit = float(stats.binom.cdf((h - m) / 2, h, 0.5) + stats.binom.cdf((h - m - 1) / 2, h, 0.5))
    p_comb = float(np.clip(1.0 - p_hit, 0.0, 1.0))

    # --- drifted reflection (Gaussian approximation, continuous) ---
    #  P(min_{t<=T} (mu t + sig W_t) > -b)
    T = h
    p_drift = float(_drifted_survival(mu, sig, b_lr, T))

    # --- Monte-Carlo with the empirical return distribution (block bootstrap) ---
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ret), size=(n_mc, h))
    paths = np.cumsum(ret[idx], axis=1)
    p_mc = float(np.mean(paths.min(axis=1) > -b_lr))

    # --- empirical: how often historically ---
    lr = _logret(close)
    if len(lr) > h + 5:
        wins = np.lib.stride_tricks.sliding_window_view(lr, h)
        cum = np.cumsum(wins, axis=1)
        p_emp = float(np.mean(cum.min(axis=1) > -b_lr))
    else:
        p_emp = None

    n_even = h // 2
    return DrawdownSurvival(
        horizon_days=h,
        floor_pct=floor_pct,
        p_survive_combinatorial=p_comb,
        p_survive_drifted=p_drift,
        p_survive_montecarlo=p_mc,
        p_survive_empirical=p_emp,
        catalan_number=catalan(n_even),
        dyck_paths_total=float(comb(h, h // 2, exact=False)),
        note=f"floor is ~{m} downward lattice steps of size sigma={sig:.4f}; "
        f"C_{n_even} = {catalan(n_even):,} counts the strictly-non-negative "
        f"Dyck paths of length {2 * n_even}.",
    )


def _drifted_survival(mu: float, sig: float, b: float, T: float) -> float:
    """P(inf_{t<=T} (mu t + sig W_t) > -b) for Brownian motion with drift."""
    if b <= 0:
        return 0.0
    s = sig * math.sqrt(T)
    a1 = (b + mu * T) / s
    a2 = (-b + mu * T) / s
    return float(stats.norm.cdf(a1) - math.exp(-2 * mu * b / sig**2) * stats.norm.cdf(a2))


# ===========================================================================
# 3.  LINEAR ALGEBRA  ->  Koopman / Dynamic Mode Decomposition
# ===========================================================================
@dataclass
class DMDSpectrum:
    rank: int
    delay: int
    eigenvalues: np.ndarray  # complex, discrete-time
    growth_rate_per_day: np.ndarray  # Re(log lambda)
    period_days: np.ndarray  # 2*pi / |Im(log lambda)|  (inf if none)
    mode_amplitude: np.ndarray
    dominant_period_days: float
    spectral_radius: float  # max |lambda|  -> >1 explosive, <1 mean-reverting
    reconstruction_r2: float
    forecast: np.ndarray  # h-step-ahead cumulative-return forecast
    note: str


def dmd_spectrum(
    close: pd.Series,
    rank: int = 8,
    delay: int = 20,
    lookback: int = DEFAULT_LOOKBACK,
    horizon_days: int = 20,
) -> DMDSpectrum:
    """Hankel-DMD: the best linear operator A with x_{t+1} ~ A x_t on a
    time-delay embedding of the (de-trended, EWMA-smoothed) log-price. Its
    eigenvalues are the Koopman modes -- the chart decomposed into
    linear-algebra primitives: a slow trend mode plus oscillatory cycles."""
    s = close.astype(float).dropna()
    if lookback:
        s = s.tail(lookback)
    logp = np.log(s.to_numpy())
    # remove the linear trend, keep the cyclical/mean-reverting part, light EWMA
    t = np.arange(len(logp))
    trend = np.polyval(np.polyfit(t, logp, 1), t)
    resid = pd.Series(logp - trend).ewm(span=3).mean().to_numpy()
    resid = resid - resid.mean()

    L = delay
    H = np.lib.stride_tricks.sliding_window_view(resid, L).T  # (L, N)
    X, Y = H[:, :-1], H[:, 1:]

    U, s, Vt = svd(X, full_matrices=False)
    rank = int(min(rank, np.sum(s > 1e-10)))
    Ur, sr, Vr = U[:, :rank], s[:rank], Vt[:rank].conj().T
    Atil = Ur.conj().T @ Y @ Vr @ np.diag(1.0 / sr)
    lam, W = np.linalg.eig(Atil)

    order = np.argsort(-np.abs(lam))
    lam, W = lam[order], W[:, order]
    Phi = Y @ Vr @ np.diag(1.0 / sr) @ W  # DMD modes

    # optimal constant amplitudes: least squares over the whole window
    #   min_b || X - Phi diag(b) [lam^0 ... lam^{T-1}] ||
    tcol = np.arange(X.shape[1])
    vander = lam[:, None] ** tcol[None, :]  # (rank, T)
    K = np.vstack([Phi * vander[:, j] for j in range(X.shape[1])])  # (L*T, rank)
    b = np.linalg.lstsq(K, X.T.reshape(-1), rcond=None)[0]

    log_lam = np.log(lam.astype(complex))
    growth = log_lam.real
    with np.errstate(divide="ignore"):
        period = np.where(np.abs(log_lam.imag) > 1e-9, 2 * np.pi / np.abs(log_lam.imag), np.inf)

    # reconstruction quality on the training window
    tt = np.arange(X.shape[1])
    recon = (Phi @ (b[:, None] * lam[:, None] ** tt[None, :])).real
    ss_res = float(np.sum((X - recon) ** 2))
    ss_tot = float(np.sum((X - X.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # forecast: advance the last state h steps; the top delay coordinate is the
    # de-trended log-price residual -- add the fitted trend slope back to get a
    # cumulative-log-return forecast from today.
    slope = float(np.polyfit(t, logp, 1)[0])
    t0 = X.shape[1] - 1
    resid_now = float(resid[-1])
    fc = []
    for hh in range(1, horizon_days + 1):
        state = (Phi @ (b * lam ** (t0 + hh))).real
        fc.append(float(state[0]) - resid_now + slope * hh)
    fc_cum = np.array(fc)

    osc = period[np.isfinite(period)]
    dom = float(osc[0]) if len(osc) else float("inf")
    return DMDSpectrum(
        rank=rank,
        delay=L,
        eigenvalues=lam,
        growth_rate_per_day=growth,
        period_days=period,
        mode_amplitude=np.abs(b),
        dominant_period_days=dom,
        spectral_radius=float(np.max(np.abs(lam))),
        reconstruction_r2=float(r2),
        forecast=fc_cum,
        note="|lambda|<1: the mode decays (mean-reversion); |lambda|~1: "
        "persistent; |lambda|>1: locally explosive. arg(lambda) gives an "
        "oscillation period in trading days.",
    )


# ===========================================================================
# 4.  PDEs  ->  Kolmogorov forward / Fokker-Planck density evolution
# ===========================================================================
@dataclass
class FokkerPlanckSolution:
    x_grid: np.ndarray  # log-return coordinate
    t_grid: np.ndarray  # days
    density: np.ndarray  # (n_t, n_x)
    price_grid: np.ndarray  # s0 * exp(x)
    terminal_mean_logret: float
    terminal_std_logret: float
    mass_drift: float  # |1 - integral(p dx)| at the last step
    analytic_gaussian_check: float  # max |numeric - N(mu t, sig^2 t)| for const coeffs
    local_vol_used: bool
    note: str


def fokker_planck_density(
    close: pd.Series,
    horizon_days: int = 20,
    lookback: int = DEFAULT_LOOKBACK,
    n_x: int = 401,
    local_vol: bool = True,
) -> FokkerPlanckSolution:
    """Evolve the forward log-return density p(x, t) under

        dp/dt = -d/dx[ a(x) p ] + 1/2 d^2/dx^2[ b(x)^2 p ]

    with drift a and diffusion b estimated from the data (optionally
    state-dependent via a Nadaraya-Watson fit). The CARN-X forecast
    distribution is one slice of this PDE solution."""
    r = _logret(close, lookback)
    mu, sig = float(np.mean(r)), float(np.std(r, ddof=1))
    s0 = float(close.astype(float).dropna().iloc[-1])

    span = 8.0 * sig * math.sqrt(horizon_days)
    x = np.linspace(-span, span, n_x)
    dx = x[1] - x[0]

    if local_vol and len(r) > 250:
        # Nadaraya-Watson local drift / variance of the next-day return as a
        # function of the current return level, shrunk toward the global
        # estimate by the local effective sample size (James-Stein style).
        r_lvl = np.cumsum(r) - np.cumsum(r).mean()
        bw = max(2.0 * sig, 0.6 * sig * math.sqrt(horizon_days))
        prior_n = 120.0  # shrinkage strength
        a_x = np.full(n_x, mu)
        b2_x = np.full(n_x, sig**2)
        in_range = (x >= r_lvl.min()) & (x <= r_lvl.max())
        for i in np.where(in_range)[0]:
            w = np.exp(-0.5 * ((r_lvl[:-1] - x[i]) / bw) ** 2)
            n_eff = w.sum() ** 2 / (np.sum(w**2) + 1e-12)
            k = n_eff / (n_eff + prior_n)
            wsum = w.sum() + 1e-12
            a_loc = float(np.dot(w, r[1:]) / wsum)
            b_loc = float(np.dot(w, (r[1:] - a_loc) ** 2) / wsum)
            a_x[i] = k * a_loc + (1 - k) * mu
            b2_x[i] = k * b_loc + (1 - k) * sig**2
        b2_x = np.clip(b2_x, 0.6 * sig**2, 1.8 * sig**2)
        used = True
    else:
        a_x = np.full(n_x, mu)
        b2_x = np.full(n_x, sig**2)
        used = False

    # implicit (backward-Euler) tridiagonal solve of the conservative form
    #   p_t = -d/dx(a p) + 1/2 d^2/dx^2(b^2 p)
    # central differences; unconditionally stable, so dt = 1 day per snapshot.
    sub = 4  # sub-steps per day (accuracy)
    dtau = 1.0 / sub
    alpha = a_x / (2.0 * dx)  # advection coefficient
    beta = 0.5 * b2_x / dx**2  # diffusion coefficient
    lower = -dtau * (beta[1:] + alpha[1:])  # sub-diagonal
    upper = -dtau * (beta[:-1] - alpha[:-1])  # super-diagonal
    diag = 1.0 + 2.0 * dtau * beta
    M = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
    M[0, :] = 0.0
    M[0, 0] = 1.0  # absorbing far boundaries
    M[-1, :] = 0.0
    M[-1, -1] = 1.0
    Minv = np.linalg.inv(M)

    sigma0 = max(0.12 * sig, 1.5 * dx)
    p = np.exp(-0.5 * (x / sigma0) ** 2)
    p /= p.sum() * dx
    snaps = [p.copy()]
    for _ in range(horizon_days):
        for _ in range(sub):
            p = Minv @ p
            p = np.clip(p, 0.0, None)
        snaps.append(p.copy())

    dens = np.array(snaps)
    dens /= dens.sum(axis=1, keepdims=True) * dx + 1e-12
    pT = dens[-1]
    m1 = float(np.dot(pT, x) * dx)
    m2 = float(np.dot(pT, (x - m1) ** 2) * dx)

    # constant-coefficient sanity check vs the exact Gaussian
    gauss = stats.norm.pdf(x, mu * horizon_days, sig * math.sqrt(horizon_days))
    if not used:
        chk = float(np.max(np.abs(pT - gauss)))
    else:
        cc = fokker_planck_density(close, horizon_days, lookback, n_x, local_vol=False)
        chk = cc.analytic_gaussian_check

    return FokkerPlanckSolution(
        x_grid=x,
        t_grid=np.arange(horizon_days + 1),
        density=dens,
        price_grid=s0 * np.exp(x),
        terminal_mean_logret=m1,
        terminal_std_logret=math.sqrt(max(m2, 0.0)),
        mass_drift=float(abs(1.0 - dens[-1].sum() * dx)),
        analytic_gaussian_check=chk,
        local_vol_used=used,
        note="a slice at t = h is the model-free forward density of the "
        "h-day log-return; the peak's spread grows like sqrt(t) (diffusion).",
    )


# ===========================================================================
# 5.  ODEs / STOCHASTIC CALCULUS  ->  Merton jump-diffusion, fitted by MLE
# ===========================================================================
@dataclass
class JumpDiffusionFit:
    mu_drift_annual: float
    sigma_diffusion_annual: float
    jump_intensity_annual: float
    jump_mean: float
    jump_std: float
    loglik: float
    loglik_gaussian: float
    lr_stat: float  # likelihood-ratio vs pure Gaussian
    lr_pvalue: float
    jump_days: list[str]
    deterministic_skeleton: str
    note: str


def fit_jump_diffusion(close: pd.Series, lookback: int = 756) -> JumpDiffusionFit:
    """Merton model  dS/S = mu dt + sigma dW + (J-1) dN,  N ~ Poisson(lambda).

    Daily returns are a mixture: (no jump) N(mu_d, sig_d^2) w.p. 1-lambda_d,
    (jump) N(mu_d + mJ, sig_d^2 + sJ^2) w.p. lambda_d. Fitted by ML."""
    idx = close.astype(float).dropna().tail(lookback + 1)
    r = np.log(idx).diff().dropna()
    x = r.to_numpy()
    dates = idx.index[1:]

    sd0 = float(np.std(x, ddof=1))
    m0 = float(np.mean(x))

    # two-Gaussian mixture fitted by EM (robust where a raw optimiser is not):
    #   diffusion day  ~ N(mu0, sig0^2),   weight 1 - lam
    #   jump day       ~ N(mu1, sig1^2),   weight lam,  sig1 >= sig0.
    mad = float(stats.median_abs_deviation(x, scale="normal")) or sd0
    med = float(np.median(x))
    lam_max = 0.06  # <= ~15 jump days / year
    # weak Beta(1, prior_strength) prior on lambda (MAP-EM): without it the
    # mixture inflates lambda to absorb ordinary kurtosis.
    prior_strength = 0.02 * len(x)

    # warm start: seed the jump component from the 3-MAD outliers so EM starts
    # in the "narrow core + fat jump" basin rather than a degenerate split.
    outlier = np.abs(x - med) > 3.0 * mad
    if outlier.sum() >= 3:
        core = ~outlier
        mu0, sig0 = float(x[core].mean()), float(x[core].std(ddof=1))
        mu1, sig1 = float(x[outlier].mean()), float(max(x[outlier].std(ddof=1), 2 * sig0))
        lam = min(lam_max, float(outlier.mean()))
    else:
        lam, mu0, sig0, mu1, sig1 = 0.03, med, mad, med, 4.0 * mad

    for _ in range(300):
        p0 = (1 - lam) * stats.norm.pdf(x, mu0, sig0)
        p1 = lam * stats.norm.pdf(x, mu1, sig1)
        g = p1 / (p0 + p1 + 1e-300)  # posterior P(jump | r)
        sg, s1g = float((1 - g).sum()), float(g.sum())
        lam_new = min(lam_max, s1g / (len(x) + prior_strength))
        mu0 = float(np.dot(1 - g, x) / max(sg, 1e-9))
        sig0 = math.sqrt(max(float(np.dot(1 - g, (x - mu0) ** 2) / max(sg, 1e-9)), 1e-10))
        mu1 = float(np.dot(g, x) / max(s1g, 1e-9))
        sig1 = math.sqrt(max(float(np.dot(g, (x - mu1) ** 2) / max(s1g, 1e-9)), sig0**2 * 1.05))
        if abs(lam_new - lam) < 1e-9:
            lam = lam_new
            break
        lam = lam_new

    mu_d, sig_d = mu0, sig0
    mJ, sJ = mu1 - mu0, math.sqrt(max(sig1**2 - sig0**2, 0.0))
    post = g

    c0 = stats.norm.logpdf(x, mu0, sig0) + math.log1p(-lam)
    c1 = stats.norm.logpdf(x, mu1, sig1) + math.log(lam)
    ll = float(np.sum(np.logaddexp(c0, c1)))
    ll_gauss = float(np.sum(stats.norm.logpdf(x, m0, sd0)))
    lr = 2.0 * (ll - ll_gauss)
    lr_p = float(stats.chi2.sf(max(lr, 0.0), df=3))
    jump_dates = [str(d.date()) for d, pj in zip(dates, post, strict=False) if pj > 0.5]

    ann = TRADING_DAYS_YEAR
    return JumpDiffusionFit(
        mu_drift_annual=float(mu_d * ann),
        sigma_diffusion_annual=float(sig_d * math.sqrt(ann)),
        jump_intensity_annual=float(lam * ann),
        jump_mean=float(mJ),
        jump_std=float(sJ),
        loglik=float(ll),
        loglik_gaussian=ll_gauss,
        lr_stat=float(lr),
        lr_pvalue=lr_p,
        jump_days=jump_dates[-12:],
        deterministic_skeleton=f"dS/S = {mu_d * ann:+.3f} dt   (drop dW and dN)",
        note="the ODE skeleton is exponential growth at the drift rate; the "
        "SDE adds sigma dW (everyday noise) and rare Poisson jumps.",
    )


# ===========================================================================
# 6.  COMBINATORIAL PATH STATISTICS  (runs, arcsine, longest run)
# ===========================================================================
@dataclass
class PathStatistics:
    n_days: int
    up_day_prob: float
    up_day_binom_p: float  # two-sided p vs Binomial(n, 0.5)
    runs_z: float
    runs_pvalue: float  # Wald-Wolfowitz: are up/down runs random?
    longest_up_run: int
    longest_down_run: int
    expected_longest_run: float
    frac_time_at_high: float
    arcsine_pvalue: float  # Levy arcsine law on the fraction above start
    note: str


def path_statistics(close: pd.Series, lookback: int = DEFAULT_LOOKBACK) -> PathStatistics:
    from statsmodels.sandbox.stats.runs import runstest_1samp

    r = _logret(close, lookback)
    signs = (r > 0).astype(int)
    n = len(signs)
    pup = float(signs.mean())

    binom_p = float(stats.binomtest(int(signs.sum()), n, 0.5).pvalue)
    try:
        z, rp = runstest_1samp(signs, correction=False)
    except Exception:
        z, rp = float("nan"), float("nan")

    def _longest(v):
        best = cur = 0
        for b in v:
            cur = cur + 1 if b else 0
            best = max(best, cur)
        return best

    lu = _longest(signs == 1)
    ld = _longest(signs == 0)
    exp_long = (
        math.log(n * max(pup, 1e-6)) / math.log(1.0 / max(pup, 1e-6))
        if 0 < pup < 1
        else float("nan")
    )

    lvl = np.cumsum(r)
    frac_above = float(np.mean(lvl > 0))
    # Levy arcsine: fraction of time a driftless walk spends positive ~ Beta(1/2,1/2)
    arc_p = float(stats.kstest([frac_above], lambda q: stats.beta.cdf(q, 0.5, 0.5)).pvalue)

    return PathStatistics(
        n_days=n,
        up_day_prob=pup,
        up_day_binom_p=binom_p,
        runs_z=float(z),
        runs_pvalue=float(rp),
        longest_up_run=lu,
        longest_down_run=ld,
        expected_longest_run=float(exp_long),
        frac_time_at_high=frac_above,
        arcsine_pvalue=arc_p,
        note="Wald-Wolfowitz tests whether the up/down sequence is more "
        "streaky (or more alternating) than fair coin flips.",
    )


# ===========================================================================
# 7.  FIBONACCI / GOLDEN RATIO  --  a charting convention, tested honestly
# ===========================================================================
FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)


@dataclass
class FibonacciLevels:
    swing_high: float
    swing_low: float
    swing_high_date: str
    swing_low_date: str
    direction: str
    levels: dict[str, float]  # ratio -> price
    current_price: float
    nearest_level: str
    distance_pct: float


def fibonacci_levels(close: pd.Series, swing_lookback: int = 120) -> FibonacciLevels:
    s = close.astype(float).dropna().tail(swing_lookback)
    hi_i, lo_i = s.idxmax(), s.idxmin()
    hi, lo = float(s.max()), float(s.min())
    up = lo_i < hi_i  # low then high -> up-swing
    rng = hi - lo
    if up:
        levels = {f"{q:.3f}": hi - q * rng for q in FIB_RATIOS}
    else:
        levels = {f"{q:.3f}": lo + q * rng for q in FIB_RATIOS}
    cur = float(s.iloc[-1])
    near = min(levels, key=lambda k: abs(levels[k] - cur))
    return FibonacciLevels(
        swing_high=hi,
        swing_low=lo,
        swing_high_date=str(hi_i.date()),
        swing_low_date=str(lo_i.date()),
        direction="up-swing (retrace down)" if up else "down-swing (retrace up)",
        levels=levels,
        current_price=cur,
        nearest_level=near,
        distance_pct=float(abs(levels[near] - cur) / cur),
    )


@dataclass
class FibonacciPivotTest:
    n_pivots: int
    n_near_fib: int
    hit_rate: float
    null_hit_rate: float
    p_value: float  # permutation: does clustering beat chance?
    band_pct: float
    verdict: str


def fibonacci_pivot_test(
    close: pd.Series,
    lookback: int = 1000,
    band_pct: float = 0.01,
    order: int = 5,
    n_perm: int = 2000,
    seed: int = 0,
) -> FibonacciPivotTest:
    """The honest experiment. Take every swing (pivot-to-pivot) move; for the
    following counter-move, measure the retracement fraction; count how many
    land within `band_pct` of a Fibonacci ratio; compare with a permutation
    null that shuffles the ratios' positions."""
    s = close.astype(float).dropna().tail(lookback).to_numpy()
    hi = argrelextrema(s, np.greater_equal, order=order)[0]
    lo = argrelextrema(s, np.less_equal, order=order)[0]
    piv = np.array(sorted(set(hi.tolist()) | set(lo.tolist())))
    piv = piv[np.insert(np.diff(piv) > order, 0, True)]
    if len(piv) < 4:
        return FibonacciPivotTest(len(piv), 0, 0.0, 0.0, 1.0, band_pct, "too few pivots to test")

    retr = []
    for a, b, c in zip(piv[:-2], piv[1:-1], piv[2:], strict=False):
        leg = s[b] - s[a]
        counter = s[b] - s[c]
        if abs(leg) < 1e-9:
            continue
        f = counter / leg
        if 0.0 < f < 1.2:
            retr.append(f)
    retr = np.array(retr)
    if len(retr) < 5:
        return FibonacciPivotTest(
            len(piv), 0, 0.0, 0.0, 1.0, band_pct, "too few clean retracements to test"
        )

    def hits(levels):
        return int(
            np.sum(np.min(np.abs(retr[:, None] - np.array(levels)[None, :]), axis=1) < band_pct)
        )

    obs = hits(FIB_RATIOS)
    rng = np.random.default_rng(seed)
    null = np.array([hits(rng.uniform(0.05, 1.0, len(FIB_RATIOS))) for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return FibonacciPivotTest(
        n_pivots=len(piv),
        n_near_fib=obs,
        hit_rate=obs / len(retr),
        null_hit_rate=float(null.mean() / len(retr)),
        p_value=p,
        band_pct=band_pct,
        verdict=(
            "retracements cluster near Fibonacci levels beyond chance (p<0.05)"
            if p < 0.05
            else "no clustering beyond chance -- consistent with the levels having "
            "no special predictive role"
        ),
    )


# ===========================================================================
# 8.  WHERE THE FAMOUS CONSTANTS ACTUALLY LIVE  (honesty panel)
# ===========================================================================
def math_constants_in_finance() -> list[dict[str, str]]:
    return [
        {
            "constant": "e = 2.71828...",
            "genuine role": "continuous compounding: (1 + rT/n)^n -> e^{rT}. The "
            "limit of the binomial tree as the step count grows.",
            "predictive?": "structural, not a signal",
        },
        {
            "constant": "pi = 3.14159...",
            "genuine role": "the 1/sqrt(2*pi) normaliser of the Gaussian; appears in "
            "the Black-Scholes density and every CLT-based interval.",
            "predictive?": "structural, not a signal",
        },
        {
            "constant": "phi = 1.61803... (golden ratio)",
            "genuine role": "eigenvalue of the Fibonacci recurrence matrix "
            "[[1,1],[1,0]]. In markets it is only a *charting "
            "convention* (0.618 = 1/phi retracement).",
            "predictive?": "no fitted evidence -- see the Fibonacci pivot test",
        },
        {
            "constant": "prime numbers",
            "genuine role": "none in asset pricing. Used in finance only for "
            "hashing / RNG seeding / cryptography.",
            "predictive?": "no",
        },
        {
            "constant": "Catalan numbers C_n",
            "genuine role": "count monotone lattice paths that stay non-negative -- "
            "exactly the combinatorics of a no-drawdown price path.",
            "predictive?": "structural: sets the barrier-survival probability",
        },
    ]


# ===========================================================================
# 9.  STRATEGIC PUZZLES  ->  the decision layer's actual arithmetic
# ===========================================================================
@dataclass
class PuzzleMap:
    egg_drop_probes: int  # minimal trades to localise support to tol
    egg_drop_tol_pct: float
    bezout_alignment: dict[str, Any]  # BTC-cycle vs MSTR dominant period
    ternary_best_horizon: int  # horizon with the strongest drift signal-to-noise
    ternary_horizon_tstat: dict[int, float]
    staircase_reachable_states: int  # recombined lattice levels
    staircase_distinct_paths: int  # Fibonacci-style up/down day-path count
    note: str


def strategic_puzzle_map(
    close: pd.Series, btc_cycle_days: float = 1400.0, lookback: int = DEFAULT_LOOKBACK
) -> PuzzleMap:

    # egg-drop: how many "probe" trades to bracket a support band of width tol?
    tol = 0.02
    span = 0.30  # search a +/-30% band
    n_levels = int(round(span / tol))
    probes = int(math.ceil((-1 + math.sqrt(1 + 8 * n_levels)) / 2))

    # Bezout / hourglass: align the BTC 4-year cycle with MSTR's dominant DMD period
    try:
        dom = dmd_spectrum(close, rank=6, delay=15, lookback=lookback).dominant_period_days
    except Exception:
        dom = float("inf")
    a, b = int(round(btc_cycle_days)), int(round(dom)) if np.isfinite(dom) else 0
    g = math.gcd(a, b) if b else a
    bez = {
        "btc_cycle_days": a,
        "mstr_dominant_period_days": b,
        "gcd_days": g,
        "beat_period_days": (a * b // g) if b else None,
    }

    # ternary / weighing: which horizon carries the strongest drift *signal*
    # relative to its noise? t = mean(h-day ret) / SE over non-overlapping windows.
    lr_full = _logret(close)
    bits = {}
    for h in (1, 5, 20):
        agg = pd.Series(lr_full).groupby(np.arange(len(lr_full)) // h).sum()
        agg = agg[agg.index < len(lr_full) // h]  # drop the ragged tail
        if len(agg) > 3:
            bits[h] = float(abs(agg.mean()) / (agg.std(ddof=1) / math.sqrt(len(agg)) + 1e-12))
        else:
            bits[h] = 0.0
    best_h = max(bits, key=bits.get)  # strongest signal-to-noise

    # staircase / Fibonacci: reachable price states after h days on a +-1 lattice
    h = 20
    reachable = h + 1  # a recombining lattice
    fib_paths = int(round(((PHI ** (h + 1)) - ((-1 / PHI) ** (h + 1))) / math.sqrt(5)))

    return PuzzleMap(
        egg_drop_probes=probes,
        egg_drop_tol_pct=tol,
        bezout_alignment=bez,
        ternary_best_horizon=best_h,
        ternary_horizon_tstat=bits,
        staircase_reachable_states=reachable,
        staircase_distinct_paths=fib_paths,
        note=f"egg-drop: {probes} probe trades bracket a {span:.0%} band to "
        f"{tol:.0%}; staircase: {fib_paths:,} distinct up/down day-paths "
        f"over {h} days collapse onto {reachable} recombined price levels "
        f"(the same 2^h -> h+1 reduction the CRR tree uses).",
    )


# ===========================================================================
# self-test
# ===========================================================================
if __name__ == "__main__":
    import sys

    from data_layer import build_panel, primary_close, DataConfig

    print("=" * 70)
    print("CARN-X  Mathematical Structures  --  self test")
    print("=" * 70)
    try:
        close = primary_close(build_panel(DataConfig()))
    except Exception as e:  # noqa: BLE001
        print(f"(no market panel: {e}) -- synthesising a walk")
        rng = np.random.default_rng(0)
        close = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0.0006, 0.03, 900))),
            index=pd.bdate_range("2022-01-01", periods=900),
        )

    lat = crr_lattice(close, horizon_days=20, steps=20)
    print(
        f"\n[Pascal/CRR]  u={lat.u:.4f} p*={lat.p_riskneutral:.4f}  "
        f"E[S_T]^RN={lat.expected_terminal_rn:,.2f}  vs  S0 e^rT={lat.bs_lognormal_mean:,.2f}  "
        f"(rel {abs(lat.expected_terminal_rn / lat.bs_lognormal_mean - 1):.1e})"
    )

    ds = drawdown_survival(close, floor_pct=0.20, horizon_days=20)
    print(
        f"[Catalan]     P(no -20% in 20d): comb={ds.p_survive_combinatorial:.3f} "
        f"drift={ds.p_survive_drifted:.3f} MC={ds.p_survive_montecarlo:.3f} "
        f"emp={ds.p_survive_empirical}"
    )

    dm = dmd_spectrum(close)
    print(
        f"[LinAlg/DMD]  rho={dm.spectral_radius:.3f}  R2={dm.reconstruction_r2:.2f}  "
        f"dominant period={dm.dominant_period_days:.1f}d  20d forecast={dm.forecast[-1]:+.3f}"
    )

    fp = fokker_planck_density(close, horizon_days=20)
    print(
        f"[PDE/FP]      terminal logret {fp.terminal_mean_logret:+.4f} "
        f"+/- {fp.terminal_std_logret:.4f}  mass drift {fp.mass_drift:.1e}  "
        f"gauss check {fp.analytic_gaussian_check:.1e}"
    )

    jd = fit_jump_diffusion(close)
    print(
        f"[ODE/SDE]     sigma={jd.sigma_diffusion_annual:.2f}/yr  "
        f"lambda={jd.jump_intensity_annual:.1f} jumps/yr  LR p={jd.lr_pvalue:.2e}"
    )

    ps = path_statistics(close)
    print(
        f"[Path stats]  P(up)={ps.up_day_prob:.3f}  runs p={ps.runs_pvalue:.3f}  "
        f"longest up/down {ps.longest_up_run}/{ps.longest_down_run}"
    )

    ft = fibonacci_pivot_test(close)
    print(
        f"[Fibonacci]   {ft.n_near_fib}/{ft.n_pivots} pivots near a level  "
        f"p={ft.p_value:.3f}  -> {ft.verdict}"
    )

    pm = strategic_puzzle_map(close)
    print(
        f"[Puzzles]     egg-drop probes={pm.egg_drop_probes}  "
        f"best horizon={pm.ternary_best_horizon}d  "
        f"reachable states={pm.staircase_reachable_states}"
    )
    print("\nOK")
    sys.exit(0)
