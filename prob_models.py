"""
CARN-X  --  Probability & Combinatorics Lab
===========================================
Two sensors -- a target price and a target date -- drive every classical model
of probability, statistics and combinatorics, each parameterised from the last
~2 years of MSTR data and reported with תוחלת (E) / שונות (Var) / סטיית תקן (SD)
plus the specific question the sensors ask.

Families
    A. discrete      Bernoulli, Binomial, Poisson, Geometric, NegBinomial,
                     Hypergeometric, Discrete-Uniform
    B. continuous    Normal, Lognormal, Student-t, Laplace, Exponential,
                     Continuous-Uniform, Gamma, Beta, Chi-square, Cauchy
    C. extreme value GEV / Gumbel (max move over X days), Pareto tail (Hill)
    D. combinatorics up/down path counts, C(n,k), the 4 classic cases,
                     Catalan / ballot ("never below start"), a named pattern
    E. inference     CLT mean-return CI, t-test H0: drift=0, Chebyshev bound,
                     Beta-Binomial Bayesian P(up)

Plus ``next_day_expected_price`` -- the mean next-day price implied by the
trailing 2-year mean daily log-return.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb, gammaln

TRADING_DAYS_YEAR = 252
LOOKBACK_2Y = 504


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class DistResult:
    key: str
    hebrew: str
    family: str  # discrete | continuous | extreme | combinatorics | inference
    params: dict[str, float]
    mean: float
    variance: float
    std: float
    answers: dict[str, float]  # label -> probability in [0,1]  OR  a count/value
    note: str = ""

    def row(self) -> dict[str, object]:
        return {
            "התפלגות": self.hebrew,
            "משפחה": self.family,
            "תוחלת": self.mean,
            "שונות": self.variance,
            "סטיית תקן": self.std,
        }


# ---------------------------------------------------------------------------
# empirical inputs
# ---------------------------------------------------------------------------


@dataclass
class MarketInputs:
    last_price: float
    r: np.ndarray  # daily log-returns, trailing window
    mu_daily: float
    sigma_daily: float
    skew: float
    kurt: float
    up_day_prob: float
    big_move_prob: float
    move_threshold: float
    hist_low: float
    hist_high: float
    window_days: int
    t_nu: float  # fitted Student-t dof on r
    t_scale: float
    lap_scale: float  # fitted Laplace scale on r
    hill_alpha: float  # Pareto tail exponent of large losses


def _hill_alpha(losses: np.ndarray, k: int = 25) -> float:
    x = np.sort(losses[losses > 0])[::-1]
    if len(x) < k + 2:
        return 3.0
    xk = x[k]
    return float(k / np.sum(np.log(x[:k] / xk))) if xk > 0 else 3.0


def market_inputs(
    close: pd.Series, lookback: int = LOOKBACK_2Y, move_threshold: float = 0.05
) -> MarketInputs:
    close = close.astype(float).dropna()
    w = close.tail(lookback + 1)
    r = np.log(w).diff().dropna().to_numpy()
    try:
        nu, _, sc = stats.t.fit(r, floc=0.0)
        nu = float(np.clip(nu, 2.1, 60))
    except Exception:
        nu, sc = 5.0, float(np.std(r, ddof=1))
    lap_scale = float(np.mean(np.abs(r - np.median(r))))
    return MarketInputs(
        last_price=float(close.iloc[-1]),
        r=r,
        mu_daily=float(np.mean(r)),
        sigma_daily=float(np.std(r, ddof=1)),
        skew=float(stats.skew(r)),
        kurt=float(stats.kurtosis(r)),
        up_day_prob=float(np.mean(r > 0)),
        big_move_prob=float(np.mean(np.abs(r) >= move_threshold)),
        move_threshold=float(move_threshold),
        hist_low=float(w.min()),
        hist_high=float(w.max()),
        window_days=int(len(r)),
        t_nu=nu,
        t_scale=float(sc),
        lap_scale=lap_scale,
        hill_alpha=_hill_alpha(-r[r < 0]),
    )


def next_day_expected_price(mi: MarketInputs) -> dict[str, float]:
    mean_price = mi.last_price * math.exp(mi.mu_daily + 0.5 * mi.sigma_daily**2)
    return {
        "expected_price": mean_price,
        "expected_change_pct": (mean_price / mi.last_price - 1) * 100,
        "band80_low": mi.last_price * math.exp(mi.mu_daily - 1.2816 * mi.sigma_daily),
        "band80_high": mi.last_price * math.exp(mi.mu_daily + 1.2816 * mi.sigma_daily),
        "drift_daily_pct": (math.exp(mi.mu_daily) - 1) * 100,
        "vol_daily_pct": mi.sigma_daily * 100,
        "vol_annual_pct": mi.sigma_daily * math.sqrt(TRADING_DAYS_YEAR) * 100,
        "skew": mi.skew,
        "excess_kurtosis": mi.kurt,
        "based_on_days": mi.window_days,
    }


def _p(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _safe_big(log_value: float) -> float:
    """exp(log_value) but returns +inf instead of raising OverflowError, and
    snaps to the exact integer while the count is still exactly representable.
    Used for combinatorial counts that blow past the float range at long
    horizons (display-only; mean/variance stay nan)."""
    if not math.isfinite(log_value):
        return math.inf if log_value > 0 else 0.0
    if log_value > 709.0:
        return math.inf
    v = math.exp(log_value)
    return float(round(v)) if v < 9.0e15 else v


# ---------------------------------------------------------------------------
# NYSE trading-day count (weekends + US market holidays)
# ---------------------------------------------------------------------------

_US_MARKET_HOLIDAYS = [
    # New Year, MLK, Presidents, Good Friday, Memorial, Juneteenth, July 4,
    # Labor, Thanksgiving, Christmas -- 2026..2028 (observed dates)
    "2026-01-01",
    "2026-01-19",
    "2026-02-16",
    "2026-04-03",
    "2026-05-25",
    "2026-06-19",
    "2026-07-03",
    "2026-09-07",
    "2026-11-26",
    "2026-12-25",
    "2027-01-01",
    "2027-01-18",
    "2027-02-15",
    "2027-03-26",
    "2027-05-31",
    "2027-06-18",
    "2027-07-05",
    "2027-09-06",
    "2027-11-25",
    "2027-12-24",
    "2028-01-17",
    "2028-02-21",
    "2028-04-14",
    "2028-05-29",
    "2028-06-19",
    "2028-07-04",
    "2028-09-04",
    "2028-11-23",
    "2028-12-25",
]
_HOLIDAY_ARR = np.array(_US_MARKET_HOLIDAYS, dtype="datetime64[D]")


def trading_days_between(start, end) -> int:
    """NYSE trading days strictly after `start`, up to and including `end`."""
    s = np.datetime64(pd.Timestamp(start).date(), "D") + np.timedelta64(1, "D")
    e = np.datetime64(pd.Timestamp(end).date(), "D") + np.timedelta64(1, "D")
    if e <= s:
        return 0
    return int(np.busday_count(s, e, holidays=_HOLIDAY_ARR))


def add_trading_days(start, n: int) -> pd.Timestamp:
    """The calendar date that is exactly ``n`` NYSE trading days after ``start``
    (inverse of :func:`trading_days_between`).  ``n<=0`` returns ``start``."""
    d = np.datetime64(pd.Timestamp(start).date(), "D")
    if n <= 0:
        return pd.Timestamp(d)
    d1 = d + np.timedelta64(1, "D")  # "strictly after start"
    out = np.busday_offset(d1, int(n) - 1, roll="forward", holidays=_HOLIDAY_ARR)
    return pd.Timestamp(out)


# ---------------------------------------------------------------------------
# A. discrete
# ---------------------------------------------------------------------------


def bernoulli(mi: MarketInputs) -> DistResult:
    p = mi.up_day_prob
    return DistResult(
        "bernoulli",
        "ברנולי (יום עולה בודד)",
        "discrete",
        {"p": p},
        p,
        p * (1 - p),
        math.sqrt(p * (1 - p)),
        {"P(מחר יום עולה)": p, "P(מחר יום יורד)": 1 - p},
    )


def binomial(mi: MarketInputs, n: int, k: int, mode: str = "up") -> DistResult:
    n = max(int(n), 1)
    p = mi.up_day_prob if mode == "up" else mi.big_move_prob
    k = int(np.clip(k, 0, n))
    lab = "ימי עלייה" if mode == "up" else f"ימי תנועה≥{mi.move_threshold * 100:.0f}%"
    return DistResult(
        f"binomial_{mode}",
        f"בינומי — {lab}",
        "discrete",
        {"n": n, "p": p},
        n * p,
        n * p * (1 - p),
        math.sqrt(n * p * (1 - p)),
        {
            f"P(בדיוק {k} {lab} ב-{n} ימים)": float(stats.binom.pmf(k, n, p)),
            f"P(לפחות {k})": float(stats.binom.sf(k - 1, n, p)),
            f"P(לכל היותר {k})": float(stats.binom.cdf(k, n, p)),
        },
    )


def poisson_crossings(
    mi: MarketInputs, close: pd.Series, target: float, horizon: int, k: int
) -> DistResult:
    w = close.astype(float).dropna().tail(mi.window_days + 1).to_numpy()
    cr = int(np.sum(np.sign(w[1:] - target) != np.sign(w[:-1] - target)))
    lam = max(cr / max(len(w) - 1, 1) * max(horizon, 1), 1e-9)
    k = max(int(k), 0)
    return DistResult(
        "poisson",
        "פואסון — חציות של מחיר היעד",
        "discrete",
        {"lambda": lam},
        lam,
        lam,
        math.sqrt(lam),
        {
            f"P(בדיוק {k} חציות של ${target:,.0f} ב-{horizon} ימים)": float(
                stats.poisson.pmf(k, lam)
            ),
            "P(לפחות חצייה אחת)": float(1 - stats.poisson.pmf(0, lam)),
            f"P(לפחות {max(k, 1)} חציות)": float(stats.poisson.sf(max(k, 1) - 1, lam)),
        },
        note=f"{cr} חציות בפועל ב-{len(w) - 1} הימים האחרונים",
    )


def geometric(mi: MarketInputs, p_success: float, horizon: int, day_n: int) -> DistResult:
    p = float(np.clip(p_success, 1e-4, 1 - 1e-6))
    day_n = max(int(day_n), 1)
    return DistResult(
        "geometric",
        "גיאומטרי — היום הראשון של הצלחה",
        "discrete",
        {"p": p},
        1 / p,
        (1 - p) / p**2,
        math.sqrt((1 - p) / p**2),
        {
            f"P(ההצלחה הראשונה בדיוק ביום {day_n})": float(stats.geom.pmf(day_n, p)),
            f"P(הצלחה תוך {horizon} ימים)": _p(1 - (1 - p) ** max(horizon, 1)),
            "יום צפוי להצלחה הראשונה (תוחלת)": 1 / p,
        },
        note=f"p = הסתברות יומית קדימה שהמחיר יגיע ליעד ({p:.1%}, מסימולציה)",
    )


def negative_binomial(mi: MarketInputs, r_succ: int, day_n: int, p_success: float) -> DistResult:
    p = float(np.clip(p_success, 1e-4, 1 - 1e-6))
    r_succ = max(int(r_succ), 1)
    day_n = max(int(day_n), r_succ)
    mean = r_succ / p
    var = r_succ * (1 - p) / p**2
    # P(the r-th success falls exactly on day_n): NB in "trials until r successes"
    x = day_n - r_succ
    pmf = float(comb(day_n - 1, r_succ - 1) * p**r_succ * (1 - p) ** x)
    return DistResult(
        "negbinom",
        "בינומי שלילי — היום של ההצלחה ה-r",
        "discrete",
        {"r": r_succ, "p": p},
        mean,
        var,
        math.sqrt(var),
        {
            f"P(ההצלחה ה-{r_succ} בדיוק ביום {day_n})": pmf,
            f"P({r_succ} הצלחות תוך {day_n} ימים)": float(stats.nbinom.cdf(x, r_succ, p)),
        },
    )


def hypergeometric(mi: MarketInputs, sample_days: int, k: int) -> DistResult:
    N = mi.window_days
    K = int(round(mi.up_day_prob * N))  # up-days in the population
    nn = int(np.clip(sample_days, 1, N))
    k = int(np.clip(k, 0, nn))
    mean = nn * K / N
    var = nn * (K / N) * (1 - K / N) * (N - nn) / (N - 1)
    return DistResult(
        "hypergeom",
        "היפרגאומטרי — דגימה מהעבר (תיאורי, לא חיזוי)",
        "discrete",
        {"N": N, "K": K, "n": nn},
        mean,
        var,
        math.sqrt(max(var, 0)),
        {
            f"P(בדיוק {k} ימי עלייה בדגימה של {nn} ימים מ-2 השנים האחרונות)": float(
                stats.hypergeom.pmf(k, N, K, nn)
            ),
            f"P(לפחות {k})": float(stats.hypergeom.sf(k - 1, N, K, nn)),
        },
        note="⚠ שאלה תיאורית על העבר (שליפה ללא החזרה מ-N ימים), לא תחזית "
        "לעתיד — ימי מסחר עתידיים אינם 'כד' סופי",
    )


def discrete_uniform_day(horizon: int, day_n: int) -> DistResult:
    n = max(int(horizon), 1)
    day_n = int(np.clip(day_n, 1, n))
    return DistResult(
        "disc_uniform",
        "אחידה בדידה — פריור חסר מידע",
        "discrete",
        {"a": 1, "b": n},
        (n + 1) / 2,
        (n**2 - 1) / 12,
        math.sqrt((n**2 - 1) / 12),
        {f"P(קורה בדיוק ביום {day_n})": 1 / n, f"P(קורה עד יום {day_n})": day_n / n},
        note="כל יום שווה-סיכוי, בלי מידע מהשוק",
    )


# ---------------------------------------------------------------------------
# B. continuous  (h-day log-return space)
# ---------------------------------------------------------------------------


def _h_params(mi: MarketInputs, h: int) -> tuple[float, float]:
    return mi.mu_daily * h, mi.sigma_daily * math.sqrt(h)


def normal_h(mi: MarketInputs, target: float, h: int) -> DistResult:
    h = max(h, 1)
    mu_h, sd_h = _h_params(mi, h)
    z = (math.log(max(target, 1e-9) / mi.last_price) - mu_h) / (sd_h + 1e-12)
    m = mi.last_price * math.exp(mu_h + 0.5 * sd_h**2)
    v = m**2 * (math.exp(sd_h**2) - 1)
    return DistResult(
        "normal",
        "נורמלי — תשואת h ימים",
        "continuous",
        {"mu_h": mu_h, "sigma_h": sd_h},
        m,
        v,
        math.sqrt(v),
        {
            f"P(מחיר ≥ ${target:,.0f} עד היעד)": _p(1 - stats.norm.cdf(z)),
            f"P(מחיר ≤ ${target:,.0f})": _p(stats.norm.cdf(z)),
            "מחיר חציוני צפוי ליעד": mi.last_price * math.exp(mu_h),
        },
    )


def lognormal_price(mi: MarketInputs, target: float, h: int) -> DistResult:
    h = max(h, 1)
    mu_h, sd_h = _h_params(mi, h)
    s, scale = sd_h, mi.last_price * math.exp(mu_h)
    m = scale * math.exp(s**2 / 2)
    v = (math.exp(s**2) - 1) * math.exp(2 * math.log(scale) + s**2)
    return DistResult(
        "lognormal",
        "לוג-נורמלי — רמת המחיר",
        "continuous",
        {"mu": mu_h, "sigma": sd_h},
        m,
        v,
        math.sqrt(v),
        {
            f"P(מחיר ≥ ${target:,.0f})": _p(1 - stats.lognorm.cdf(target, s, scale=scale)),
            f"P(מחיר בטווח ±10% מ-${target:,.0f})": _p(
                stats.lognorm.cdf(target * 1.1, s, scale=scale)
                - stats.lognorm.cdf(target * 0.9, s, scale=scale)
            ),
        },
    )


def _price_moments(mi: MarketInputs, mu_h: float, sigma_eff_sq: float) -> tuple[float, float]:
    """Lognormal-style price E / Var from a log-return with location mu_h and
    (effective) variance sigma_eff_sq.  Approximate for non-Gaussian returns."""
    if not math.isfinite(sigma_eff_sq):
        return math.inf, math.inf
    scale = mi.last_price * math.exp(mu_h)
    m = scale * math.exp(sigma_eff_sq / 2)
    v = m**2 * (math.exp(sigma_eff_sq) - 1)
    return m, v


def student_t_h(mi: MarketInputs, target: float, h: int) -> DistResult:
    h = max(h, 1)
    mu_h = mi.mu_daily * h
    sc_h = mi.t_scale * math.sqrt(h)
    nu = mi.t_nu
    z = (math.log(max(target, 1e-9) / mi.last_price) - mu_h) / (sc_h + 1e-12)
    sigma_eff_sq = sc_h**2 * nu / (nu - 2) if nu > 2 else math.inf  # variance of the return
    m, v = _price_moments(mi, mu_h, sigma_eff_sq)
    return DistResult(
        "student_t",
        "Student-t — זנבות עבים (מחיר, קירוב)",
        "continuous",
        {"nu": nu, "scale_h": sc_h},
        m,
        v,
        math.sqrt(v) if math.isfinite(v) else math.inf,
        {
            f"P(מחיר ≥ ${target:,.0f})": _p(1 - stats.t.cdf(z, nu)),
            f"P(מחיר ≤ ${target:,.0f})": _p(stats.t.cdf(z, nu)),
            "מחיר חציוני צפוי ליעד": mi.last_price * math.exp(mu_h),
        },
        note=f"ν={nu:.1f} (מותאם מהנתונים) — ככל שקטן, הזנבות עבים יותר. "
        f"תוחלת/שונות בקירוב לוג-נורמלי",
    )


def laplace_h(mi: MarketInputs, target: float, h: int) -> DistResult:
    h = max(h, 1)
    loc = mi.mu_daily * h
    b = mi.lap_scale * math.sqrt(h)
    z = math.log(max(target, 1e-9) / mi.last_price)
    m, v = _price_moments(mi, loc, 2 * b**2)  # Laplace return-variance = 2b²
    return DistResult(
        "laplace",
        "לפלס — אקספוננציאלי כפול (מחיר, קירוב)",
        "continuous",
        {"loc": loc, "b": b},
        m,
        v,
        math.sqrt(v),
        {
            f"P(מחיר ≥ ${target:,.0f})": _p(1 - stats.laplace.cdf(z, loc, b)),
            f"P(מחיר ≤ ${target:,.0f})": _p(stats.laplace.cdf(z, loc, b)),
            "מחיר חציוני צפוי ליעד": mi.last_price * math.exp(loc),
        },
        note="מתאים היטב לתשואות יומיות (זנבות מעריכיים); תוחלת/שונות בקירוב לוג-נורמלי",
    )


def exponential_wait(mi: MarketInputs, close: pd.Series, target: float, h: int) -> DistResult:
    """Waiting time (days) until the price first touches the target level."""
    w = close.astype(float).dropna().tail(mi.window_days + 1).to_numpy()
    above = w >= target if target >= mi.last_price else w <= target
    cross = int(np.sum(above[1:] & ~above[:-1]))
    rate = max(cross / max(len(w) - 1, 1), 1e-9)  # touches per day
    mean_wait = 1 / rate
    return DistResult(
        "exponential",
        "אקספוננציאלי — זמן המתנה לנגיעה",
        "continuous",
        {"rate_per_day": rate},
        mean_wait,
        1 / rate**2,
        1 / rate,
        {
            f"P(נוגע ב-${target:,.0f} תוך {h} ימים)": _p(1 - math.exp(-rate * max(h, 1))),
            "ימים צפויים עד נגיעה (תוחלת)": mean_wait,
        },
        note=f"{cross} נגיעות בפועל ב-{len(w) - 1} ימים",
    )


def continuous_uniform(mi: MarketInputs, a: float, b: float) -> DistResult:
    lo, hi = mi.hist_low, mi.hist_high
    a, b = min(a, b), max(a, b)
    span = max(hi - lo, 1e-9)
    ac, bc = max(a, lo), min(b, hi)
    return DistResult(
        "cont_uniform",
        "אחידה רציפה — טווח מחירים",
        "continuous",
        {"low": lo, "high": hi},
        (lo + hi) / 2,
        span**2 / 12,
        span / math.sqrt(12),
        {
            f"P(מחיר בין ${a:,.0f} ל-${b:,.0f})": _p(max(0.0, bc - ac) / span),
            f"P(מחיר ≥ ${b:,.0f})": _p(max(0.0, hi - max(b, lo)) / span),
        },
        note="מניח מחיר שווה-סיכוי בכל הטווח ההיסטורי — בסיס להשוואה",
    )


def gamma_vol(mi: MarketInputs, h: int) -> DistResult:
    """Chi-square-scaled: the realised variance over h days ~ Gamma."""
    h = max(h, 1)
    k_shape = h / 2
    theta = 2 * mi.sigma_daily**2
    mean, var = k_shape * theta, k_shape * theta**2
    ann = math.sqrt(mean / h) * math.sqrt(TRADING_DAYS_YEAR)
    return DistResult(
        "gamma",
        "גמא / χ² — שונות מצטברת",
        "continuous",
        {"shape": k_shape, "scale": theta},
        mean,
        var,
        math.sqrt(var),
        {
            "תנודתיות שנתית משתמעת מהתוחלת": ann,
            "P(שונות ה-h-ימים גבוהה פי 2 מהצפוי)": _p(
                1 - stats.gamma.cdf(2 * mean, k_shape, scale=theta)
            ),
        },
        note="מודל לפיזור התנודתיות, לא לכיוון",
    )


def beta_pup(mi: MarketInputs, x_low: float, x_high: float) -> DistResult:
    """Bayesian posterior on the true P(up-day): Beta(1+ups, 1+downs)."""
    ups = int(np.sum(mi.r > 0))
    downs = int(len(mi.r) - ups)
    a, b = 1 + ups, 1 + downs
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    return DistResult(
        "beta",
        "בטא — פוסטריור על P(יום עולה)",
        "continuous",
        {"alpha": a, "beta": b},
        mean,
        var,
        math.sqrt(var),
        {
            f"P( P(עלייה) בטווח [{x_low:.2f}, {x_high:.2f}] )": _p(
                stats.beta.cdf(x_high, a, b) - stats.beta.cdf(x_low, a, b)
            ),
            "רווח סמך 90% תחתון": float(stats.beta.ppf(0.05, a, b)),
            "רווח סמך 90% עליון": float(stats.beta.ppf(0.95, a, b)),
        },
    )


def cauchy_ref(mi: MarketInputs, target: float, h: int) -> DistResult:
    h = max(h, 1)
    loc = mi.mu_daily * h
    gamma = mi.sigma_daily * math.sqrt(h) * 0.7
    z = math.log(max(target, 1e-9) / mi.last_price)
    return DistResult(
        "cauchy",
        "קושי — זנב קיצוני (ייחוס)",
        "continuous",
        {"loc": loc, "gamma": gamma},
        math.nan,
        math.nan,
        math.nan,
        {f"P(מחיר ≥ ${target:,.0f})": _p(1 - stats.cauchy.cdf(z, loc, gamma))},
        note="לקושי אין תוחלת/שונות — ייחוס לזנב הכי שמן שאפשר",
    )


# ---------------------------------------------------------------------------
# C. extreme value
# ---------------------------------------------------------------------------


def gev_max_move(mi: MarketInputs, h: int, thr_pct: float) -> DistResult:
    """Distribution of the single largest daily move over the next h days (GEV
    on block maxima of |r|)."""
    h = max(h, 2)
    absr = np.abs(mi.r)
    block = max(5, h)
    maxes = [absr[i : i + block].max() for i in range(0, len(absr) - block + 1, block)]
    if len(maxes) >= 5:
        c, loc, scale = stats.genextreme.fit(maxes)
    else:
        c, loc, scale = 0.1, float(np.mean(absr)), float(np.std(absr))
    mean = stats.genextreme.mean(c, loc, scale)
    var = stats.genextreme.var(c, loc, scale)
    thr = thr_pct
    return DistResult(
        "gev",
        "GEV / גומבל — התנועה הקיצונית ב-h ימים",
        "extreme",
        {"shape": float(c), "loc": float(loc), "scale": float(scale)},
        float(mean),
        float(var),
        float(math.sqrt(max(var, 0))),
        {
            f"P(התנועה הגדולה ביותר ב-{h} ימים תעלה על {thr * 100:.0f}%)": _p(
                1 - stats.genextreme.cdf(thr, c, loc, scale)
            ),
            "גודל התנועה הקיצונית הצפוי (תוחלת)": float(mean),
        },
        note="מודל לגודל היום הכי תנודתי, לא לכיוון",
    )


def pareto_tail(mi: MarketInputs, loss_pct: float, h: int) -> DistResult:
    """Power-law tail of daily losses (Hill exponent).  P(a day loses > x%)."""
    a = mi.hill_alpha
    losses = -mi.r[mi.r < 0]
    xm = float(np.percentile(losses, 75)) if len(losses) else 0.02
    x = loss_pct
    p_day = _p((xm / x) ** a) if x > xm else 1.0
    p_h = _p(1 - (1 - p_day) ** max(h, 1))
    mean = a * xm / (a - 1) if a > 1 else math.inf
    var = xm**2 * a / ((a - 1) ** 2 * (a - 2)) if a > 2 else math.inf
    return DistResult(
        "pareto",
        "פארטו / חוק חזקה — זנב ההפסדים",
        "extreme",
        {"alpha": a, "x_min": xm},
        mean,
        var,
        math.sqrt(var) if math.isfinite(var) else math.inf,
        {f"P(יום בודד מפסיד יותר מ-{x * 100:.0f}%)": p_day, f"P(לפחות יום כזה ב-{h} ימים)": p_h},
        note=f"מעריך זנב α≈{a:.1f} (Hill). נמוך = זנב שמן",
    )


# ---------------------------------------------------------------------------
# D. combinatorics
# ---------------------------------------------------------------------------


def path_combinatorics(mi: MarketInputs, h: int, k_up: int) -> DistResult:
    """Count up/down day paths.  h days, each up or down."""
    n = max(int(h), 1)
    k = int(np.clip(k_up, 0, n))

    def _lf(x):
        return float(gammaln(x + 1))

    def _lchoose(a, b):
        return _lf(a) - _lf(b) - _lf(a - b) if 0 <= b <= a else -math.inf

    total = _safe_big(n * math.log(2.0))
    ways_k = _safe_big(_lchoose(n, k))
    p = mi.up_day_prob

    # never-below-start paths that end at k ups (ballot / reflection)
    def ballot(a, b):
        if a < b:
            return 0.0
        l1 = _lchoose(a + b, b)
        if b < 1 or l1 > 709.0:
            return _safe_big(l1)
        return math.exp(l1) - _safe_big(_lchoose(a + b, b - 1))

    never_below = ballot(k, n - k)
    return DistResult(
        "comb_paths",
        "קומבינטוריקה — ספירת מסלולי עלייה/ירידה",
        "combinatorics",
        {"n_days": n, "k_up": k},
        n * p,
        n * p * (1 - p),
        math.sqrt(n * p * (1 - p)),
        {
            "סך כל המסלולים האפשריים (2ⁿ)": total,
            f"מסלולים עם בדיוק {k} ימי עלייה  C({n},{k})": ways_k,
            "מתוכם — שלא יורדים מתחת לנקודת ההתחלה (בליסטה/קטלן)": never_below,
            f"P(בדיוק {k} ימי עלייה)": float(stats.binom.pmf(k, n, p)),
            f"P(המסלול לא יורד מתחת להתחלה | {k} עליות)": _p((2 * k - n + 1) / (k + 1))
            if k >= n - k
            else 0.0,
        },
        note="כל יום = צעד +1 (עלייה) או −1 (ירידה)",
    )


def classic_cases(n_days: int, k_events: int) -> DistResult:
    """The 4 classic selection counts applied to 'which k of the n days see the
    event' -- ordered/unordered × with/without repetition."""
    n = max(int(n_days), 1)
    k = max(int(k_events), 0)

    def logfact(x):
        return float(gammaln(x + 1))

    n_pow_k = _safe_big(k * math.log(n)) if (n > 0 and k > 0) else 1.0
    perm = _safe_big(logfact(n) - logfact(n - k)) if k <= n else 0.0
    multiset = _safe_big(logfact(n + k - 1) - logfact(k) - logfact(n - 1)) if n >= 1 else 0.0
    choose = _safe_big(logfact(n) - logfact(k) - logfact(n - k)) if k <= n else 0.0
    return DistResult(
        "comb_classic",
        "קומבינטוריקה — 4 המקרים הקלאסיים",
        "combinatorics",
        {"n": n, "k": k},
        math.nan,
        math.nan,
        math.nan,
        {
            "nᵏ  — עם סדר, עם חזרות": n_pow_k,
            "סידורים n!/(n−k)!  — עם סדר, בלי חזרות": perm,
            "C(n+k−1, k)  — בלי סדר, עם חזרות (multiset)": multiset,
            "C(n, k)  — בלי סדר, בלי חזרות": choose,
        },
        note="בחירת k ימים/אירועים מתוך n ימי מסחר",
    )


def named_pattern(mi: MarketInputs, pattern: str) -> DistResult:
    """P of a specific up/down sequence, e.g. 'UUDUD' (U=up, D=down)."""
    pat = [c.upper() for c in pattern if c.upper() in "UD"]
    if not pat:
        pat = list("UUD")
    p, q = mi.up_day_prob, 1 - mi.up_day_prob
    prob = 1.0
    for c in pat:
        prob *= p if c == "U" else q
    exp_ups = sum(1 for c in pat if c == "U")
    return DistResult(
        "comb_pattern",
        "קומבינטוריקה — תבנית ימים ספציפית",
        "combinatorics",
        {"length": len(pat)},
        exp_ups,
        math.nan,
        math.nan,
        {
            f"P(רצף בדיוק '{''.join(pat)}')": prob,
            f"P(רצף כלשהו עם {exp_ups} עליות מתוך {len(pat)})": float(
                stats.binom.pmf(exp_ups, len(pat), p)
            ),
            "מספר הרצפים באורך זה עם אותו מספר עליות": float(comb(len(pat), exp_ups)),
        },
        note="U = יום עולה, D = יום יורד. הנחה: ימים בלתי-תלויים",
    )


# ---------------------------------------------------------------------------
# F. time-dependent structure  (answers the "IID is violated" critique)
# ---------------------------------------------------------------------------


def garch_h(mi: MarketInputs, target: float, h: int) -> DistResult:
    """GARCH(1,1) forecast of the h-day volatility -- models volatility
    clustering instead of the static sqrt-time assumption."""
    h = max(int(h), 1)
    static_h = mi.sigma_daily * math.sqrt(h)
    sigma_h, params, note = static_h, {}, "נפילה לחישוב סטטי (GARCH לא התכנס)"
    try:
        from arch import arch_model

        am = arch_model(mi.r * 100, vol="GARCH", p=1, q=1, mean="Constant", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=h, reindex=False)
        daily_var = np.asarray(fc.variance.values).ravel()[:h] / 1e4
        sigma_h = float(math.sqrt(max(np.sum(daily_var), 1e-12)))
        params = {
            "omega": float(res.params.get("omega", np.nan)) / 1e4,
            "alpha[1]": float(res.params.get("alpha[1]", np.nan)),
            "beta[1]": float(res.params.get("beta[1]", np.nan)),
        }
        persist = params["alpha[1]"] + params["beta[1]"]
        note = (
            f"α+β = {persist:.2f} (התמדת תנודתיות). "
            f"תנודתיות עכשווית {'מעל' if sigma_h > static_h else 'מתחת ל'}ממוצע"
        )
    except Exception as e:  # noqa: BLE001
        note = f"GARCH נכשל ({type(e).__name__}); סטטי בשימוש"

    mu_h = mi.mu_daily * h
    z = (math.log(max(target, 1e-9) / mi.last_price) - mu_h) / (sigma_h + 1e-12)
    m, v = _price_moments(mi, mu_h, sigma_h**2)
    return DistResult(
        "garch",
        "GARCH(1,1) — תנודתיות תלוית-זמן",
        "timedep",
        params,
        m,
        v,
        math.sqrt(v),
        {
            f"P(מחיר ≥ ${target:,.0f}) לפי GARCH": _p(1 - stats.norm.cdf(z)),
            "תנודתיות h-ימים — GARCH": sigma_h,
            "תנודתיות h-ימים — סטטית (√t)": static_h,
            "יחס GARCH / סטטי": sigma_h / (static_h + 1e-12),
        },
        note=note,
    )


def markov_pattern(mi: MarketInputs, pattern: str) -> DistResult:
    """A 2-state (Up/Down) Markov chain fit to the data -- gives the *conditional*
    probability of a day sequence, which IID combinatorics cannot."""
    s = (mi.r > 0).astype(int)
    T = np.ones((2, 2))  # Laplace-smoothed counts
    for a, b in zip(s[:-1], s[1:]):
        T[a, b] += 1
    T = T / T.sum(axis=1, keepdims=True)
    ev, evec = np.linalg.eig(T.T)
    stat = np.real(evec[:, int(np.argmin(np.abs(ev - 1)))])
    stat = np.abs(stat) / np.abs(stat).sum()

    pat = [1 if c.upper() == "U" else 0 for c in pattern if c.upper() in "UD"] or [1, 1, 0]
    p_markov = stat[pat[0]]
    for a, b in zip(pat[:-1], pat[1:]):
        p_markov *= T[a, b]
    pu = mi.up_day_prob
    p_iid = float(np.prod([pu if x else 1 - pu for x in pat]))
    ups = sum(pat)
    return DistResult(
        "markov",
        "שרשרת מרקוב — הסתברות מותנית לתבנית",
        "timedep",
        {"P(U|U)": float(T[1, 1]), "P(U|D)": float(T[0, 1]), "P(D|D)": float(T[0, 0])},
        float(ups),
        math.nan,
        math.nan,
        {
            f"P(רצף '{''.join('U' if x else 'D' for x in pat)}') — מרקוב": p_markov,
            "P(אותו רצף) — בהנחת אי-תלות (להשוואה)": p_iid,
            "יחס מרקוב / אי-תלות": p_markov / (p_iid + 1e-12),
            "P(יום עולה אחרי יום עולה)": float(T[1, 1]),
            "P(יום עולה אחרי יום יורד)": float(T[0, 1]),
        },
        note="לוכד מומנטום / mean-reversion יומי. |P(U|U) − P(U|D)| גדול = תלות חזקה",
    )


def markov_updown_run(mi: MarketInputs, h: int, run_len: int) -> DistResult:
    """P of a run of `run_len` consecutive up-days somewhere in the next h days,
    under the fitted Markov chain (vs IID)."""
    s = (mi.r > 0).astype(int)
    T = np.ones((2, 2))
    for a, b in zip(s[:-1], s[1:]):
        T[a, b] += 1
    T = T / T.sum(axis=1, keepdims=True)
    puu = T[1, 1]
    h, run_len = max(int(h), 1), max(int(run_len), 1)
    # rough: expected number of run starts * per-start completion prob
    p_start = mi.up_day_prob
    p_run_here = p_start * puu ** (run_len - 1)
    p_any = _p(1 - (1 - p_run_here) ** max(h - run_len + 1, 1))
    p_iid = _p(1 - (1 - mi.up_day_prob**run_len) ** max(h - run_len + 1, 1))
    return DistResult(
        "markov_run",
        f"מרקוב — רצף של {run_len} ימי עלייה",
        "timedep",
        {"P(U|U)": float(puu)},
        p_run_here * h,
        math.nan,
        math.nan,
        {
            f"P(רצף ≥ {run_len} ימי עלייה תוך {h} ימים) — מרקוב": p_any,
            "אותו דבר — בהנחת אי-תלות": p_iid,
        },
        note="לוכד את הנטייה של ימי עלייה לבוא בסדרות",
    )


# ---------------------------------------------------------------------------
# E. statistical inference
# ---------------------------------------------------------------------------


def clt_mean_return(mi: MarketInputs, h: int) -> DistResult:
    h = max(h, 1)
    se = mi.sigma_daily / math.sqrt(mi.window_days)  # SE of the daily-mean estimate
    mean_h = mi.mu_daily * h
    sd_h = se * h
    return DistResult(
        "clt",
        "משפט הגבול המרכזי — אומדן התשואה הממוצעת",
        "inference",
        {"SE_daily_mean": se},
        mean_h,
        sd_h**2,
        sd_h,
        {
            "רווח סמך 95% תחתון (תשואת h ימים)": mean_h - 1.96 * sd_h,
            "רווח סמך 95% עליון": mean_h + 1.96 * sd_h,
            "המרה למחיר — תחתון": mi.last_price * math.exp(mean_h - 1.96 * sd_h),
            "המרה למחיר — עליון": mi.last_price * math.exp(mean_h + 1.96 * sd_h),
        },
        note="אי-ודאות באומדן ה-drift עצמו, לא בתנודתיות היומית",
    )


def drift_t_test(mi: MarketInputs) -> DistResult:
    t_stat, p_val = stats.ttest_1samp(mi.r, 0.0)
    return DistResult(
        "ttest",
        "מבחן t — האם ה-drift שונה מאפס?",
        "inference",
        {"t": float(t_stat)},
        mi.mu_daily,
        (mi.sigma_daily / math.sqrt(mi.window_days)) ** 2,
        mi.sigma_daily / math.sqrt(mi.window_days),
        {"p-value (H0: drift=0)": float(p_val), "מובהק ב-5%?": 1.0 if p_val < 0.05 else 0.0},
        note="p גבוה = אין ראיה שהכיוון היומי צפוי (רעש)",
    )


def chebyshev_bound(mi: MarketInputs, target: float, h: int) -> DistResult:
    h = max(h, 1)
    mu_h, sd_h = _h_params(mi, h)
    z = abs(math.log(max(target, 1e-9) / mi.last_price) - mu_h) / (sd_h + 1e-12)
    two_sided = 1.0 / z**2 if z > 1 else 1.0  # Chebyshev
    one_sided = 1.0 / (1.0 + z**2)  # Cantelli (valid one-sided)
    return DistResult(
        "chebyshev",
        "צ'בישב / קנטלי — חסם ללא הנחת התפלגות",
        "inference",
        {"k_sigma": z},
        mu_h,
        sd_h**2,
        sd_h,
        {
            f"P(|תשואה − ממוצע| ≥ {z:.1f}σ)  ≤  (צ'בישב, דו-צדדי)": _p(two_sided),
            f"P(המחיר מגיע ל-${target:,.0f})  ≤  (קנטלי, חד-צדדי)": _p(one_sided),
        },
        note="תקף לכל התפלגות עם שונות סופית — חסם עליון רופף אך מובטח",
    )


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------


@dataclass
class ProbReport:
    inputs: MarketInputs
    next_day: dict[str, float]
    results: list[DistResult]

    def by_family(self, fam: str) -> list[DistResult]:
        return [r for r in self.results if r.family == fam]


# which optional sensors each model needs -- turning a sensor OFF hides the
# models that depend on it (so the user can build a lighter calculation).
MODEL_SENSORS: dict[str, tuple[str, ...]] = {
    "bernoulli": (),
    "gamma": (),
    "beta": (),
    "clt": (),
    "ttest": (),
    "binomial_up": ("k",),
    "binomial_move": ("k", "move_thr"),
    "poisson": ("strike", "k"),
    "geometric": ("strike", "day_n"),
    "negbinom": ("strike", "day_n", "r"),
    "hypergeom": ("sample_days", "k"),
    "disc_uniform": ("day_n",),
    "normal": ("strike",),
    "lognormal": ("strike",),
    "student_t": ("strike",),
    "laplace": ("strike",),
    "cauchy": ("strike",),
    "exponential": ("strike",),
    "garch": ("strike",),
    "chebyshev": ("strike",),
    "cont_uniform": ("price_band",),
    "gev": ("extreme_thr",),
    "pareto": ("loss_pct",),
    "comb_paths": ("k",),
    "comb_classic": ("k",),
    "comb_pattern": ("pattern",),
    "markov": ("pattern",),
    "markov_run": ("k",),
}
ALL_SENSOR_NAMES: tuple[str, ...] = (
    "strike",
    "move_thr",
    "k",
    "day_n",
    "r",
    "loss_pct",
    "extreme_thr",
    "sample_days",
    "pattern",
    "price_band",
)


def full_report(
    close: pd.Series,
    target_price: float,
    horizon_days: int,
    k_count: int = 3,
    day_n: int = 5,
    r_successes: int = 2,
    move_threshold: float = 0.05,
    sample_days: int = 20,
    loss_pct: float = 0.10,
    extreme_thr: float = 0.10,
    pattern: str = "UUD",
    price_band: tuple[float, float] | None = None,
    pup_band: tuple[float, float] = (0.45, 0.55),
    lookback: int = LOOKBACK_2Y,
    active: dict[str, bool] | None = None,
) -> ProbReport:
    mi = market_inputs(close, lookback=lookback, move_threshold=move_threshold)
    h = max(int(horizon_days), 1)
    band = price_band or (mi.last_price * 0.8, mi.last_price * 1.2)

    # FORWARD-conditional per-day probability of being at/above the target, from a
    # GBM simulation. (Using the historical fraction of days above the level is
    # wrong: MSTR spent 2024-25 far above today's price, so that fraction is huge
    # for any target below the old highs.)
    _paths = gbm_paths(mi, h, n_paths=20000)
    p_touch_h, _ = _first_passage_mc(_paths, target_price)  # true P(reach within h)
    p_geo = _geo_rate_from_touch(p_touch_h, h)  # correct cumulative rate

    results: list[DistResult] = [
        bernoulli(mi),
        binomial(mi, h, k_count, "up"),
        binomial(mi, h, k_count, "move"),
        poisson_crossings(mi, close, target_price, h, k_count),
        geometric(mi, p_geo, h, day_n),
        negative_binomial(mi, r_successes, day_n, p_geo),
        hypergeometric(mi, sample_days, k_count),
        discrete_uniform_day(h, day_n),
        normal_h(mi, target_price, h),
        lognormal_price(mi, target_price, h),
        student_t_h(mi, target_price, h),
        laplace_h(mi, target_price, h),
        exponential_wait(mi, close, target_price, h),
        continuous_uniform(mi, min(band), max(band)),
        gamma_vol(mi, h),
        beta_pup(mi, pup_band[0], pup_band[1]),
        cauchy_ref(mi, target_price, h),
        gev_max_move(mi, h, extreme_thr),
        pareto_tail(mi, loss_pct, h),
        path_combinatorics(mi, h, k_count),
        classic_cases(h, k_count),
        named_pattern(mi, pattern),
        garch_h(mi, target_price, h),
        markov_pattern(mi, pattern),
        markov_updown_run(mi, h, max(2, k_count)),
        clt_mean_return(mi, h),
        drift_t_test(mi),
        chebyshev_bound(mi, target_price, h),
    ]

    if active is not None:
        results = [
            r for r in results if all(active.get(s, True) for s in MODEL_SENSORS.get(r.key, ()))
        ]

    return ProbReport(inputs=mi, next_day=next_day_expected_price(mi), results=results)


# ===========================================================================
#  CUSTOM QUERY BUILDER  --  one window per model, your own sensors
# ===========================================================================
# Example the user cares about:
#   "P that MSTR touches $231 at least 2 times in the next 23 trading days"
# Every model below takes its own spec dict and answers in its own idiom.


@dataclass
class Sensor:
    name: str
    label: str
    kind: str  # price | days | count | pct | int | band | text
    lo: float = 0.0
    hi: float = 0.0
    default: float = 0.0
    step: float = 1.0
    help: str = ""


def _S0_band(mi: MarketInputs):
    return round(mi.hist_low * 0.3, 2), round(mi.hist_high * 1.6, 2)


def gbm_paths(
    mi: MarketInputs, horizon: int, n_paths: int = 20000, seed: int = 0, fat_tails: bool = True
) -> np.ndarray:
    """Simulate daily log-return paths from the fitted model (Student-t if
    fat_tails else Normal), scaled to the trailing daily drift/vol.
    Returns price paths of shape [n_paths, horizon+1] including t0."""
    rng = np.random.default_rng(seed)
    h = max(int(horizon), 1)
    if fat_tails and mi.t_nu > 2:
        z = rng.standard_t(mi.t_nu, size=(n_paths, h))
        z *= math.sqrt((mi.t_nu - 2) / mi.t_nu)  # unit variance
    else:
        z = rng.standard_normal((n_paths, h))
    steps = mi.mu_daily + mi.sigma_daily * z
    logp = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)], axis=1)
    return mi.last_price * np.exp(logp)


def _touch_counts(paths: np.ndarray, strike: float, s0: float) -> np.ndarray:
    """Per path: number of days the price crosses the strike level (either dir)."""
    side = np.sign(paths - strike)
    crossings = np.sum(np.abs(np.diff(side, axis=1)) > 0, axis=1)
    # if it starts on the "wrong" side and the strike is a target, count reaching it
    return crossings


def _first_passage_mc(paths: np.ndarray, strike: float) -> tuple[float, float]:
    """From simulated paths: P(reach `strike` within the horizon) and the mean
    first-touch day among the paths that reached it."""
    s0 = paths[0, 0]
    reached = paths[:, 1:] >= strike if strike >= s0 else paths[:, 1:] <= strike
    hit_any = reached.any(axis=1)
    if not hit_any.any():
        return 0.0, float("nan")
    first_day = np.argmax(reached[hit_any], axis=1) + 1
    return float(hit_any.mean()), float(first_day.mean())


def _geo_rate_from_touch(p_touch: float, horizon: int) -> float:
    """Per-day effective geometric rate whose cumulative over `horizon` equals
    the true first-passage probability -- so 'P(reach within h days)' is correct
    even though calendar days are correlated."""
    p_touch = float(np.clip(p_touch, 1e-6, 1 - 1e-6))
    h = max(int(horizon), 1)
    return float(1.0 - (1.0 - p_touch) ** (1.0 / h))


def first_passage_prob(mi: MarketInputs, strike: float, horizon: int) -> tuple[float, float]:
    """Analytic P(the GBM path touches `strike` at least once within `horizon`
    days) via the reflection principle, and the expected first-touch day."""
    h = max(int(horizon), 1)
    S0 = mi.last_price
    m = math.log(max(strike, 1e-9) / S0)
    mu, sig = mi.mu_daily, mi.sigma_daily
    if abs(m) < 1e-9:
        return 1.0, 0.0
    up = m > 0
    a = abs(m)
    sqt = sig * math.sqrt(h)
    drift = mu if up else -mu
    p = stats.norm.cdf((-a + drift * h) / sqt) + math.exp(2 * drift * a / sig**2) * stats.norm.cdf(
        (-a - drift * h) / sqt
    )
    p = float(np.clip(p, 0.0, 1.0))
    # expected first-passage day (inverse-Gaussian mean) if drift points toward it
    exp_day = a / abs(drift) if abs(drift) > 1e-9 else float("inf")
    return p, float(min(exp_day, 1e9))


# ---- per-model compute functions (spec-driven) ----------------------------


def _sim(mi, spec, n_paths=25000):
    return gbm_paths(
        mi, int(spec["horizon"]), n_paths=n_paths, fat_tails=bool(spec.get("fat_tails", True))
    )


def _fwd_day_prob(paths, strike) -> float:
    """Forward per-day probability that the price is at/above (or at/below) the
    strike -- the average over simulated days. Conditions on where we are now."""
    if strike >= paths[0, 0]:
        return float(np.clip(np.mean(paths[:, 1:] >= strike), 1e-6, 1 - 1e-6))
    return float(np.clip(np.mean(paths[:, 1:] <= strike), 1e-6, 1 - 1e-6))


def _c_barrier_touch(mi, close, spec) -> DistResult:
    strike, h, k = spec["strike"], int(spec["horizon"]), int(spec["k"])
    p1, exp_day = first_passage_prob(mi, strike, h)
    paths = _sim(mi, spec)
    counts = _touch_counts(paths, strike, mi.last_price)
    return DistResult(
        "barrier_touch",
        "מחסום — נגיעה בשער (reflection + סימולציה)",
        "path",
        {"strike": strike, "horizon": h, "k": k},
        float(np.mean(counts)),
        float(np.var(counts)),
        float(np.std(counts)),
        {
            f"P(נוגע ב-${strike:,.0f} לפחות פעם אחת ב-{h} ימים)": p1,
            f"P(נוגע לפחות {k} פעמים)": float(np.mean(counts >= k)),
            f"P(נוגע בדיוק {k} פעמים)": float(np.mean(counts == k)),
            f"P(סוגר מעל ${strike:,.0f} ביום היעד)": float(np.mean(paths[:, -1] >= strike)),
            "יום צפוי לנגיעה הראשונה": exp_day if math.isfinite(exp_day) else float("nan"),
        },
        note="P(פעם אחת) אנליטי (reflection principle); הספירה מ-25k סימולציות GBM עם זנבות Student-t",
    )


def _c_binomial_close(mi, close, spec) -> DistResult:
    strike, n, k = spec["strike"], int(spec["horizon"]), int(spec["k"])
    p = _fwd_day_prob(_sim(mi, spec), strike)
    mean, var = n * p, n * p * (1 - p)
    rel = "≥" if strike >= mi.last_price else "≤"
    return DistResult(
        "binomial_close",
        f"בינומי — ימים שסוגרים {rel} השער",
        "discrete",
        {"n": n, "p": p},
        mean,
        var,
        math.sqrt(var),
        {
            f"P(בדיוק {k} ימים סוגרים {rel} ${strike:,.0f})": float(stats.binom.pmf(k, n, p)),
            f"P(לפחות {k} ימים כאלה)": float(stats.binom.sf(k - 1, n, p)),
            f"P(אף יום לא סוגר {rel} ${strike:,.0f})": float(stats.binom.pmf(0, n, p)),
        },
        note=f"p = הסתברות יומית קדימה (מסימולציה) שהמחיר {rel} ${strike:,.0f}  ({p:.1%})",
    )


def _c_poisson_cross(mi, close, spec) -> DistResult:
    strike, h, k = spec["strike"], int(spec["horizon"]), int(spec["k"])
    paths = _sim(mi, spec)
    counts = _touch_counts(paths, strike, mi.last_price)
    lam = max(float(np.mean(counts)), 1e-9)
    return DistResult(
        "poisson",
        "פואסון — חציות של השער (קדימה)",
        "discrete",
        {"lambda": lam},
        lam,
        lam,
        math.sqrt(lam),
        {
            f"P(בדיוק {k} חציות של ${strike:,.0f} ב-{h} ימים)": float(stats.poisson.pmf(k, lam)),
            "P(לפחות חצייה אחת)": float(1 - stats.poisson.pmf(0, lam)),
            f"P(לפחות {max(k, 1)} חציות)": float(stats.poisson.sf(max(k, 1) - 1, lam)),
        },
        note="λ = מספר החציות הצפוי מ-25k סימולציות GBM קדימה",
    )


def _c_geometric_touch(mi, close, spec) -> DistResult:
    h = int(spec["horizon"])
    p_touch, _ = _first_passage_mc(_sim(mi, spec), spec["strike"])
    return geometric(mi, _geo_rate_from_touch(p_touch, h), h, int(spec["day_n"]))


def _c_negbinom_touch(mi, close, spec) -> DistResult:
    h = int(spec["horizon"])
    p_touch, _ = _first_passage_mc(_sim(mi, spec), spec["strike"])
    return negative_binomial(
        mi, int(spec["k"]), int(spec["day_n"]), _geo_rate_from_touch(p_touch, h)
    )


def _c_normal(mi, close, spec) -> DistResult:
    return normal_h(mi, spec["strike"], int(spec["horizon"]))


def _c_lognormal(mi, close, spec) -> DistResult:
    return lognormal_price(mi, spec["strike"], int(spec["horizon"]))


def _c_student_t(mi, close, spec) -> DistResult:
    return student_t_h(mi, spec["strike"], int(spec["horizon"]))


def _c_gev(mi, close, spec) -> DistResult:
    return gev_max_move(mi, int(spec["horizon"]), float(spec["move_pct"]) / 100.0)


def _c_pareto(mi, close, spec) -> DistResult:
    return pareto_tail(mi, float(spec["loss_pct"]) / 100.0, int(spec["horizon"]))


def _c_montecarlo(mi, close, spec) -> DistResult:
    strike, h, k = spec["strike"], int(spec["horizon"]), int(spec["k"])
    n_paths = int(spec.get("n_paths", 30000))
    paths = gbm_paths(mi, h, n_paths=n_paths, fat_tails=bool(spec.get("fat_tails", True)))
    counts = _touch_counts(paths, strike, mi.last_price)
    ends = paths[:, -1]
    maxp = paths.max(axis=1)
    return DistResult(
        "montecarlo",
        "Monte-Carlo — סימולציה מלאה של המסלול",
        "path",
        {"n_paths": n_paths, "strike": strike, "horizon": h},
        float(np.mean(ends)),
        float(np.var(ends)),
        float(np.std(ends)),
        {
            f"P(נוגע ב-${strike:,.0f} לפחות {k} פעמים)": float(np.mean(counts >= k)),
            "P(נוגע לפחות פעם אחת)": float(
                np.mean(maxp >= strike)
                if strike >= mi.last_price
                else np.mean(paths.min(axis=1) <= strike)
            ),
            f"P(סוגר מעל ${strike:,.0f})": float(np.mean(ends >= strike)),
            "מחיר סופי חציוני": float(np.median(ends)),
            "מחיר סופי — אחוזון 5": float(np.percentile(ends, 5)),
            "מחיר סופי — אחוזון 95": float(np.percentile(ends, 95)),
        },
        note=f"{n_paths:,} מסלולי GBM, צעד יומי Student-t (ν={mi.t_nu:.1f}), "
        f"drift ו-vol מ-2 השנים האחרונות",
    )


def _c_clt_ci(mi, close, spec) -> DistResult:
    return clt_mean_return(mi, int(spec["horizon"]))


def _c_garch(mi, close, spec) -> DistResult:
    return garch_h(mi, spec["strike"], int(spec["horizon"]))


def _c_markov_pattern(mi, close, spec) -> DistResult:
    return markov_pattern(mi, str(spec.get("pattern", "UUD")))


def _c_markov_run(mi, close, spec) -> DistResult:
    return markov_updown_run(mi, int(spec["horizon"]), int(spec["k"]))


def _PRICE(mi: MarketInputs) -> Sensor:
    return Sensor(
        "strike",
        "שער / מחיר יעד ($)",
        "price",
        *_S0_band(mi),
        round(mi.last_price, 0),
        1.0,
        "המחיר שאתה רוצה לבדוק",
    )


_DAYS = Sensor("horizon", "מספר ימי מסחר", "days", 2, 252, 23, 1, "בכמה ימי מסחר קדימה")
_K = Sensor(
    "k", "כמות הפעמים / אירועים", "count", 0, 60, 2, 1, "כמה פעמים המנייה תיגע / תסגור בשער"
)
_DAYN = Sensor("day_n", "יום ספציפי n", "int", 1, 252, 5, 1, "לבדיקת 'בדיוק ביום n'")

CUSTOM_MODELS: dict[str, dict] = {
    "montecarlo": dict(
        hebrew="Monte-Carlo — סימולציה מלאה",
        family="path",
        fn=_c_montecarlo,
        sensors=lambda mi: [_PRICE(mi), _DAYS, _K],
        flagship=True,
    ),
    "barrier_touch": dict(
        hebrew="מחסום — נגיעה בשער (reflection)",
        family="path",
        fn=_c_barrier_touch,
        sensors=lambda mi: [_PRICE(mi), _DAYS, _K],
    ),
    "binomial_close": dict(
        hebrew="בינומי — ימים שסוגרים בשער+",
        family="discrete",
        fn=_c_binomial_close,
        sensors=lambda mi: [_PRICE(mi), _DAYS, _K],
    ),
    "poisson_cross": dict(
        hebrew="פואסון — חציות של השער",
        family="discrete",
        fn=_c_poisson_cross,
        sensors=lambda mi: [_PRICE(mi), _DAYS, _K],
    ),
    "geometric_touch": dict(
        hebrew="גיאומטרי — היום הראשון בשער",
        family="discrete",
        fn=_c_geometric_touch,
        sensors=lambda mi: [_PRICE(mi), _DAYS, _DAYN],
    ),
    "negbinom_touch": dict(
        hebrew="בינומי שלילי — היום של הפעם ה-k",
        family="discrete",
        fn=_c_negbinom_touch,
        sensors=lambda mi: [_PRICE(mi), _DAYS, _K, _DAYN],
    ),
    "normal": dict(
        hebrew="נורמלי — P(מחיר ≥ שער עד היעד)",
        family="continuous",
        fn=_c_normal,
        sensors=lambda mi: [_PRICE(mi), _DAYS],
    ),
    "lognormal": dict(
        hebrew="לוג-נורמלי — רמת המחיר",
        family="continuous",
        fn=_c_lognormal,
        sensors=lambda mi: [_PRICE(mi), _DAYS],
    ),
    "student_t": dict(
        hebrew="Student-t — זנבות עבים",
        family="continuous",
        fn=_c_student_t,
        sensors=lambda mi: [_PRICE(mi), _DAYS],
    ),
    "gev": dict(
        hebrew="GEV — התנועה הקיצונית ב-h ימים",
        family="extreme",
        fn=_c_gev,
        sensors=lambda mi: [_DAYS, Sensor("move_pct", "תנועת קיצון %", "pct", 2, 40, 10, 1)],
    ),
    "pareto": dict(
        hebrew="פארטו — זנב ההפסדים",
        family="extreme",
        fn=_c_pareto,
        sensors=lambda mi: [_DAYS, Sensor("loss_pct", "הפסד יומי %", "pct", 2, 40, 10, 1)],
    ),
    "garch": dict(
        hebrew="GARCH(1,1) — תנודתיות תלוית-זמן",
        family="timedep",
        fn=_c_garch,
        sensors=lambda mi: [_PRICE(mi), _DAYS],
    ),
    "markov_pattern": dict(
        hebrew="שרשרת מרקוב — תבנית ימים מותנית",
        family="timedep",
        fn=_c_markov_pattern,
        sensors=lambda mi: [Sensor("pattern", "תבנית (U/D)", "text", default="UUD")],
    ),
    "markov_run": dict(
        hebrew="מרקוב — רצף ימי עלייה",
        family="timedep",
        fn=_c_markov_run,
        sensors=lambda mi: [_DAYS, Sensor("k", "אורך הרצף", "count", 2, 15, 3, 1)],
    ),
    "clt_ci": dict(
        hebrew="CLT — רווח סמך למחיר ביעד",
        family="inference",
        fn=_c_clt_ci,
        sensors=lambda mi: [_DAYS],
    ),
}


def custom_model_keys() -> list[str]:
    return list(CUSTOM_MODELS)


def custom_sensors(key: str, close: pd.Series, lookback: int = LOOKBACK_2Y) -> list[Sensor]:
    mi = market_inputs(close, lookback=lookback)
    return CUSTOM_MODELS[key]["sensors"](mi)


def compute_custom(
    close: pd.Series, key: str, spec: dict[str, float], lookback: int = LOOKBACK_2Y
) -> DistResult:
    mi = market_inputs(
        close,
        lookback=lookback,
        move_threshold=float(spec.get("move_pct", 5.0)) / 100 if "move_pct" in spec else 0.05,
    )
    spec = {**{"strike": mi.last_price, "horizon": 23, "k": 2, "day_n": 5}, **spec}
    return CUSTOM_MODELS[key]["fn"](mi, close, spec)


if __name__ == "__main__":
    from data_layer import DataConfig, build_panel, primary_close

    print("=" * 78)
    print("CARN-X Probability & Combinatorics Lab -- Self Test")
    print("=" * 78)
    close = primary_close(build_panel(DataConfig()))
    S0 = float(close.iloc[-1])
    rep = full_report(close, target_price=round(S0 * 1.10), horizon_days=20)

    nd = rep.next_day
    print(f"\nlast ${S0:,.2f} · based on {nd['based_on_days']} days")
    print(
        f">>> next-day expected price: ${nd['expected_price']:,.2f} ({nd['expected_change_pct']:+.2f}%)"
    )
    print(
        f"    skew {nd['skew']:+.2f}  excess-kurt {nd['excess_kurtosis']:+.2f}  "
        f"vol/yr {nd['vol_annual_pct']:.0f}%"
    )

    for fam in ("discrete", "continuous", "extreme", "combinatorics", "timedep", "inference"):
        print(f"\n===== {fam.upper()} =====")
        for r in rep.by_family(fam):
            e = f"{r.mean:,.4g}" if math.isfinite(r.mean) else "—"
            v = f"{r.variance:,.4g}" if math.isfinite(r.variance) else "—"
            s = f"{r.std:,.4g}" if math.isfinite(r.std) else "—"
            print(f"  {r.hebrew}   [E={e}  Var={v}  SD={s}]")
            for lab, val in r.answers.items():
                shown = f"{val:.1%}" if (0 <= val <= 1) else f"{val:,.3g}"
                print(f"      {lab} = {shown}")

    print(f"\ntotal models: {len(rep.results)}")
    print("=" * 78)
