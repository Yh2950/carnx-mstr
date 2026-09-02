"""
CARN-X  --  Bitcoin 4-Year (Halving) Cycle
==========================================
MSTR is, structurally, a leveraged Bitcoin holding. Its multi-month path is
dominated by BTC, and BTC's multi-year path has historically been organised
around the ~4-year halving cycle. This module:

  1. loads the full BTC-USD daily history (yfinance from 2014-09; a short monthly
     supplement extends it back to 2010 for the cycle template only)
  2. builds strictly-causal halving-cycle features (days since / to halving,
     cycle phase 0..1, phase sin/cos, epoch number) -- halving dates are known
     years in advance, so these leak nothing
  3. aligns every completed cycle by "days since halving" in log-return space,
     yielding a median cycle template + quantile dispersion band
  4. projects BTC forward to an arbitrary date by splicing the realised current
     cycle with the historical template for the remaining days
  5. maps a BTC path to an MSTR path via the rolling BTC beta and an mNAV
     premium assumption (mstr_path_from_btc)

Everything here is a *structural scenario*, deliberately separate from the
neural model's calibrated short-horizon forecast. Two prior analog cycles is a
tiny sample and the current cycle is visibly weaker than both -- treat the band,
not the median, as the message.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf

    HAS_YF = True
except ImportError:
    HAS_YF = False

# Bitcoin halving dates (block-height events, known far in advance).
HALVINGS: tuple[pd.Timestamp, ...] = tuple(
    pd.Timestamp(d)
    for d in (
        "2012-11-28",
        "2016-07-09",
        "2020-05-11",
        "2024-04-20",
    )
)
# projected next halving (~210k blocks / ~4y after the last)
NEXT_HALVING = pd.Timestamp("2028-04-17")
CYCLE_DAYS = 1400.0  # nominal cycle length used for the phase encoding

# Coarse monthly BTC closes 2010-07 .. 2014-08, only to give the cycle template
# one extra (partial) early cycle. Source: historical spot references, monthly.
_EARLY_BTC_MONTHLY: dict[str, float] = {
    "2010-08-31": 0.07,
    "2010-12-31": 0.30,
    "2011-06-30": 15.9,
    "2011-12-31": 4.7,
    "2012-06-30": 6.7,
    "2012-11-28": 12.4,
    "2013-04-30": 139.0,
    "2013-11-30": 1130.0,
    "2014-04-30": 445.0,
    "2014-08-31": 480.0,
}


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def load_btc_history(start: str = "2014-01-01", with_early_supplement: bool = True) -> pd.Series:
    """Daily BTC-USD close (naive index), newest completed day last."""
    if not HAS_YF:
        raise RuntimeError("yfinance not installed")
    raw = yf.download("BTC-USD", start=start, auto_adjust=True, progress=False, threads=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s[s.index < pd.Timestamp.now().normalize()]
    s.name = "btc"

    if with_early_supplement:
        early = pd.Series(_EARLY_BTC_MONTHLY)
        early.index = pd.to_datetime(early.index)
        early = early[early.index < s.index.min()]
        s = pd.concat([early, s]).sort_index()
    return s


# ---------------------------------------------------------------------------
# causal halving-cycle features
# ---------------------------------------------------------------------------


def _last_halving(dt: pd.Timestamp) -> pd.Timestamp:
    past = [h for h in HALVINGS if h <= dt]
    return past[-1] if past else HALVINGS[0] - pd.Timedelta(days=int(CYCLE_DAYS))


def _next_halving(dt: pd.Timestamp) -> pd.Timestamp:
    fut = [h for h in HALVINGS if h > dt] + [NEXT_HALVING]
    return fut[0]


def halving_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-date halving-cycle features. Causal: every value depends only on the
    (publicly pre-scheduled) halving calendar, never on price."""
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=idx)
    since = np.array([(d - _last_halving(d)).days for d in idx], dtype=float)
    to_next = np.array([(_next_halving(d) - d).days for d in idx], dtype=float)
    cycle_len = since + to_next
    phase = np.clip(since / np.where(cycle_len > 0, cycle_len, CYCLE_DAYS), 0.0, 1.0)

    out["btc_days_since_halving"] = since
    out["btc_days_to_halving"] = to_next
    out["btc_cycle_phase"] = phase
    out["btc_cycle_phase_sin"] = np.sin(2 * np.pi * phase)
    out["btc_cycle_phase_cos"] = np.cos(2 * np.pi * phase)
    out["btc_halving_epoch"] = np.array(
        [sum(1 for h in HALVINGS if h <= d) for d in idx], dtype=float
    )
    # a smooth "expected cycle regime" prior in [-1,1]: peak ~ day 540, trough ~ day 900
    out["btc_cycle_regime_prior"] = np.cos(2 * np.pi * (since - 540.0) / CYCLE_DAYS)
    return out


# ---------------------------------------------------------------------------
# cycle template (align completed cycles by days-since-halving)
# ---------------------------------------------------------------------------


@dataclass
class CycleTemplate:
    ages: np.ndarray  # grid of days-since-halving
    median_logret: np.ndarray  # weighted-median log(P / P_halving)
    q10: np.ndarray
    q90: np.ndarray
    per_cycle: dict[str, np.ndarray]  # halving-date -> logret path on `ages`
    weights: dict[str, float]
    amplitude_decay: float  # ratio of successive cycles' peak log-gain
    n_cycles: int


# BTC cycle amplitude has collapsed every cycle (2016 peak ~x100 from the low,
# 2020 ~x7, 2024 ~x1.8 so far). The template therefore leans on the RECENT
# cycles; 2012/2016 get a small weight for shape only.
DEFAULT_CYCLE_WEIGHTS: dict[str, float] = {
    "2012-11-28": 0.10,
    "2016-07-09": 0.20,
    "2020-05-11": 0.70,
    "2024-04-20": 1.00,
}


def _weighted_quantile(vals: np.ndarray, w: np.ndarray, q: float) -> float:
    ok = np.isfinite(vals)
    if not ok.any():
        return np.nan
    v, ww = vals[ok], w[ok]
    order = np.argsort(v)
    v, ww = v[order], ww[order]
    cw = np.cumsum(ww) - 0.5 * ww
    cw /= ww.sum()
    return float(np.interp(q, cw, v))


def fit_cycle_template(
    btc: pd.Series,
    max_age_days: int = 1400,
    grid_step: int = 5,
    weights: dict[str, float] | None = None,
) -> CycleTemplate:
    weights = weights or DEFAULT_CYCLE_WEIGHTS
    ages = np.arange(0, max_age_days + 1, grid_step, dtype=float)
    per_cycle: dict[str, np.ndarray] = {}
    used_w: dict[str, float] = {}
    peak_gains: list[tuple[int, float]] = []

    for epoch, h in enumerate(HALVINGS):
        seg = btc[btc.index >= h]
        if len(seg) < 30:
            continue
        p0 = float(seg.iloc[0])
        age = (seg.index - h).days.to_numpy().astype(float)
        lr = np.log(seg.to_numpy() / p0)
        valid = ages <= age.max()
        path = np.full_like(ages, np.nan)
        path[valid] = np.interp(ages[valid], age, lr)
        key = str(h.date())
        per_cycle[key] = path
        used_w[key] = float(weights.get(key, 0.5))
        if valid.sum() > 100:  # a reasonably complete cycle
            peak_gains.append((epoch, float(np.nanmax(path))))

    keys = list(per_cycle)
    stack = np.vstack([per_cycle[k] for k in keys]) if keys else np.zeros((0, len(ages)))
    wv = np.array([used_w[k] for k in keys])
    median = np.array([_weighted_quantile(stack[:, j], wv, 0.5) for j in range(len(ages))])
    q10 = np.array([_weighted_quantile(stack[:, j], wv, 0.10) for j in range(len(ages))])
    q90 = np.array([_weighted_quantile(stack[:, j], wv, 0.90) for j in range(len(ages))])

    decay = 0.5
    if len(peak_gains) >= 2:
        peak_gains.sort()
        ratios = [
            peak_gains[i + 1][1] / peak_gains[i][1]
            for i in range(len(peak_gains) - 1)
            if peak_gains[i][1] > 0
        ]
        if ratios:
            decay = float(np.clip(np.mean(ratios), 0.15, 0.95))

    return CycleTemplate(
        ages=ages,
        median_logret=median,
        q10=q10,
        q90=q90,
        per_cycle=per_cycle,
        weights=used_w,
        amplitude_decay=decay,
        n_cycles=len(per_cycle),
    )


# ---------------------------------------------------------------------------
# macro / micro overlay
# ---------------------------------------------------------------------------


@dataclass
class MacroState:
    risk_score: float  # -1 (risk-off) .. +1 (risk-on)
    drift_adj: float  # logret shift applied to the BTC median path
    band_mult: float  # >1 widens the projection band
    notes: list[str] = field(default_factory=list)


def macro_state(panel: pd.DataFrame) -> MacroState:
    """A compact risk-on/off read from the assets already in the panel
    (VIX, DX-Y, ^TNX, S&P, gold). Risk-off shaves the BTC median and widens
    the band; risk-on nudges it up slightly."""
    notes: list[str] = []
    comp: list[float] = []

    def z(series: pd.Series, win: int = 252) -> float:
        s = series.dropna()
        if len(s) < 30:
            return 0.0
        return float((s.iloc[-1] - s.tail(win).mean()) / (s.tail(win).std() + 1e-9))

    vix = panel.get("vix_close")
    if vix is not None:
        zv = z(vix)
        comp.append(-np.tanh(zv / 1.5))  # high VIX -> risk-off
        if zv > 1.0:
            notes.append(f"VIX elevated (z={zv:+.1f}) → risk-off")
        elif zv < -0.7:
            notes.append(f"VIX subdued (z={zv:+.1f}) → risk-on")

    dxy = panel.get("dxy_close")
    if dxy is not None:
        mom = float(np.log(dxy.dropna().iloc[-1] / dxy.dropna().iloc[-63]))
        comp.append(-np.tanh(mom * 12))  # strong USD -> headwind for BTC
        if abs(mom) > 0.02:
            notes.append(
                f"USD {'strengthening' if mom > 0 else 'weakening'} "
                f"({mom * 100:+.1f}% / 3m) → {'headwind' if mom > 0 else 'tailwind'}"
            )

    tnx = panel.get("tnx_close")
    if tnx is not None:
        chg = float(tnx.dropna().iloc[-1] - tnx.dropna().iloc[-63])
        comp.append(-np.tanh(chg / 0.8))  # rising yields -> tighter liquidity
        if abs(chg) > 0.4:
            notes.append(
                f"10y yield {chg:+.2f} pts / 3m → {'tighter' if chg > 0 else 'easier'} liquidity"
            )

    spx = panel.get("spx_close")
    if spx is not None:
        s = spx.dropna()
        trend = float(s.iloc[-1] / s.tail(200).mean() - 1)
        comp.append(np.tanh(trend * 6))
        notes.append(f"S&P {'above' if trend > 0 else 'below'} its 200d mean ({trend * 100:+.1f}%)")

    score = float(np.clip(np.mean(comp), -1, 1)) if comp else 0.0
    drift_adj = 0.12 * score  # +/-12% logret tilt at the extremes
    band_mult = float(1.0 + 0.35 * max(0.0, -score))  # only risk-off widens
    notes.insert(
        0,
        f"macro risk score {score:+.2f} "
        f"({'risk-off' if score < -0.2 else 'risk-on' if score > 0.2 else 'neutral'})",
    )
    return MacroState(risk_score=score, drift_adj=drift_adj, band_mult=band_mult, notes=notes)


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------


@dataclass
class BTCProjection:
    target_date: pd.Timestamp
    last_date: pd.Timestamp
    last_price: float
    horizon_days: int
    current_age: int
    median_price: float
    low_price: float  # ~10th pct
    high_price: float  # ~90th pct
    median_logret: float
    band_logret: tuple[float, float]
    template_used: str
    daily_path: pd.DataFrame  # date, p_median, p_low, p_high


def project_btc(
    btc: pd.Series,
    target_date,
    template: CycleTemplate | None = None,
    current_cycle_weight: float = 0.6,
    extra_sigma_per_year: float = 0.60,
    macro: MacroState | None = None,
) -> BTCProjection:
    """Project BTC from the last close to ``target_date``.

    remaining path = blend( historical template increment ,
                            current cycle's own trailing drift )
    widened by a random-walk-in-log variance term so the band reflects that the
    cycle is only a weak prior (2-3 analogs).
    """
    template = template or fit_cycle_template(btc)
    btc = btc[btc.index < pd.Timestamp.now().normalize()]
    last_date = btc.index[-1]
    last_price = float(btc.iloc[-1])
    target_date = pd.Timestamp(target_date)
    horizon_days = (target_date - last_date).days
    if horizon_days <= 0:
        raise ValueError("target_date must be in the future")

    h = _last_halving(last_date)
    age_now = (last_date - h).days
    age_tgt = age_now + horizon_days

    def tmpl(age):
        return float(
            np.interp(
                np.clip(age, template.ages[0], template.ages[-1]),
                template.ages,
                template.median_logret,
            )
        )

    # raw template increment over the remaining window (shape / timing)
    tmpl_incr_raw = tmpl(min(age_tgt, template.ages[-1])) - tmpl(age_now)

    # THE key correction: this cycle's amplitude has been a fraction of the
    # historical template so far -- scale the template increment by that fraction
    # so we borrow the cycle's *shape*, not the (collapsed) magnitude of 2016/2020.
    # realised log-gain of the current cycle so far -- take it straight from the
    # price (the template path can be NaN right at the current-age edge)
    seg_cur = btc[btc.index >= h]
    realized_now = float(np.log(btc.iloc[-1] / seg_cur.iloc[0])) if len(seg_cur) else 0.0
    tmpl_now = tmpl(age_now)
    if math.isfinite(realized_now) and tmpl_now > 0.1:
        amp_ratio = float(np.clip(realized_now / tmpl_now, 0.05, 1.5))
    else:
        amp_ratio = template.amplitude_decay
    tmpl_incr = tmpl_incr_raw * amp_ratio
    if not math.isfinite(tmpl_incr):
        tmpl_incr = realized_now if math.isfinite(realized_now) else 0.0

    # current cycle's own realised drift (per day) over the last ~120d
    lb = min(120, len(btc) - 1)
    cur_incr = float(np.log(btc.iloc[-1] / btc.iloc[-lb]) / lb) * horizon_days

    w = float(np.clip(current_cycle_weight, 0.0, 1.0))
    median_lr = (1 - w) * tmpl_incr + w * cur_incr

    # macro / micro overlay
    macro_drift = float(macro.drift_adj) if macro is not None else 0.0
    macro_band = float(macro.band_mult) if macro is not None else 1.0
    median_lr += macro_drift * (horizon_days / 365.0)

    # dispersion: template spread at the target age + a growing RW term
    tmpl_band = 0.5 * abs(
        float(
            np.interp(
                age_tgt,
                template.ages,
                template.q90 - template.q10,
                right=(template.q90 - template.q10)[-1],
            )
        )
    )
    rw_sigma = extra_sigma_per_year * np.sqrt(horizon_days / 365.0)
    total_sigma = float(np.sqrt(tmpl_band**2 + rw_sigma**2)) * macro_band

    lo_lr, hi_lr = median_lr - 1.2816 * total_sigma, median_lr + 1.2816 * total_sigma

    # daily path (linear in log space for the median, cone for the band)
    days = pd.date_range(last_date, target_date, freq="D")[1:]
    frac = np.linspace(0, 1, len(days))
    med = last_price * np.exp(median_lr * frac)
    sig_t = total_sigma * np.sqrt(frac)
    path = pd.DataFrame(
        {
            "date": days,
            "p_median": med,
            "p_low": last_price * np.exp(median_lr * frac - 1.2816 * sig_t),
            "p_high": last_price * np.exp(median_lr * frac + 1.2816 * sig_t),
        }
    )

    return BTCProjection(
        target_date=target_date,
        last_date=last_date,
        last_price=last_price,
        horizon_days=horizon_days,
        current_age=age_now,
        median_price=last_price * np.exp(median_lr),
        low_price=last_price * np.exp(lo_lr),
        high_price=last_price * np.exp(hi_lr),
        median_logret=median_lr,
        band_logret=(lo_lr, hi_lr),
        template_used=(
            f"{template.n_cycles} cycles · amp ratio vs template {amp_ratio:.2f} · "
            f"blend w_current={w:.2f}"
            + (
                f" · macro drift {macro_drift:+.2f}/yr, band ×{macro_band:.2f}"
                if macro is not None
                else ""
            )
        ),
        daily_path=path,
    )


# ---------------------------------------------------------------------------
# BTC path -> MSTR path
# ---------------------------------------------------------------------------


@dataclass
class MSTRFromBTC:
    beta: float
    beta_source: str
    mnav_drift_annual: float
    last_mstr: float
    median_price: float
    low_price: float
    high_price: float
    daily_path: pd.DataFrame


def mstr_path_from_btc(
    mstr_close: pd.Series,
    btc_close: pd.Series,
    proj: BTCProjection,
    beta: float | None = None,
    beta_window: int = 180,
    mnav_drift_annual: float = 0.0,
) -> MSTRFromBTC:
    """Translate a BTC projection into an MSTR projection.

    MSTR log-return  ≈  beta · BTC log-return  +  mNAV-premium drift
    beta is the trailing OLS slope of MSTR daily log-returns on BTC's (a number
    that has run ~1.5-2.5 through the ETF era); the premium drift term lets the
    user express a view on mNAV expansion / compression.
    """
    m = np.log(mstr_close).diff().dropna()
    b = np.log(btc_close).diff().reindex(m.index).dropna()
    m, b = m.align(b, join="inner")
    if beta is None:
        win = slice(-beta_window, None)
        cov = np.cov(m.iloc[win], b.iloc[win])
        beta = float(cov[0, 1] / (cov[1, 1] + 1e-12))
        beta_src = f"trailing {beta_window}d OLS"
    else:
        beta_src = "user"
    beta = float(np.clip(beta, 0.5, 4.0))

    last_mstr = float(mstr_close.iloc[-1])
    n = len(proj.daily_path)
    frac = np.linspace(0, 1, n)
    drift_m = mnav_drift_annual * (proj.horizon_days / 365.0)

    def scale(btc_col):
        btc_lr = np.log(btc_col.to_numpy() / proj.last_price)
        return last_mstr * np.exp(beta * btc_lr + drift_m * frac)

    path = pd.DataFrame(
        {
            "date": proj.daily_path["date"],
            "p_median": scale(proj.daily_path["p_median"]),
            "p_low": scale(proj.daily_path["p_low"]),
            "p_high": scale(proj.daily_path["p_high"]),
        }
    )
    return MSTRFromBTC(
        beta=beta,
        beta_source=beta_src,
        mnav_drift_annual=mnav_drift_annual,
        last_mstr=last_mstr,
        median_price=float(path["p_median"].iloc[-1]),
        low_price=float(path["p_low"].iloc[-1]),
        high_price=float(path["p_high"].iloc[-1]),
        daily_path=path,
    )


# ---------------------------------------------------------------------------
# Monte-Carlo under the cycle regime  (for the probability calculator)
# ---------------------------------------------------------------------------


def simulate_btc_paths(
    btc: pd.Series,
    horizon_days: int,
    template: CycleTemplate | None = None,
    n_paths: int = 20000,
    current_cycle_weight: float = 0.6,
    macro: MacroState | None = None,
    seed: int = 0,
    t_nu: float = 4.0,
) -> np.ndarray:
    """Daily BTC price paths whose *drift* is the halving-cycle projection (not
    the trailing 2y drift), keeping BTC's realised daily vol and fat tails.
    Shape [n_paths, horizon_days+1], price units, includes t0."""
    template = template or fit_cycle_template(btc)
    proj = project_btc(
        btc,
        btc.index[-1] + pd.Timedelta(days=int(horizon_days * 1.5) + 5),
        template,
        current_cycle_weight=current_cycle_weight,
        macro=macro,
    )
    h = max(int(horizon_days), 1)
    total_drift = proj.median_logret * (h / max(proj.horizon_days, 1))  # scale to h
    daily_drift = total_drift / h

    r = np.log(btc).diff().dropna().to_numpy()
    daily_sigma = float(np.std(r[-500:], ddof=1))

    rng = np.random.default_rng(seed)
    z = rng.standard_t(t_nu, size=(n_paths, h)) * math.sqrt((t_nu - 2) / t_nu)
    steps = daily_drift + daily_sigma * z
    logp = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)], axis=1)
    return float(btc.iloc[-1]) * np.exp(logp)


def mstr_from_btc_paths(
    mstr_close: pd.Series,
    btc_close: pd.Series,
    btc_paths: np.ndarray,
    beta: float | None = None,
    beta_window: int = 252,
    mnav_drift_annual: float = 0.0,
) -> np.ndarray:
    """Map BTC price paths to MSTR price paths:
    MSTR_t = MSTR_0 * exp( beta * log(BTC_t / BTC_0) + mNAV_drift * t/252 )."""
    m = np.log(mstr_close).diff().dropna()
    b = np.log(btc_close).diff().reindex(m.index).dropna()
    m, b = m.align(b, join="inner")
    if beta is None:
        cov = np.cov(m.iloc[-beta_window:], b.iloc[-beta_window:])
        beta = float(cov[0, 1] / (cov[1, 1] + 1e-12))
    beta = float(np.clip(beta, 0.5, 4.0))

    btc0 = btc_paths[:, [0]]
    btc_lr = np.log(btc_paths / btc0)
    h = btc_paths.shape[1] - 1
    tfrac = np.linspace(0, 1, btc_paths.shape[1])[None, :]
    drift_m = mnav_drift_annual * (h / 252.0)
    return float(mstr_close.iloc[-1]) * np.exp(beta * btc_lr + drift_m * tfrac)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 74)
    print("CARN-X BTC 4-Year Cycle -- Self Test")
    print("=" * 74)

    btc = load_btc_history()
    print(
        f"\nBTC history: {btc.index.min().date()} -> {btc.index.max().date()}  "
        f"({len(btc)} points)  last=${btc.iloc[-1]:,.0f}"
    )

    tmpl = fit_cycle_template(btc)
    print(f"\ncycle template: {tmpl.n_cycles} completed cycles aligned by days-since-halving")
    for age in (180, 365, 540, 730, 900, 1095, 1300):
        j = np.argmin(np.abs(tmpl.ages - age))
        print(
            f"  day {age:4d}:  median x{np.exp(tmpl.median_logret[j]):5.2f}   "
            f"band x[{np.exp(tmpl.q10[j]):.2f}, {np.exp(tmpl.q90[j]):.2f}]"
        )

    hf = halving_features(pd.DatetimeIndex([btc.index[-1]]))
    print(f"\nnow: {hf.iloc[0].to_dict()}")

    for tgt in ("2027-04-30", "2027-12-31"):
        proj = project_btc(btc, tgt, tmpl, current_cycle_weight=0.55)
        print(
            f"\n--- BTC projection to {tgt} ({proj.horizon_days}d, age {proj.current_age}->{proj.current_age + proj.horizon_days}) ---"
        )
        print(
            f"  median  ${proj.median_price:>10,.0f}   ({proj.median_logret:+.2f} logret, x{np.exp(proj.median_logret):.2f})"
        )
        print(f"  80% band ${proj.low_price:,.0f}  ..  ${proj.high_price:,.0f}")

        try:
            from data_layer import build_panel, DataConfig

            panel = build_panel(DataConfig())
            mproj = mstr_path_from_btc(
                panel["mstr_close"].dropna(), panel["btc_close"].dropna(), proj
            )
            print(f"  -> MSTR (beta {mproj.beta:.2f}, {mproj.beta_source}):")
            print(
                f"     median ${mproj.median_price:,.0f}   80% band ${mproj.low_price:,.0f} .. ${mproj.high_price:,.0f}"
            )
        except Exception as e:
            print(f"  (MSTR mapping skipped: {e})")

    print("\n" + "=" * 74)
    print("BTC Cycle module ready.")
    print("=" * 74)
