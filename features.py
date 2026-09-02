"""
CARN-X  --  Feature Engineering  (causal)
========================================
Turns the raw multi-asset panel from data_layer.build_panel into a strictly
causal feature matrix: every value in row ``t`` is computable from information
available at the close of day ``t``. No target leakage, no forward fill of the
future.

Feature families
----------------
- momentum / returns        : multi-lag log returns, cumulative returns
- realized volatility       : rolling std, EWMA (RiskMetrics), Parkinson, vol-of-vol
- distribution shape        : rolling skew / kurtosis
- price location            : z-score vs rolling mean, drawdown from rolling peak
- oscillators               : RSI, rate-of-change, MACD histogram
- cross-asset (MSTR<->BTC)  : rolling beta, correlation, idiosyncratic residual return
- leverage-ETF sentiment    : MSTU/MSTZ relative moves (when available)
- macro                     : VIX / TNX / DXY / SPX / NDX levels & changes, z-scores
- NAV premium               : optional, merged from nav_premium if a CSV is supplied
- calendar                  : day-of-week, month, turn-of-month
- missingness flags         : 1.0 when an asset column was NaN before imputation

``build_features`` returns ``(X, meta)`` where ``X`` is a float32 DataFrame on
the panel index (minus a warmup prefix) and ``meta`` lists the column groups.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

RISKMETRICS_LAMBDA = 0.94
EPS = 1e-12


@dataclass
class FeatureConfig:
    primary_key: str = "mstr"
    btc_key: str = "btc"
    return_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    cum_windows: tuple[int, ...] = (5, 10, 20, 60)
    vol_windows: tuple[int, ...] = (5, 10, 20, 60)
    shape_window: int = 60
    zscore_windows: tuple[int, ...] = (20, 60)
    drawdown_windows: tuple[int, ...] = (20, 60, 252)
    rsi_window: int = 14
    beta_windows: tuple[int, ...] = (20, 60)
    # "rich" per-asset feature set only for these; others get a light set
    rich_assets: tuple[str, ...] = ("mstr", "btc", "ibit")
    macro_assets: tuple[str, ...] = ("vix", "tnx", "dxy", "spx", "ndx", "gld", "qqq")
    warmup: int = 260  # rows dropped from the front (longest window)
    nav_csv_path: str | None = None


# ---------------------------------------------------------------------------
# primitives (all causal)
# ---------------------------------------------------------------------------


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.clip(lower=EPS)).diff()


def _ewma_vol(rets: pd.Series, lam: float = RISKMETRICS_LAMBDA) -> pd.Series:
    """RiskMetrics exponentially-weighted volatility (causal)."""
    var = rets.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / (loss + EPS)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig) / (close + EPS)


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window).mean()
    s = x.rolling(window).std()
    return (x - m) / (s + EPS)


def _drawdown_from_peak(close: pd.Series, window: int) -> pd.Series:
    peak = close.rolling(window, min_periods=window // 2).max()
    return close / (peak + EPS) - 1.0


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    hl = np.log((high / low.clip(lower=EPS)).clip(lower=EPS)) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4.0 * np.log(2.0)))


# ---------------------------------------------------------------------------
# per-asset blocks
# ---------------------------------------------------------------------------


def _asset_features(panel: pd.DataFrame, key: str, cfg: FeatureConfig, rich: bool) -> pd.DataFrame:
    close = panel.get(f"{key}_close")
    if close is None:
        return pd.DataFrame(index=panel.index)

    out = pd.DataFrame(index=panel.index)
    rets = _log_returns(close)

    for lag in cfg.return_lags:
        out[f"{key}_ret_{lag}"] = rets.shift(lag - 1) if lag > 1 else rets
    for w in cfg.cum_windows:
        out[f"{key}_cumret_{w}"] = rets.rolling(w).sum()
    for w in cfg.vol_windows:
        out[f"{key}_rv_{w}"] = rets.rolling(w).std()
    out[f"{key}_ewmavol"] = _ewma_vol(rets)
    out[f"{key}_volofvol"] = rets.rolling(20).std().rolling(20).std()

    if rich:
        for w in cfg.zscore_windows:
            out[f"{key}_pricez_{w}"] = _rolling_zscore(np.log(close.clip(lower=EPS)), w)
        for w in cfg.drawdown_windows:
            out[f"{key}_dd_{w}"] = _drawdown_from_peak(close, w)
        out[f"{key}_skew_{cfg.shape_window}"] = rets.rolling(cfg.shape_window).skew()
        out[f"{key}_kurt_{cfg.shape_window}"] = rets.rolling(cfg.shape_window).kurt()
        out[f"{key}_rsi"] = _rsi(close, cfg.rsi_window) / 100.0 - 0.5
        out[f"{key}_macd"] = _macd_hist(close)
        out[f"{key}_ret_accel"] = rets.rolling(5).mean() - rets.rolling(20).mean()

        high, low = panel.get(f"{key}_high"), panel.get(f"{key}_low")
        if high is not None and low is not None:
            out[f"{key}_parkinson_20"] = _parkinson_vol(high, low, 20)
            out[f"{key}_hl_range"] = (high - low) / (close + EPS)

        vol = panel.get(f"{key}_volume")
        if vol is not None:
            out[f"{key}_volz_20"] = _rolling_zscore(np.log1p(vol), 20)
            out[f"{key}_dollarvol_z"] = _rolling_zscore(np.log1p(vol * close), 60)

    return out


# ---------------------------------------------------------------------------
# cross-asset blocks
# ---------------------------------------------------------------------------


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Causal rolling OLS slope of y on x, plus rolling correlation."""
    cov = y.rolling(window).cov(x)
    var_x = x.rolling(window).var()
    beta = cov / (var_x + EPS)
    corr = y.rolling(window).corr(x)
    return beta, corr


def _cross_asset_features(panel: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    p = cfg.primary_key
    prim_ret = _log_returns(panel[f"{p}_close"])

    for other in ("btc", "ibit", "spx", "ndx", "gld"):
        oc = panel.get(f"{other}_close")
        if oc is None:
            continue
        oret = _log_returns(oc)
        for w in cfg.beta_windows:
            beta, corr = _rolling_beta(prim_ret, oret, w)
            out[f"{p}_{other}_beta_{w}"] = beta
            out[f"{p}_{other}_corr_{w}"] = corr
            # idiosyncratic (residual) return: MSTR move not explained by `other`
            out[f"{p}_{other}_resid_{w}"] = prim_ret - beta * oret
        # relative-strength ratio z-score
        ratio = np.log((panel[f"{p}_close"] / (oc + EPS)).clip(lower=EPS))
        out[f"{p}_{other}_ratio_z_60"] = _rolling_zscore(ratio, 60)

    # leverage-ETF sentiment (present only after 2024) -- signed 2x/-2x mismatch
    mstu, mstz = panel.get("mstu_close"), panel.get("mstz_close")
    if mstu is not None and mstz is not None:
        lret = _log_returns(mstu)
        sret = _log_returns(mstz)
        # if the pair were perfectly rebalanced, lret ~= -sret; deviation = flow/decay
        out["lev_etf_decay"] = (lret + sret).rolling(5).mean()
        out["lev_etf_skew"] = lret.rolling(20).mean() + sret.rolling(20).mean()

    return out


# ---------------------------------------------------------------------------
# macro block
# ---------------------------------------------------------------------------


def _macro_features(panel: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    for key in cfg.macro_assets:
        c = panel.get(f"{key}_close")
        if c is None:
            continue
        if key in ("vix", "tnx"):
            # levels matter for these
            out[f"{key}_level"] = c
            out[f"{key}_level_z_60"] = _rolling_zscore(c, 60)
            out[f"{key}_chg_5"] = c.diff(5)
        else:
            r = _log_returns(c)
            out[f"{key}_ret_1"] = r
            out[f"{key}_cumret_20"] = r.rolling(20).sum()
            out[f"{key}_rv_20"] = r.rolling(20).std()
    # VIX term-structure proxy: fast vs slow VIX
    vix = panel.get("vix_close")
    if vix is not None:
        out["vix_contango_proxy"] = vix.rolling(5).mean() / (vix.rolling(20).mean() + EPS) - 1.0
    return out


# ---------------------------------------------------------------------------
# NAV premium block (optional)
# ---------------------------------------------------------------------------


def _nav_features(panel: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    if not cfg.nav_csv_path:
        return out
    try:
        from nav_premium import NAVPremiumCalculator, load_nav_csv

        nav_df = load_nav_csv(cfg.nav_csv_path)
        calc = NAVPremiumCalculator()
        macro = calc.to_macro_series(calc.compute(nav_df), pd.DatetimeIndex(panel.index))
        from nav_premium import MACRO_VECTOR_LABELS

        for i, lab in enumerate(MACRO_VECTOR_LABELS):
            out[f"nav_{lab}"] = macro[:, i]
    except Exception as e:  # noqa: BLE001
        print(f"[features] NAV block skipped: {e}")
    return out


# ---------------------------------------------------------------------------
# calendar block
# ---------------------------------------------------------------------------


def _data_health_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Turn data_layer's per-asset forward-fill flags into causal features so
    the model can learn to distrust rows where an input was stale."""
    out = pd.DataFrame(index=panel.index)
    flag_cols = [c for c in panel.columns if c.endswith("_ffilled")]
    for c in flag_cols:
        out[f"health_{c[:-8]}_stale5"] = panel[c].rolling(5, min_periods=1).sum()
    if flag_cols:
        out["health_any_stale"] = panel[flag_cols].max(axis=1)
    return out


def _btc_cycle_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Bitcoin halving-cycle features -- causal (halving dates are pre-scheduled).
    MSTR tracks BTC, and BTC's multi-month regime has historically been organised
    around the ~4-year halving cycle, so cycle phase is a real conditioning
    variable for the model."""
    try:
        from btc_cycle import halving_features

        return halving_features(index)
    except Exception as e:  # noqa: BLE001
        print(f"[features] btc_cycle block skipped: {e}")
        return pd.DataFrame(index=index)


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=index)
    dow = idx.dayofweek.values.astype(float)
    out["cal_dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    out["cal_dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    moy = idx.month.values.astype(float)
    out["cal_moy_sin"] = np.sin(2 * np.pi * moy / 12.0)
    out["cal_moy_cos"] = np.cos(2 * np.pi * moy / 12.0)
    dim = idx.days_in_month.values.astype(float)
    out["cal_tom"] = (idx.day.values >= (dim - 2)).astype(float)  # turn of month
    out["cal_som"] = (idx.day.values <= 3).astype(float)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


@dataclass
class FeatureMeta:
    columns: list[str]
    groups: dict[str, list[str]]
    warmup_rows_dropped: int
    n_missing_flag_cols: int
    date_min: str
    date_max: str


def build_features(
    panel: pd.DataFrame, cfg: FeatureConfig | None = None
) -> tuple[pd.DataFrame, FeatureMeta]:
    cfg = cfg or FeatureConfig()
    assets = sorted({c.split("_")[0] for c in panel.columns})

    groups: dict[str, pd.DataFrame] = {}
    for key in assets:
        rich = key in cfg.rich_assets or key == cfg.primary_key
        if key in cfg.macro_assets and not rich:
            continue  # macro handled separately
        groups[f"asset_{key}"] = _asset_features(panel, key, cfg, rich=rich)

    groups["cross_asset"] = _cross_asset_features(panel, cfg)
    groups["macro"] = _macro_features(panel, cfg)
    groups["nav"] = _nav_features(panel, cfg)
    groups["calendar"] = _calendar_features(panel.index)
    groups["data_health"] = _data_health_features(panel)
    groups["btc_cycle"] = _btc_cycle_features(panel.index)

    X = pd.concat([g for g in groups.values() if g.shape[1] > 0], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]

    # replace inf, add missingness flags, then impute
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_cols = [c for c in X.columns if X[c].isna().any()]
    flags = pd.DataFrame(
        {f"{c}__isna": X[c].isna().astype(np.float32) for c in missing_cols},
        index=X.index,
    )
    # causal imputation: forward-fill then fill remaining head with 0
    X = X.ffill().fillna(0.0)
    X = pd.concat([X, flags], axis=1)

    # drop warmup prefix
    X = X.iloc[cfg.warmup :]

    # drop columns that carry no information over the usable window.
    # (a column constant across the whole sample cannot leak; the fold scaler
    #  handles per-fold degeneracy via its +1e-8 std floor.)
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        X = X.drop(columns=const_cols)

    # NOTE: no global winsorization here -- it would use full-sample quantiles
    # (future rows) to clip past rows. Outlier control happens per-fold in
    # dataset.FoldScaler (train-only stats + z-clip).
    X = X.astype(np.float32)

    group_cols = {
        name: [c for c in df.columns if c in X.columns]
        for name, df in groups.items()
        if df.shape[1] > 0
    }
    group_cols["missingness"] = [c for c in X.columns if c.endswith("__isna")]

    meta = FeatureMeta(
        columns=list(X.columns),
        groups=group_cols,
        warmup_rows_dropped=cfg.warmup,
        n_missing_flag_cols=len(flags.columns),
        date_min=str(X.index.min().date()),
        date_max=str(X.index.max().date()),
    )
    return X, meta


if __name__ == "__main__":
    from data_layer import DataConfig, build_panel

    print("=" * 70)
    print("CARN-X Feature Engineering -- Self Test")
    print("=" * 70)

    panel = build_panel(DataConfig())
    X, meta = build_features(panel)

    print(f"\nfeature matrix : {X.shape}")
    print(f"date range     : {meta.date_min} -> {meta.date_max}")
    print(f"missing flags  : {meta.n_missing_flag_cols}")
    print("\nfeature groups :")
    for name, cols in meta.groups.items():
        print(f"  {name:<18} {len(cols):>3} cols")

    assert not X.isna().any().any(), "NaNs leaked into the feature matrix"
    assert np.isfinite(X.values).all(), "non-finite values in the feature matrix"

    corr_with_next = {}
    fwd = np.log(panel["mstr_close"]).diff().shift(-1).reindex(X.index)
    for c in X.columns:
        corr_with_next[c] = np.corrcoef(X[c].values, fwd.fillna(0.0).values)[0, 1]
    top = sorted(corr_with_next.items(), key=lambda kv: -abs(kv[1]))[:10]
    print("\ntop |corr| with next-day MSTR log-return (sanity, not a promise):")
    for c, r in top:
        print(f"  {c:<32} {r:+.3f}")

    print("\n" + "=" * 70)
    print("Feature Engineering ready.")
    print("=" * 70)
