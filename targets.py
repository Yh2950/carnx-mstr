"""
CARN-X  --  Targets  (multi-horizon, vol-normalized)
===================================================
Builds the supervised targets for the forecasting model from the raw price
series. The model predicts, for each horizon ``h``:

    y_scaled[t, h] = ( log(P[t+h] / P[t]) - drift_h )  /  ( sigma_ewma[t] * sqrt(h) )

i.e. the *vol-normalized* forward log-return. Working in vol units keeps the
regression target roughly stationary across MSTR's wildly changing volatility
regimes; walk_forward converts the model's output back to raw return space
using the trailing EWMA vol that is known at prediction time.

Also produced:
    y_raw[t, h]        raw forward log-return (for reporting / economic backtest)
    y_fwdvol[t, h]     realized vol over the forward window (a vol-forecast head target)
    y_sign[t, h]       {0,1} up/down label
    sample_w[t]        recency + inverse-frequency sample weights

The final ``max(horizons)`` rows have undefined targets and are returned as NaN
so the caller can drop / embargo them explicitly.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-12
RISKMETRICS_LAMBDA = 0.94


@dataclass
class TargetConfig:
    horizons: tuple[int, ...] = (1, 5, 20)
    vol_lambda: float = RISKMETRICS_LAMBDA
    vol_floor_annual: float = 0.15  # don't divide by an absurdly small vol
    detrend: bool = True  # subtract a slow rolling drift per horizon
    drift_window: int = 252
    recency_halflife_days: float = 504.0  # ~2y half-life on sample weights
    clip_scaled: float = 8.0  # clip vol-normalized targets to +/- this


@dataclass
class TargetBundle:
    y_scaled: pd.DataFrame  # [T, n_horizons]  primary regression target (de-trended, vol-normed)
    y_raw: pd.DataFrame  # [T, n_horizons]  raw forward log-returns
    y_fwdvol: pd.DataFrame  # [T, n_horizons]  forward realized vol
    y_sign: pd.DataFrame  # [T, n_horizons]  up/down labels {0,1}
    y_drift: (
        pd.DataFrame
    )  # [T, n_horizons]  causal drift subtracted from y_scaled -- ADD BACK at predict
    sigma_ewma: pd.Series  # [T]  trailing EWMA vol used for the normalization
    sample_w: pd.Series  # [T]  sample weights
    horizons: tuple[int, ...]
    valid_mask: pd.Series  # [T]  True where every horizon target is defined

    def usable_index(self) -> pd.Index:
        return self.valid_mask.index[self.valid_mask.values]


def _ewma_vol(rets: pd.Series, lam: float) -> pd.Series:
    var = rets.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var)


def build_targets(close: pd.Series, cfg: TargetConfig | None = None) -> TargetBundle:
    cfg = cfg or TargetConfig()
    close = close.astype(float).dropna()
    logp = np.log(close.clip(lower=EPS))
    rets = logp.diff()

    # trailing vol known at time t (daily units)
    sigma_daily = _ewma_vol(rets, cfg.vol_lambda)
    vol_floor_daily = cfg.vol_floor_annual / np.sqrt(252.0)
    sigma_daily = sigma_daily.clip(lower=vol_floor_daily)

    y_scaled = pd.DataFrame(index=close.index)
    y_raw = pd.DataFrame(index=close.index)
    y_fwdvol = pd.DataFrame(index=close.index)
    y_sign = pd.DataFrame(index=close.index)
    y_drift = pd.DataFrame(index=close.index)

    for h in cfg.horizons:
        fwd = logp.shift(-h) - logp  # log(P[t+h]/P[t])
        drift = pd.Series(0.0, index=close.index)
        if cfg.detrend:
            # slow, causal drift estimate (per-day) scaled to the horizon
            drift = rets.rolling(cfg.drift_window, min_periods=60).mean().fillna(0.0) * h
        denom = (sigma_daily * np.sqrt(h)).clip(lower=vol_floor_daily * np.sqrt(h))
        scaled = (fwd - drift) / denom

        y_raw[f"h{h}"] = fwd
        y_drift[f"h{h}"] = drift
        y_scaled[f"h{h}"] = scaled.clip(-cfg.clip_scaled, cfg.clip_scaled)
        y_sign[f"h{h}"] = (fwd > 0).astype(float).where(fwd.notna())

        # realized vol over the *forward* window [t+1 .. t+h]
        # RMS of forward returns (well-defined even for h=1, where it is |ret|)
        fwd_rv = np.sqrt(rets.pow(2).rolling(h, min_periods=1).mean()).shift(-h)
        y_fwdvol[f"h{h}"] = fwd_rv

    # recency weights: exponential decay toward the past
    age_days = (close.index[-1] - close.index).days.to_numpy().astype(float)
    decay = np.log(2.0) / max(cfg.recency_halflife_days, 1.0)
    w = np.exp(-decay * age_days)
    sample_w = pd.Series(w / w.mean(), index=close.index)

    valid_mask = y_scaled.notna().all(axis=1)
    # also require the trailing vol to be defined (front warmup)
    valid_mask &= sigma_daily.notna()

    return TargetBundle(
        y_scaled=y_scaled,
        y_raw=y_raw,
        y_fwdvol=y_fwdvol,
        y_sign=y_sign,
        y_drift=y_drift,
        sigma_ewma=sigma_daily,
        sample_w=sample_w,
        horizons=cfg.horizons,
        valid_mask=valid_mask,
    )


def descale(y_scaled, sigma_daily, horizon: int, drift=0.0):
    """Inverse of the normalization: vol units -> raw log-return units.

    Must add back ``drift`` (TargetBundle.y_drift at the same row/horizon) to
    undo the de-trending applied when the target was built.
    """
    return np.asarray(y_scaled) * (np.asarray(sigma_daily) * np.sqrt(horizon)) + np.asarray(drift)


if __name__ == "__main__":
    from data_layer import DataConfig, build_panel, primary_close

    print("=" * 70)
    print("CARN-X Targets -- Self Test")
    print("=" * 70)

    panel = build_panel(DataConfig())
    close = primary_close(panel)
    tb = build_targets(close)

    print(f"\nrows              : {len(close)}")
    print(f"usable rows       : {int(tb.valid_mask.sum())}")
    print(f"horizons          : {tb.horizons}")
    print(f"EWMA vol (ann.)   : {tb.sigma_ewma.iloc[-1] * np.sqrt(252):.1%} (latest)")

    print("\nvol-normalized target stats (should be ~mean 0, ~std 1):")
    for c in tb.y_scaled.columns:
        s = tb.y_scaled[c].dropna()
        print(
            f"  {c:<5} mean={s.mean():+.3f}  std={s.std():.3f}  "
            f"skew={s.skew():+.2f}  kurt={s.kurt():+.2f}"
        )

    print("\nraw forward log-return stats:")
    for c in tb.y_raw.columns:
        s = tb.y_raw[c].dropna()
        print(f"  {c:<5} mean={s.mean():+.4f}  std={s.std():.4f}  P(up)={(s > 0).mean():.1%}")

    # a descale round-trip check
    h = 5
    last = tb.usable_index()[-1]
    rt = descale(
        tb.y_scaled.loc[last, f"h{h}"], tb.sigma_ewma.loc[last], h, tb.y_drift.loc[last, f"h{h}"]
    )
    raw = tb.y_raw.loc[last, f"h{h}"]
    print(
        f"\ndescale round-trip @ {last.date()} h={h}: {rt:+.5f} vs raw {raw:+.5f}  "
        f"(err {abs(rt - raw):.2e})"
    )
    assert abs(rt - raw) < 1e-4, "descale is not the exact inverse of the target transform"

    print("\n" + "=" * 70)
    print("Targets ready.")
    print("=" * 70)
