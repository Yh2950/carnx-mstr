"""
CARN-X – NAV Premium Module (Domain Layer, standalone)
=======================================================
Computes MSTR's premium/discount to the net asset value (NAV) of its
Bitcoin holdings, from a manually-maintained CSV. Zero dependency on the
7-layer neural pipeline — pure pandas/numpy, independently testable and
directly usable from the Streamlit UI's data page.

Definitions:
    market_cap    = shares_outstanding * stock_price_usd
    nav_total     = btc_holdings * btc_price_usd
    nav_per_share = nav_total / shares_outstanding
    mnav_ratio    = market_cap / nav_total
    premium       = stock_price_usd / nav_per_share - 1

CSV schema (user-maintained, irregular cadence allowed):
    date               YYYY-MM-DD
    btc_holdings       float  – total BTC held as of that date
    btc_price_usd      float  – BTC spot price on that date
    shares_outstanding float  – diluted/basic share count as of that date
    stock_price_usd    float  – MSTR closing price on that date
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "date",
    "btc_holdings",
    "btc_price_usd",
    "shares_outstanding",
    "stock_price_usd",
)

# regime_label thresholds, in premium units (e.g. 0.10 = +10% premium to NAV)
DEFAULT_REGIME_THRESHOLDS: dict[str, tuple[float, float]] = {
    "deep_discount": (-np.inf, -0.15),
    "discount": (-0.15, -0.02),
    "fair_value": (-0.02, 0.15),
    "elevated_premium": (0.15, 0.50),
    "extreme_premium": (0.50, np.inf),
}

# 8-slot macro vector, matching MacroMicroFusion's default macro_input_dim=8
MACRO_VECTOR_LABELS = (
    "premium_level",
    "premium_zscore",
    "premium_percentile",
    "premium_regime_code",
    "mean_reversion_signal",
    "premium_roc_5",
    "log_mnav_ratio",
    "data_staleness_normalized",
)


def load_nav_csv(csv_path: str) -> pd.DataFrame:
    """Load and validate the manually-maintained NAV CSV.

    Sorts ascending by date, drops duplicate dates (keeps the last row per
    date), and validates required columns/numeric dtypes.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"nav CSV missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    for col in REQUIRED_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")
    df = df.reset_index(drop=True)

    if (df["shares_outstanding"] <= 0).any():
        raise ValueError("shares_outstanding must be positive in every row")
    if (df["btc_price_usd"] <= 0).any() or (df["stock_price_usd"] <= 0).any():
        raise ValueError("btc_price_usd and stock_price_usd must be positive in every row")

    return df


@dataclass
class NAVPremiumSnapshot:
    date: pd.Timestamp
    market_cap: float
    nav_total: float
    nav_per_share: float
    mnav_ratio: float
    premium: float
    premium_zscore: float
    premium_percentile: float
    regime_label: str
    mean_reversion_signal: float
    premium_roc_5_normalized: float
    staleness_days: int


class NAVPremiumCalculator:
    """Computes MSTR NAV-premium metrics from a loaded NAV DataFrame."""

    def __init__(
        self,
        rolling_window: int = 60,
        regime_thresholds: dict[str, tuple[float, float]] | None = None,
    ):
        self.rolling_window = rolling_window
        self.regime_thresholds = regime_thresholds or DEFAULT_REGIME_THRESHOLDS

    def _regime_label(self, premium: float) -> str:
        if not np.isfinite(premium):
            return "fair_value"
        for label, (lo, hi) in self.regime_thresholds.items():
            if lo <= premium < hi:
                return label
        return "fair_value"

    def _regime_code(self, label: str) -> float:
        # ordered -2..2, matches the order of DEFAULT_REGIME_THRESHOLDS
        order = list(self.regime_thresholds.keys())
        idx = order.index(label) if label in order else len(order) // 2
        mid = (len(order) - 1) / 2.0
        return float((idx - mid) / mid) if mid > 0 else 0.0

    def compute(self, nav_df: pd.DataFrame) -> pd.DataFrame:
        """Adds derived NAV-premium columns to a copy of nav_df."""
        df = nav_df.copy()

        df["market_cap"] = df["shares_outstanding"] * df["stock_price_usd"]
        df["nav_total"] = df["btc_holdings"] * df["btc_price_usd"]
        df["nav_per_share"] = df["nav_total"] / df["shares_outstanding"]
        df["mnav_ratio"] = df["market_cap"] / df["nav_total"]
        df["premium"] = df["stock_price_usd"] / df["nav_per_share"] - 1.0

        roll = df["premium"].rolling(self.rolling_window, min_periods=5)
        roll_mean = roll.mean()
        roll_std = roll.std().replace(0.0, np.nan)
        df["premium_zscore"] = ((df["premium"] - roll_mean) / roll_std).fillna(0.0)
        df["premium_percentile"] = (
            df["premium"]
            .rolling(self.rolling_window, min_periods=5)
            .apply(lambda w: (w.rank(pct=True).iloc[-1]) if len(w) > 1 else 0.5, raw=False)
        )
        df["premium_percentile"] = df["premium_percentile"].fillna(0.5)

        df["regime_label"] = df["premium"].apply(self._regime_label)
        df["mean_reversion_signal"] = -np.tanh(df["premium_zscore"] / 2.0)
        df["premium_roc_5"] = df["premium"].diff(5).fillna(0.0)
        roc_std = df["premium_roc_5"].rolling(self.rolling_window, min_periods=5).std()
        roc_std = roc_std.bfill().ffill().fillna(1.0).replace(0.0, 1.0)
        df["premium_roc_5_normalized"] = (df["premium_roc_5"] / roc_std).clip(-1.0, 1.0)

        return df

    def latest_snapshot(
        self,
        nav_df_computed: pd.DataFrame,
        as_of: pd.Timestamp | None = None,
    ) -> NAVPremiumSnapshot:
        """Returns the most recent (or as-of) computed row as a snapshot."""
        df = nav_df_computed
        if as_of is not None:
            df = df[df["date"] <= as_of]
        if len(df) == 0:
            raise ValueError("no NAV data available at or before the requested date")
        row = df.iloc[-1]

        query_date = as_of if as_of is not None else pd.Timestamp.now().normalize()
        staleness = int((query_date - row["date"]).days) if query_date >= row["date"] else 0

        return NAVPremiumSnapshot(
            date=row["date"],
            market_cap=float(row["market_cap"]),
            nav_total=float(row["nav_total"]),
            nav_per_share=float(row["nav_per_share"]),
            mnav_ratio=float(row["mnav_ratio"]),
            premium=float(row["premium"]),
            premium_zscore=float(row["premium_zscore"]),
            premium_percentile=float(row["premium_percentile"]),
            regime_label=str(row["regime_label"]),
            mean_reversion_signal=float(row["mean_reversion_signal"]),
            premium_roc_5_normalized=float(row["premium_roc_5_normalized"]),
            staleness_days=staleness,
        )

    def to_macro_vector(self, snapshot: NAVPremiumSnapshot) -> np.ndarray:
        """Packs a snapshot into the fixed 8-slot macro vector (float32)."""
        regime_code = self._regime_code(snapshot.regime_label)
        vec = np.array(
            [
                np.clip(snapshot.premium, -1.0, 3.0),
                np.clip(snapshot.premium_zscore, -4.0, 4.0) / 4.0,
                2.0 * snapshot.premium_percentile - 1.0,
                regime_code,
                snapshot.mean_reversion_signal,
                snapshot.premium_roc_5_normalized,
                np.clip(np.log(max(snapshot.mnav_ratio, 1e-6)), -1.0, 1.0),
                min(snapshot.staleness_days / 30.0, 1.0),
            ],
            dtype=np.float32,
        )
        return vec

    def to_macro_series(
        self,
        nav_df_computed: pd.DataFrame,
        target_dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Forward-fills the per-date macro vector onto an arbitrary date index.

        Returns shape (len(target_dates), 8), float32. Dates before the first
        NAV row use the first available row (staleness grows accordingly via
        latest_snapshot's as_of logic).
        """
        df = nav_df_computed.sort_values("date")
        out = np.zeros((len(target_dates), 8), dtype=np.float32)
        for i, dt in enumerate(pd.DatetimeIndex(target_dates)):
            snap = self.latest_snapshot(df, as_of=dt)
            out[i] = self.to_macro_vector(snap)
        return out


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X NAV Premium Module – Self Test")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n = 180
    dates = pd.date_range("2024-01-01", periods=n, freq="7D")
    btc_price = 40000 + np.cumsum(rng.normal(200, 1500, n))
    btc_holdings = 190000 + np.cumsum(rng.integers(0, 3000, n))
    shares = 200_000_000 + np.cumsum(rng.integers(0, 500_000, n))
    nav_per_share_true = (btc_holdings * btc_price) / shares
    premium_true = 0.15 + 0.3 * np.sin(np.arange(n) / 12) + rng.normal(0, 0.05, n)
    stock_price = nav_per_share_true * (1.0 + premium_true)

    synthetic = pd.DataFrame(
        {
            "date": dates,
            "btc_holdings": btc_holdings,
            "btc_price_usd": btc_price,
            "shares_outstanding": shares,
            "stock_price_usd": stock_price,
        }
    )

    calc = NAVPremiumCalculator(rolling_window=26)
    computed = calc.compute(synthetic)
    snap = calc.latest_snapshot(computed)

    print(f"\nLatest snapshot ({snap.date.date()}):")
    print(f"  market_cap          = ${snap.market_cap:,.0f}")
    print(f"  nav_total           = ${snap.nav_total:,.0f}")
    print(f"  nav_per_share       = ${snap.nav_per_share:,.2f}")
    print(f"  mnav_ratio          = {snap.mnav_ratio:.3f}")
    print(f"  premium             = {snap.premium:+.2%}")
    print(f"  premium_zscore      = {snap.premium_zscore:+.2f}")
    print(f"  premium_percentile  = {snap.premium_percentile:.2f}")
    print(f"  regime_label        = {snap.regime_label}")
    print(f"  mean_reversion_sig  = {snap.mean_reversion_signal:+.3f}")
    print(f"  staleness_days      = {snap.staleness_days}")

    macro_vec = calc.to_macro_vector(snap)
    print(f"\nMacro vector (8 slots): {np.round(macro_vec, 3)}")

    price_dates = pd.date_range(dates.min(), dates.max(), freq="1D")
    macro_series = calc.to_macro_series(computed, price_dates)
    print(f"\nAligned macro series shape: {macro_series.shape}")
    assert macro_series.shape == (len(price_dates), 8)

    print("\n" + "=" * 70)
    print("NAV Premium Module ready.")
    print("=" * 70)
