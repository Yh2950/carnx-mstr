"""
CARN-X  --  Windowing & Fold Scaling
====================================
Bridges the causal feature matrix (features.py) and multi-horizon targets
(targets.py) to the tensors ForecastNet consumes.

A *sample* anchored at row ``i`` (a trading day) is:
    x_seq : the curated sequence features over rows [i-W+1 .. i]   -> [W, C_seq]
    x_tab : the full engineered feature vector at row i            -> [F]
    y_*   : the multi-horizon targets defined at row i             -> [H]

``FoldScaler`` is fit on the training rows of a fold ONLY and then applied to
train/val/test alike -- no leakage of test-period statistics.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_layer import DataConfig, build_panel, primary_close
from features import FeatureConfig, build_features
from targets import TargetConfig, build_targets, TargetBundle


@dataclass
class AssembledData:
    X: pd.DataFrame  # [T, F]  full causal features
    seq_cols: list[str]  # subset of X used by the temporal encoder
    tb: TargetBundle
    close: pd.Series
    index: pd.DatetimeIndex  # feature rows in use (see require_targets)
    feature_meta: object
    trainable_mask: np.ndarray | None = None  # bool over `index`: every-horizon target defined

    def trainable_positions(self) -> np.ndarray:
        if self.trainable_mask is None:
            return np.arange(len(self.index))
        return np.where(self.trainable_mask)[0]


def assemble(
    feature_cfg: FeatureConfig | None = None,
    target_cfg: TargetConfig | None = None,
    data_cfg: DataConfig | None = None,
    seq_feature_names: Sequence[str] = (),
    require_targets: bool = True,
) -> AssembledData:
    """Assemble features + targets on a shared index.

    require_targets=True  (evaluation): index = feature rows that also have every
        forward-horizon target defined -- the last ~max(horizon) bars drop off.
    require_targets=False (production inference): keep every feature row; the
        tail bars have NaN targets and are excluded from training via
        ``trainable_mask`` but can still be predicted on.
    """
    data_cfg = data_cfg or DataConfig()
    feature_cfg = feature_cfg or FeatureConfig()
    target_cfg = target_cfg or TargetConfig()

    panel = build_panel(data_cfg)
    close = primary_close(panel, data_cfg)
    X, meta = build_features(panel, feature_cfg)
    tb = build_targets(close, target_cfg)

    if require_targets:
        common = X.index.intersection(tb.usable_index())
    else:
        # every feature row that the price series also covers
        common = X.index.intersection(close.index)
    X = X.loc[common]
    trainable_mask = tb.valid_mask.reindex(common).fillna(False).to_numpy()

    seq_cols = [c for c in seq_feature_names if c in X.columns]
    missing = [c for c in seq_feature_names if c not in X.columns]
    if missing:
        print(
            f"[dataset] {len(missing)} requested seq features absent, using {len(seq_cols)}: "
            f"missing={missing[:6]}{'...' if len(missing) > 6 else ''}"
        )
    if not seq_cols:
        # fall back to the primary return + vol so the encoder always has input
        seq_cols = [c for c in ("mstr_ret_1", "mstr_rv_20", "btc_ret_1") if c in X.columns]

    return AssembledData(
        X=X,
        seq_cols=seq_cols,
        tb=tb,
        close=close,
        index=pd.DatetimeIndex(common),
        feature_meta=meta,
        trainable_mask=trainable_mask,
    )


class FoldScaler:
    """Standardize seq and tab feature blocks with TRAIN-ONLY statistics.

    Two stages, both fit on training rows only (no test-period leakage):
      1. winsorize each column to its train [q, 1-q] range
      2. standardize to train mean / std, then hard-clip to +/- ``clip`` sigma
    """

    def __init__(self, clip: float = 8.0, winsor_q: float = 0.002):
        self.clip = clip
        self.winsor_q = winsor_q
        self.seq_lo_ = self.seq_hi_ = self.seq_mean_ = self.seq_std_ = None
        self.tab_lo_ = self.tab_hi_ = self.tab_mean_ = self.tab_std_ = None

    def fit(self, X_train: pd.DataFrame, seq_cols: list[str]) -> FoldScaler:
        seq = X_train[seq_cols].to_numpy(np.float64)
        tab = X_train.to_numpy(np.float64)
        q = self.winsor_q
        self.seq_lo_, self.seq_hi_ = np.quantile(seq, q, axis=0), np.quantile(seq, 1 - q, axis=0)
        self.tab_lo_, self.tab_hi_ = np.quantile(tab, q, axis=0), np.quantile(tab, 1 - q, axis=0)
        seq_w = np.clip(seq, self.seq_lo_, self.seq_hi_)
        tab_w = np.clip(tab, self.tab_lo_, self.tab_hi_)
        self.seq_mean_, self.seq_std_ = seq_w.mean(0), seq_w.std(0) + 1e-8
        self.tab_mean_, self.tab_std_ = tab_w.mean(0), tab_w.std(0) + 1e-8
        return self

    def transform_seq(self, a: np.ndarray) -> np.ndarray:
        z = (np.clip(a, self.seq_lo_, self.seq_hi_) - self.seq_mean_) / self.seq_std_
        return np.clip(z, -self.clip, self.clip)

    def transform_tab(self, a: np.ndarray) -> np.ndarray:
        z = (np.clip(a, self.tab_lo_, self.tab_hi_) - self.tab_mean_) / self.tab_std_
        return np.clip(z, -self.clip, self.clip)


class WindowedDataset(Dataset):
    def __init__(
        self,
        data: AssembledData,
        row_positions: np.ndarray,  # integer positions into data.X / data.index
        window: int,
        scaler: FoldScaler,
    ):
        self.window = window
        self.scaler = scaler
        self.horizons = list(data.tb.horizons)

        self._seq_all = scaler.transform_seq(data.X[data.seq_cols].to_numpy(np.float64)).astype(
            np.float32
        )
        self._tab_all = scaler.transform_tab(data.X.to_numpy(np.float64)).astype(np.float32)

        idx = data.index
        self._y_scaled = data.tb.y_scaled.loc[idx].to_numpy(np.float32)
        self._y_sign = data.tb.y_sign.loc[idx].to_numpy(np.float32)
        self._y_fwdvol = data.tb.y_fwdvol.loc[idx].to_numpy(np.float32)
        self._sigma = data.tb.sigma_ewma.loc[idx].to_numpy(np.float32)
        # training target for the vol head: log(fwd_vol / trailing_daily_vol), ~N(0, .)
        self._y_fwdvol_lr = np.clip(
            np.log((self._y_fwdvol + 1e-6) / (self._sigma[:, None] + 1e-6)), -3.0, 3.0
        ).astype(np.float32)
        self._sample_w = data.tb.sample_w.loc[idx].to_numpy(np.float32)
        self._raw = data.tb.y_raw.loc[idx].to_numpy(np.float32)
        self._drift = data.tb.y_drift.loc[idx].to_numpy(np.float32)

        # only positions with a full window of history behind them
        self.positions = np.array([p for p in row_positions if p >= window - 1], dtype=int)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        p = self.positions[k]
        s0 = p - self.window + 1
        return {
            "x_seq": torch.from_numpy(self._seq_all[s0 : p + 1]),
            "x_tab": torch.from_numpy(self._tab_all[p]),
            "y_scaled": torch.from_numpy(self._y_scaled[p]),
            "y_sign": torch.from_numpy(self._y_sign[p]),
            "y_fwdvol": torch.from_numpy(self._y_fwdvol[p]),
            "y_fwdvol_lr": torch.from_numpy(self._y_fwdvol_lr[p]),
            "sigma": torch.tensor(self._sigma[p]),
            "sample_w": torch.tensor(self._sample_w[p]),
            "y_raw": torch.from_numpy(self._raw[p]),
            "y_drift": torch.from_numpy(self._drift[p]),
            "pos": torch.tensor(p),
        }


if __name__ == "__main__":
    from forecast_net import DEFAULT_SEQ_FEATURES

    print("=" * 70)
    print("CARN-X Dataset -- Self Test")
    print("=" * 70)

    data = assemble(seq_feature_names=DEFAULT_SEQ_FEATURES)
    print(
        f"\ncommon rows   : {len(data.index)}  ({data.index.min().date()} -> {data.index.max().date()})"
    )
    print(f"tab features  : {data.X.shape[1]}")
    print(f"seq features  : {len(data.seq_cols)} -> {data.seq_cols}")

    W = 64
    n = len(data.index)
    split = int(n * 0.8)
    scaler = FoldScaler().fit(data.X.iloc[:split], data.seq_cols)

    train_ds = WindowedDataset(data, np.arange(0, split), W, scaler)
    test_ds = WindowedDataset(data, np.arange(split, n), W, scaler)
    print(f"\ntrain samples : {len(train_ds)}   test samples : {len(test_ds)}")

    b = train_ds[0]
    for k, v in b.items():
        print(f"  {k:<10} {tuple(v.shape)}  dtype={v.dtype}")
    assert torch.isfinite(b["x_seq"]).all() and torch.isfinite(b["x_tab"]).all()

    print("\n" + "=" * 70)
    print("Dataset ready.")
    print("=" * 70)
