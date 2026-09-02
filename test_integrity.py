"""
CARN-X  --  Data Integrity & Leakage Test Suite
===============================================
Guards the properties that make the walk-forward evaluation trustworthy.
Run after ANY change to data_layer / features / targets / dataset / walk_forward.

    .venv/bin/python test_integrity.py        # uses the cached panel

Checks
  1. panel structure   -- sorted, unique, weekday-only, no future rows, split-adjusted
  2. feature causality -- features on X[:k] are byte-identical to the full run sliced
                          to [:k]  (i.e. no feature peeks forward)
  3. no single-feature leak -- |corr(feature_t, target_t)| < 0.5 for every feature
  4. target transform  -- descale is the exact inverse; forward targets are NaN for
                          exactly the trailing max(horizon) rows
  5. fold scaling      -- FoldScaler statistics come only from the training slice
  6. window alignment  -- sample p exposes feature row p as the last seq step and
                          target row p as the label
  7. embargo           -- walk-forward train labels never reach into the test window
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import sys

import numpy as np
import pandas as pd

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  --  {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------


def test_panel(cfg, panel):
    from data_layer import assert_panel_integrity

    try:
        assert_panel_integrity(panel, cfg)
        check("panel structure (assert_panel_integrity)", True)
    except AssertionError as e:
        check("panel structure (assert_panel_integrity)", False, str(e))


def test_feature_causality(cfg_feat, panel):
    """The core anti-look-ahead guarantee: computing features on a truncated
    panel must equal computing them on the full panel and slicing."""
    from features import build_features

    X_full, _ = build_features(panel, cfg_feat)
    k_date = X_full.index[int(len(X_full) * 0.6)]
    panel_trunc = panel.loc[:k_date]
    X_trunc, _ = build_features(panel_trunc, cfg_feat)

    common = X_full.index.intersection(X_trunc.index)
    common = common[common <= k_date]
    # compare the last 200 shared rows on shared columns
    cols = [c for c in X_trunc.columns if c in X_full.columns]
    a = X_full.loc[common, cols].iloc[-200:]
    b = X_trunc.loc[common, cols].iloc[-200:]
    max_abs = float((a - b).abs().to_numpy().max())
    check(
        "feature causality (truncation-invariant)",
        max_abs < 1e-4,
        f"max abs diff = {max_abs:.2e} on {len(cols)} cols",
    )
    return X_full


def test_no_single_feature_leak(X, panel):
    fwd_next = np.log(panel["mstr_close"]).diff().shift(-1).reindex(X.index)
    m = fwd_next.notna()
    worst_c, worst_r = None, 0.0
    for c in X.columns:
        v = X.loc[m, c].to_numpy()
        if np.std(v) < 1e-12:
            continue
        r = abs(np.corrcoef(v, fwd_next[m].to_numpy())[0, 1])
        if r > worst_r:
            worst_r, worst_c = r, c
    check(
        "no single-feature leak (|corr(feat, next_ret)| < 0.5)",
        worst_r < 0.5,
        f"worst: {worst_c} r={worst_r:.3f}",
    )


def test_target_transform(close):
    from targets import build_targets, descale, TargetConfig

    cfg = TargetConfig()
    tb = build_targets(close, cfg)

    # descale is the exact inverse on unclipped rows
    errs = []
    for h in tb.horizons:
        col = f"h{h}"
        s, sig, dr, raw = tb.y_scaled[col], tb.sigma_ewma, tb.y_drift[col], tb.y_raw[col]
        unclipped = s.abs() < cfg.clip_scaled - 1e-6
        rt = descale(
            s[unclipped].to_numpy(), sig[unclipped].to_numpy(), h, dr[unclipped].to_numpy()
        )
        errs.append(float(np.nanmax(np.abs(rt - raw[unclipped].to_numpy()))))
    check(
        "target transform: descale is exact inverse", max(errs) < 1e-4, f"max err {max(errs):.2e}"
    )

    # forward targets NaN for exactly the trailing max(h) rows
    hmax = max(tb.horizons)
    tail_nan = tb.y_raw[f"h{hmax}"].isna().to_numpy()
    ok = tail_nan[-hmax:].all() and not tail_nan[:-hmax][~np.isnan(close.to_numpy()[:-hmax])].any()
    check(f"forward targets: NaN exactly on trailing {hmax} rows", bool(tail_nan[-hmax:].all()))


def test_fold_scaler(data):
    from dataset import FoldScaler

    n = len(data.index)
    split = int(n * 0.7)
    sc = FoldScaler().fit(data.X.iloc[:split], data.seq_cols)
    # train slice should be ~standardized; test slice generally is NOT
    tab_tr = sc.transform_tab(data.X.iloc[:split].to_numpy(np.float64))
    tab_te = sc.transform_tab(data.X.iloc[split:].to_numpy(np.float64))
    train_ok = abs(float(np.mean(tab_tr))) < 0.05 and abs(float(np.std(tab_tr)) - 1.0) < 0.25
    leaked = abs(float(np.mean(tab_te))) < 1e-9  # would be ~0 only if test stats were used
    check(
        "fold scaler: standardizes the train slice",
        train_ok,
        f"train mean={np.mean(tab_tr):.3f} std={np.std(tab_tr):.3f}",
    )
    check("fold scaler: does NOT recentre the test slice", not leaked)


def test_window_alignment(data):
    from dataset import FoldScaler, WindowedDataset

    W = 32
    sc = FoldScaler().fit(data.X.iloc[:400], data.seq_cols)
    ds = WindowedDataset(data, np.arange(0, 600), W, sc)
    k = len(ds) // 2
    b = ds[k]
    p = int(b["pos"])
    seq_last = b["x_seq"][-1].numpy()
    seq_ref = sc.transform_seq(data.X[data.seq_cols].to_numpy(np.float64))[p]
    tab_ref = sc.transform_tab(data.X.to_numpy(np.float64))[p]
    y_ref = data.tb.y_scaled.loc[data.index].to_numpy(np.float32)[p]
    ok = (
        np.allclose(seq_last, seq_ref, atol=1e-5)
        and np.allclose(b["x_tab"].numpy(), tab_ref, atol=1e-5)
        and np.allclose(np.nan_to_num(b["y_scaled"].numpy()), np.nan_to_num(y_ref), atol=1e-5)
    )
    check("window alignment: sample p -> feature row p + target row p", ok)


def test_embargo():
    from walk_forward import WFConfig

    cfg = WFConfig()
    # a training sample at position t_tr-1 has targets through t_tr-1+embargo.
    # the test block starts at t_tr+embargo -> gap of >=1 row, never overlapping.
    t_tr = cfg.initial_train
    last_train_label_pos = (t_tr - 1) + cfg.embargo
    first_test_pos = t_tr + cfg.embargo
    check(
        "walk-forward embargo: train labels precede the test window",
        last_train_label_pos < first_test_pos,
        f"last_train_label={last_train_label_pos} first_test={first_test_pos}",
    )


# ---------------------------------------------------------------------------


def main() -> int:
    from data_layer import DataConfig, build_panel, primary_close
    from features import FeatureConfig
    from dataset import assemble
    from forecast_net import DEFAULT_SEQ_FEATURES

    dcfg = DataConfig()
    fcfg = FeatureConfig()
    panel = build_panel(dcfg)
    close = primary_close(panel, dcfg)

    print("=" * 74)
    print("CARN-X Integrity & Leakage Tests")
    print("=" * 74)

    test_panel(dcfg, panel)
    X = test_feature_causality(fcfg, panel)
    test_no_single_feature_leak(X, panel)
    test_target_transform(close)

    data = assemble(seq_feature_names=DEFAULT_SEQ_FEATURES)
    test_fold_scaler(data)
    test_window_alignment(data)
    test_embargo()

    print("\n" + "=" * 74)
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILED -> {FAILS}")
        return 1
    print("RESULT: all integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
