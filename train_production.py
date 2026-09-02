"""
CARN-X  --  Deep Production Training
===================================
Train the production model at the highest quality tier (``config.production_xl``:
9-seed ensemble, long schedule), save it, reload it, and print a full acceptance
report: per-horizon predictive distribution, tail probabilities, and a
Monte-Carlo price cone under every drift mode.

    .venv/bin/python train_production.py                # full (9 seeds, slow)
    .venv/bin/python train_production.py --preset production   # 5 seeds
    .venv/bin/python train_production.py --eval         # also run walk-forward

The real checkpoint is only overwritten once training + reload + sanity all pass.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

from config import production, production_xl
from dataset import assemble
from forecast_net import DEFAULT_SEQ_FEATURES
from inference import (
    fit_production_model,
    save_production_model,
    load_production_model,
    predict_distribution,
    tail_probability,
    monte_carlo_paths,
    default_model_path,
)


def _fmt_dist(dist: pd.DataFrame) -> str:
    cols = [
        "horizon",
        "expected_move_pct",
        "p_up",
        "ann_vol_implied",
        "nu",
        "ret_q05_pct",
        "ret_q95_pct",
    ]
    return dist[cols].round(3).to_string(index=False)


def main() -> None:
    t0 = time.time()
    also_eval = "--eval" in sys.argv
    cfg = production() if "--preset" in sys.argv and "production" in sys.argv else production_xl()
    cfg.sync()

    print("=" * 74)
    print("CARN-X  --  DEEP PRODUCTION TRAINING")
    print("=" * 74)
    print(
        f"ensemble={cfg.inference.ensemble}  seeds={cfg.inference.seeds}  "
        f"epochs={cfg.inference.epochs}  patience={cfg.inference.patience}"
    )
    print(f"experts={cfg.use_combinatorial_gate}  loss_schedule={cfg.use_loss_schedule}")

    # ------------------------------------------------------------------ data
    data = assemble(
        seq_feature_names=cfg.model.seq_feature_names or DEFAULT_SEQ_FEATURES,
        target_cfg=cfg.targets,
        require_targets=False,
    )
    print(
        f"\ndata: {len(data.index)} rows  "
        f"{data.index.min().date()} -> {data.index.max().date()}  "
        f"({data.X.shape[1]} tab features, {len(data.seq_cols)} seq series, "
        f"{len(data.trainable_positions())} trainable)"
    )

    # -------------------------------------------------------------- training
    def _progress(i: int, n: int) -> None:
        print(f"  [{time.time() - t0:6.0f}s]  trained ensemble member {i}/{n}")

    pm = fit_production_model(data, cfg.inference, progress=_progress)
    print(
        f"\ntrained {len(pm.nets)} nets in {time.time() - t0:.0f}s  "
        f"(train_rows={pm.train_rows}, data<= {pm.data_date_max})"
    )

    # ------------------------------------------------- save -> reload -> check
    tmp_path = default_model_path() + ".new"
    save_production_model(pm, n_tab_features=data.X.shape[1], path=tmp_path)
    pm2 = load_production_model(data, path=tmp_path)

    d1 = predict_distribution(pm, data)
    d2 = predict_distribution(pm2, data)
    assert np.allclose(d1["mu_scaled"], d2["mu_scaled"], atol=1e-5), "reload changed predictions!"
    for _, r in d1.iterrows():
        assert 0.0 <= r.p_up <= 1.0
        assert r.nu >= 2.0 and r.ann_vol_implied > 0
        assert r.ret_q05_pct < r.expected_move_pct < r.ret_q95_pct, "quantiles out of order"
    print("\nreload + sanity: OK")

    # promote to the real checkpoint
    import os

    final = default_model_path()
    os.replace(tmp_path, final)
    print(f"checkpoint written: {final}")

    # ---------------------------------------------------- acceptance report
    last_price = float(data.close.reindex(data.index).iloc[-1])
    print("\n" + "-" * 74)
    print(f"latest close (split-adjusted): ${last_price:,.2f}   as of {data.index.max().date()}")
    print("predictive distribution:")
    print(_fmt_dist(d1))
    for _, r in d1.iterrows():
        p_dd10 = tail_probability(r, -10.0)
        p_up20 = 1.0 - tail_probability(r, 20.0)
        print(f"  h={int(r.horizon):2d}: P(ret<=-10%)={p_dd10:.1%}   P(ret>=+20%)={p_up20:.1%}")

    print(f"\nMonte-Carlo 252-day cone by drift mode (target = +40% = ${last_price * 1.4:,.0f}):")
    tgt = last_price * 1.4
    hdr = (
        f"{'mode':<13} {'E[S_T]':>10} {'median':>10} {'P(up)':>7} "
        f"{'mu_ann':>8} {'VaR5':>9} {'CVaR5':>9} {'P(touch)':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for mode, kw in [
        ("model", {}),
        ("risk_neutral", {"risk_free_rate": 0.045}),
        ("custom", {"custom_drift_annual": 0.40}),
        ("historical", {}),
    ]:
        mc = monte_carlo_paths(
            pm, data, horizon=252, n_paths=60000, drift_mode=mode, target_price=tgt, **kw
        )
        print(
            f"{mode:<13} {mc.expected_price:>10.2f} {mc.median_price:>10.2f} "
            f"{mc.prob_up:>7.1%} {mc.mu_annual:>+8.1%} {mc.var_5_price:>9.2f} "
            f"{mc.es_5_price:>9.2f} {mc.p_touch_up:>9.1%}"
        )

    # martingale acceptance test under Q
    r = 0.045
    mcq = monte_carlo_paths(
        pm, data, horizon=252, n_paths=120000, drift_mode="risk_neutral", risk_free_rate=r
    )
    grow = mcq.expected_price / mcq.last_price
    ok = abs(grow - np.exp(r)) < 0.02
    print(
        f"\nmartingale check (Q, 120k paths): E[S_T]/S0 = {grow:.4f}  "
        f"target e^r = {np.exp(r):.4f}  -> {'PASS' if ok else 'FAIL'}"
    )
    assert ok, "risk-neutral martingale property violated"

    # ----------------------------------------------------- optional eval
    if also_eval:
        from walk_forward import run_walk_forward
        from evaluation import evaluate, print_report

        print("\n" + "=" * 74)
        print("WALK-FORWARD OUT-OF-SAMPLE EVALUATION  (honest skill / calibration)")
        print("=" * 74)
        wf = run_walk_forward(
            cfg.walk_forward,
            data=assemble(seq_feature_names=cfg.model.seq_feature_names, target_cfg=cfg.targets),
        )
        rep = evaluate(wf.oos)
        print_report(rep)

    print(f"\ntotal wall time: {time.time() - t0:.0f}s")
    print("=" * 74)
    print("DEEP PRODUCTION TRAINING COMPLETE")
    print("=" * 74)


if __name__ == "__main__":
    main()
