"""
CARN-X  --  Full Evaluation Runner
==================================
One command: walk-forward the neural model, walk-forward every baseline on the
same splits, and print a single comparison table + the detailed OOS report.

    .venv/bin/python run_evaluation.py            # default (minutes)
    .venv/bin/python run_evaluation.py --fast     # fewer folds / seeds, for a quick read
    .venv/bin/python run_evaluation.py --no-experts   # ablate the combinatorial-gate experts
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import sys
import time

import numpy as np
import pandas as pd

from walk_forward import WFConfig, run_walk_forward
from baselines import BaselineConfig, run_baselines
from evaluation import evaluate, print_report
from dataset import assemble
from forecast_net import DEFAULT_SEQ_FEATURES


def main() -> None:
    fast = "--fast" in sys.argv
    no_experts = "--no-experts" in sys.argv

    t0 = time.time()
    data = assemble(seq_feature_names=DEFAULT_SEQ_FEATURES)

    wf_cfg = WFConfig(
        initial_train=504,
        step=21,
        embargo=20,
        retrain_every=4 if fast else 3,
        ensemble=1 if fast else 3,
        seeds=(0,) if fast else (0, 1, 2),
        epochs=20 if fast else 45,
        patience=5 if fast else 7,
        num_threads=4,
        use_experts=not no_experts,
        tag="wf_fast" if fast else "wf_full",
    )
    print(f"\n>>> neural walk-forward  (experts={'off' if no_experts else 'on'})")
    wf = run_walk_forward(wf_cfg, data=data)

    print("\n>>> baselines walk-forward")
    b_cfg = BaselineConfig(initial_train=504, step=21, embargo=20)
    base_frames = run_baselines(b_cfg, data=data)

    # -------- comparison table --------
    frames = {"carnx_nn": wf.oos, **base_frames}
    rows = []
    for name, df in frames.items():
        if df is None or len(df) == 0:
            continue
        r = evaluate(df)
        for h, block in r["horizons"].items():
            s = block["statistical"]
            row = dict(
                model=name,
                h=h,
                n=s["n"],
                dir_acc=s["dir_acc_mean"],
                dir_p=s["dir_binom_p"],
                rmse_skill=s["rmse_skill"],
                nll=s["nll"],
                pit_ks=s["pit_ks"],
                cover90=s["coverage_90"],
            )
            if "economic" in block:
                e = block["economic"]["strategy"]
                row["sharpe"] = e.get("sharpe", np.nan)
                row["maxdd"] = e.get("max_drawdown", np.nan)
            rows.append(row)
    tab = pd.DataFrame(rows)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n" + "=" * 74)
    print("MODEL COMPARISON  (out-of-sample, same walk-forward splits)")
    print("=" * 74)
    for h in sorted(tab.h.unique()):
        print(f"\n--- horizon {h} day(s) ---")
        sub = tab[tab.h == h].drop(columns="h").set_index("model")
        print(sub.round(4).to_string())

    print("\n\n" + "=" * 74)
    print("DETAILED NEURAL-MODEL REPORT")
    print_report(evaluate(wf.oos))
    print(f"\ntotal wall time: {time.time() - t0:.0f}s")
    print(f"neural OOS predictions: {wf.run_path}/oos_predictions.parquet")


if __name__ == "__main__":
    main()
