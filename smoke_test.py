"""
CARN-X – Smoke Test
===================
Standing regression check for the full 7-layer pipeline. Imports every
layer in pipeline order and drives one short synthetic pass end to end:

    DiagnosisLayer -> CombinatorialGate -> HybridNeuralCore.forward
    -> ExtremeEventHandler -> LeverageOptimizer -> InferenceDecisionLayer
    -> CARNXBacktester (short walk-forward run)

Run after every phase of the extension work to confirm nothing broke.
Prints "OK" and exits 0 on success; raises on any failure.
"""

from __future__ import annotations

import numpy as np
import torch

from combinatorial_gate import CombinatorialGate
from diagnosis_layer import DiagnosisLayer
from evaluation_backtest import CARNXBacktester
from hybrid_neural_core import HybridNeuralCore
from inference_decision_layer import InferenceDecisionLayer, full_inference_pipeline
from training_extreme_engine import ExtremeEventHandler, LeverageOptimizer


def make_synthetic_series(n: int = 220, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    series = 0.015 * t + 2.5 * np.sin(2 * np.pi * t / 14) + rng.normal(0, 0.8, size=n)
    series[110:125] -= 7.0  # injected crash
    return series


def run_smoke_test() -> None:
    series = make_synthetic_series()

    # 1. Diagnosis
    diag = DiagnosisLayer().diagnose(series)
    assert diag.basic.n_obs == len(series)

    # 2. Combinatorial Gate
    gate_result = CombinatorialGate(temperature=0.5).route(diag)
    routing = gate_result.routing
    assert routing.embed_dim > 0

    # 3. Hybrid Neural Core forward pass
    model = HybridNeuralCore(input_dim=1)
    model.configure_from_routing(routing)
    x = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        out = model(x, routing=routing, return_aux=True)
    assert "forecast" in out and "hidden" in out
    print(
        f"  [1-3] diagnosis+gate+core OK — dominant_case={routing.dominant_case}, "
        f"forecast_shape={tuple(out['forecast'].shape)}"
    )

    # 4. Extreme handler + Leverage optimizer
    model_severity = float(out["severity"].mean()) if "severity" in out else None
    extreme_handler = ExtremeEventHandler()
    extreme = extreme_handler.evaluate(series, routing, model_severity)
    leverage_opt = LeverageOptimizer()
    leverage = None
    if "leverage_action" in out:
        leverage = leverage_opt.suggest(out["leverage_action"], extreme, routing)
    print(
        f"  [4]   extreme+leverage OK — is_extreme={extreme.is_extreme}, "
        f"severity={extreme.severity:.3f}, leverage={'set' if leverage else 'skipped (gate below threshold)'}"
    )

    # 5. Inference & Decision layer (full_inference_pipeline exercises the whole chain again)
    decision = full_inference_pipeline(
        series=series,
        model=model,
        routing=routing,
        extreme_handler=extreme_handler,
        leverage_optimizer=leverage_opt,
        decision_layer=InferenceDecisionLayer(),
        current_position=0.0,
    )
    assert decision.action is not None
    print(
        f"  [5]   decision layer OK — action={decision.action.name}, "
        f"position_scale={decision.position_scale:+.3f}"
    )

    # 6. Short walk-forward backtest (small window/step so it stays quick)
    backtester = CARNXBacktester(window=40, step=8, transaction_cost=0.0005)
    result = backtester.run(series, verbose=False)
    assert result.equity_curve.shape[0] > 1
    print(
        f"  [6]   backtest OK — n_points={len(result.equity_curve)}, "
        f"total_return={result.metrics.total_return:+.2%}, sharpe={result.metrics.sharpe:.3f}"
    )

    print("OK")


if __name__ == "__main__":
    run_smoke_test()
