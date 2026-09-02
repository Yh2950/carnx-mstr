"""
CARN-X – Orchestrator (Layer 8 / glue)
=======================================
Thin module that assembles all 7 pipeline layers plus the two new domain
modules (nav_premium, leverage_factorization) and exposes 3 public
functions the UI calls instead of touching the 7 layer classes directly:

    build_pipeline(config) -> PipelineBundle
    run_inference(bundle, price_series, nav_df=None, ...) -> DecisionOutput
    run_training(bundle, train_series, ..., progress_callback=...) -> TrainState
    run_backtest(bundle, price_series, nav_df=None, ..., progress_callback=...) -> BacktestResult

All date-alignment between the manual NAV CSV and the price series happens
here (align_nav_to_series) — every lower layer stays free of pandas-date /
NAV-specific knowledge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from combinatorial_gate import CombinatorialGate
from diagnosis_layer import DiagnosisLayer
from evaluation_backtest import BacktestResult, CARNXBacktester
from hybrid_neural_core import HybridNeuralCore
from inference_decision_layer import DecisionOutput, InferenceDecisionLayer, full_inference_pipeline
from leverage_factorization import LeverageFactorization
from nav_premium import NAVPremiumCalculator
from training_extreme_engine import (
    ExtremeEventHandler,
    LeverageOptimizer,
    TrainConfig,
    TrainingEngine,
    TrainState,
)


@dataclass
class PipelineConfig:
    diagnosis_kwargs: dict = field(default_factory=dict)
    gate_kwargs: dict = field(default_factory=dict)
    model_kwargs: dict = field(default_factory=lambda: {"input_dim": 1, "use_macro_fusion": False})
    decision_kwargs: dict = field(default_factory=dict)
    leverage_kwargs: dict = field(default_factory=dict)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    backtest_kwargs: dict = field(default_factory=dict)
    use_nav_macro: bool = False


@dataclass
class PipelineBundle:
    diag_layer: DiagnosisLayer
    gate: CombinatorialGate
    model: HybridNeuralCore
    extreme_handler: ExtremeEventHandler
    leverage_opt: LeverageOptimizer
    leverage_factorizer: LeverageFactorization | None
    decision_layer: InferenceDecisionLayer
    nav_calculator: NAVPremiumCalculator | None
    config: PipelineConfig
    training_engine: TrainingEngine | None = None


def build_pipeline(config: PipelineConfig | None = None) -> PipelineBundle:
    config = config or PipelineConfig()

    model_kwargs = dict(config.model_kwargs)
    if config.use_nav_macro:
        model_kwargs["use_macro_fusion"] = True
        model_kwargs.setdefault("macro_input_dim", 8)

    leverage_factorizer = (
        LeverageFactorization(**config.leverage_kwargs)
        if config.leverage_kwargs is not None
        else LeverageFactorization()
    )
    decision_layer = InferenceDecisionLayer(
        leverage_factorizer=leverage_factorizer, **config.decision_kwargs
    )

    return PipelineBundle(
        diag_layer=DiagnosisLayer(**config.diagnosis_kwargs),
        gate=CombinatorialGate(**config.gate_kwargs),
        model=HybridNeuralCore(**model_kwargs),
        extreme_handler=ExtremeEventHandler(),
        leverage_opt=LeverageOptimizer(),
        leverage_factorizer=leverage_factorizer,
        decision_layer=decision_layer,
        nav_calculator=NAVPremiumCalculator() if config.use_nav_macro else None,
        config=config,
    )


def align_nav_to_series(
    nav_df: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    calculator: NAVPremiumCalculator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (macro_matrix[N,8], regime_code_array[N]) aligned to price_index."""
    computed = calculator.compute(nav_df)
    macro_matrix = calculator.to_macro_series(computed, price_index)
    regime_code_array = macro_matrix[:, 3]  # slot 3 = premium_regime_code, see nav_premium.py
    return macro_matrix, regime_code_array


def run_inference(
    bundle: PipelineBundle,
    price_series: np.ndarray,
    nav_df: pd.DataFrame | None = None,
    price_dates: pd.DatetimeIndex | None = None,
    current_position: float = 0.0,
) -> DecisionOutput:
    diag = bundle.diag_layer.diagnose(price_series)
    routing = bundle.gate.route(diag).routing
    bundle.model.configure_from_routing(routing)

    macro_vec = None
    regime_code = 0.0
    if nav_df is not None and bundle.nav_calculator is not None:
        dates = (
            price_dates
            if price_dates is not None
            else pd.date_range(end=pd.Timestamp.now(), periods=len(price_series))
        )
        macro_matrix, regime_arr = align_nav_to_series(nav_df, dates, bundle.nav_calculator)
        macro_vec = macro_matrix[-1]
        regime_code = float(regime_arr[-1])

    return full_inference_pipeline(
        series=price_series,
        model=bundle.model,
        routing=routing,
        extreme_handler=bundle.extreme_handler,
        leverage_optimizer=bundle.leverage_opt,
        decision_layer=bundle.decision_layer,
        current_position=current_position,
        volatility_profile=diag.volatility,
        nav_premium_regime_code=regime_code,
        macro_features=macro_vec,
    )


def run_training(
    bundle: PipelineBundle,
    train_series: np.ndarray,
    val_series: np.ndarray | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> TrainState:
    diag = bundle.diag_layer.diagnose(train_series[-min(300, len(train_series)) :])
    routing = bundle.gate.route(diag).routing
    bundle.model.configure_from_routing(routing)

    bundle.training_engine = TrainingEngine(
        bundle.model, bundle.config.train_config, gate=bundle.gate
    )
    return bundle.training_engine.train(
        train_series, val_series, static_routing=routing, progress_callback=progress_callback
    )


def run_backtest(
    bundle: PipelineBundle,
    price_series: np.ndarray,
    nav_df: pd.DataFrame | None = None,
    price_dates: pd.DatetimeIndex | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    macro_features = None
    regime_series = None
    if nav_df is not None and bundle.nav_calculator is not None:
        dates = (
            price_dates
            if price_dates is not None
            else pd.date_range(end=pd.Timestamp.now(), periods=len(price_series))
        )
        macro_features, regime_series = align_nav_to_series(nav_df, dates, bundle.nav_calculator)

    backtester = CARNXBacktester(
        leverage_factorizer=bundle.leverage_factorizer,
        nav_premium_regime_series=regime_series,
        use_macro_fusion=bundle.config.use_nav_macro,
        **bundle.config.backtest_kwargs,
    )
    return backtester.run(
        price_series, macro_features=macro_features, progress_callback=progress_callback
    )


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Pipeline Orchestrator – Self Test (full synthetic demo)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    t = np.arange(260)
    series = 100 + 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 1.1, 260)
    series[150:170] -= 12.0

    config = PipelineConfig(
        train_config=TrainConfig(
            window=40, batch_size=16, epochs=2, log_every=50, early_stop_patience=3
        ),
        backtest_kwargs={"window": 40, "step": 8},
        use_nav_macro=False,
    )
    bundle = build_pipeline(config)

    print("\n[1/3] Training (2 quick epochs)...")
    state = run_training(bundle, series[:200], series[200:])
    print(f"  final train_loss={state.history[-1]['train_loss']:.5f}")

    print("\n[2/3] Inference...")
    decision = run_inference(bundle, series[-60:])
    print(f"  action={decision.action.name} position_scale={decision.position_scale:+.3f}")
    if decision.leverage_recommendation is not None:
        print(f"  leverage={decision.leverage_recommendation.recommended_leverage_ratio:.2f}x")

    print("\n[3/3] Backtest...")
    result = run_backtest(bundle, series)
    print(f"  total_return={result.metrics.total_return:+.2%} sharpe={result.metrics.sharpe:.3f}")

    print("\n" + "=" * 70)
    print("Pipeline orchestrator ready.")
    print("=" * 70)
