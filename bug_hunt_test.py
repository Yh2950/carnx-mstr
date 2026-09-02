"""
CARN-X – Bug-hunting test suite (ad-hoc, not part of the phase plan)
======================================================================
Exercises edge cases the basic smoke test doesn't cover: macro fusion
enabled end-to-end, extreme leverage inputs, short/degenerate series,
and the full pipeline.py orchestrator (train -> infer -> backtest).
Each test prints PASS/FAIL and any exception with full context.
"""

import traceback

import numpy as np
import torch

results = []


def test(name):
    def deco(fn):
        try:
            fn()
            results.append((name, "PASS", None))
            print(f"[PASS] {name}")
        except Exception as e:
            results.append((name, "FAIL", f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
        return fn

    return deco


@test("HybridNeuralCore forward with use_macro_fusion=True")
def _():
    from combinatorial_gate import CombinatorialGate
    from diagnosis_layer import DiagnosisLayer
    from hybrid_neural_core import HybridNeuralCore

    rng = np.random.default_rng(1)
    series = 100 + np.cumsum(rng.normal(0, 1, 150))
    diag = DiagnosisLayer().diagnose(series)
    routing = CombinatorialGate().route(diag).routing

    model = HybridNeuralCore(input_dim=1, use_macro_fusion=True, macro_input_dim=8)
    model.configure_from_routing(routing)
    x = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    macro = torch.randn(1, 8)
    out = model(x, routing=routing, macro_features=macro, return_aux=True)
    assert "fusion_strength" in out, (
        "fusion_strength missing from output when macro fusion is enabled"
    )
    assert out["forecast"].shape == (1, 150, 1)


@test(
    "HybridNeuralCore forward with use_macro_fusion=True and NO macro_features (missing-macro path)"
)
def _():
    from combinatorial_gate import CombinatorialGate
    from diagnosis_layer import DiagnosisLayer
    from hybrid_neural_core import HybridNeuralCore

    rng = np.random.default_rng(2)
    series = 100 + np.cumsum(rng.normal(0, 1, 120))
    diag = DiagnosisLayer().diagnose(series)
    routing = CombinatorialGate().route(diag).routing

    model = HybridNeuralCore(input_dim=1, use_macro_fusion=True, macro_input_dim=8)
    model.configure_from_routing(routing)
    x = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    out = model(x, routing=routing, macro_features=None, return_aux=True)
    assert "fusion_strength" in out


@test("LeverageFactorization at confidence=0 and confidence=1 extremes")
def _():
    from diagnosis_layer import DiagnosisLayer
    from leverage_factorization import LeverageFactorization

    rng = np.random.default_rng(3)
    series = 100 + np.cumsum(rng.normal(0, 2, 100))
    diag = DiagnosisLayer().diagnose(series)
    lf = LeverageFactorization()

    rec0 = lf.recommend(
        0.5,
        confidence=0.0,
        volatility_profile=diag.volatility,
        extreme_severity=0.0,
        nav_premium_regime_code=0.0,
    )
    assert abs(rec0.recommended_leverage_ratio - 1.0) < 1e-6, (
        f"confidence=0 should give exactly 1.0x, got {rec0.recommended_leverage_ratio}"
    )

    rec1 = lf.recommend(
        0.5,
        confidence=1.0,
        volatility_profile=diag.volatility,
        extreme_severity=1.0,
        nav_premium_regime_code=1.0,
    )
    assert rec1.max_leverage_ceiling >= 1.0
    assert rec1.recommended_leverage_ratio <= rec1.max_leverage_ceiling + 1e-6


@test("LeverageFactorization margin_call_distance_pct stays in [0,1] across random inputs")
def _():
    from diagnosis_layer import VolatilityProfile
    from leverage_factorization import LeverageFactorization

    lf = LeverageFactorization()
    rng = np.random.default_rng(4)
    for _ in range(200):
        vp = VolatilityProfile(
            historical_vol=float(rng.uniform(0, 2.0)),
            rolling_vol_mean=0.0,
            rolling_vol_std=0.0,
            parkinson_vol=None,
            garman_klass_vol=None,
            arch_lm_stat=None,
            arch_lm_pvalue=None,
        )
        rec = lf.recommend(
            position_scale=float(rng.uniform(-1, 1)),
            confidence=float(rng.uniform(0, 1)),
            volatility_profile=vp,
            extreme_severity=float(rng.uniform(0, 1)),
            nav_premium_regime_code=float(rng.uniform(-1, 1)),
        )
        assert 0.0 <= rec.margin_call_distance_pct <= 1.0, (
            f"out of range: {rec.margin_call_distance_pct}"
        )
        assert rec.max_leverage_ceiling >= 1.0
        assert rec.recommended_leverage_ratio >= 1.0


@test("Very short series (n=6, just above DiagnosisLayer's n<5 floor)")
def _():
    from diagnosis_layer import DiagnosisLayer

    series = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0])
    result = DiagnosisLayer().diagnose(series)
    assert result.basic.n_obs == 6


@test("DiagnosisLayer raises cleanly on n=4 (below its documented floor)")
def _():
    from diagnosis_layer import DiagnosisLayer

    series = np.array([100.0, 101.0, 99.0, 102.0])
    try:
        DiagnosisLayer().diagnose(series)
        raise AssertionError("expected ValueError for n<5 but none was raised")
    except ValueError:
        pass


@test("Constant (zero-variance) series does not crash the full pipeline")
def _():
    from combinatorial_gate import CombinatorialGate
    from diagnosis_layer import DiagnosisLayer

    series = np.full(60, 100.0)
    diag = DiagnosisLayer().diagnose(series)
    routing = CombinatorialGate().route(diag).routing
    assert routing.dominant_case is not None


@test("NAVPremiumCalculator.to_macro_series on dates entirely before the first NAV row")
def _():
    import pandas as pd

    from nav_premium import NAVPremiumCalculator

    nav_df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20, freq="7D"),
            "btc_holdings": np.linspace(190000, 200000, 20),
            "btc_price_usd": np.linspace(60000, 90000, 20),
            "shares_outstanding": np.linspace(2e8, 2.1e8, 20),
            "stock_price_usd": np.linspace(300, 400, 20),
        }
    )
    calc = NAVPremiumCalculator(rolling_window=10)
    computed = calc.compute(nav_df)
    early_dates = pd.date_range("2020-01-01", periods=5, freq="D")  # before any NAV row
    try:
        calc.to_macro_series(computed, early_dates)
        raise AssertionError("expected ValueError for dates before first NAV row")
    except ValueError:
        pass  # documented behavior: latest_snapshot raises when as_of predates all data


@test(
    "pipeline.py full orchestrator: build -> train (2 epochs) -> infer -> backtest, with NAV macro enabled"
)
def _():
    import pandas as pd

    from pipeline import PipelineConfig, build_pipeline, run_backtest, run_inference, run_training
    from training_extreme_engine import TrainConfig

    rng = np.random.default_rng(5)
    n = 220
    t = np.arange(n)
    series = 100 + 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 18) + rng.normal(0, 1.0, n)
    series[130:145] -= 10.0
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    nav_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="7D"),
            "btc_holdings": np.linspace(190000, 210000, 30),
            "btc_price_usd": np.linspace(60000, 95000, 30),
            "shares_outstanding": np.linspace(2e8, 2.2e8, 30),
            "stock_price_usd": np.linspace(300, 420, 30),
        }
    )

    config = PipelineConfig(
        train_config=TrainConfig(
            window=32, batch_size=8, epochs=2, log_every=1000, early_stop_patience=2
        ),
        backtest_kwargs={"window": 32, "step": 10},
        use_nav_macro=True,
    )
    bundle = build_pipeline(config)

    epochs_seen = []
    run_training(
        bundle,
        series[:170],
        series[170:],
        progress_callback=lambda e, tot, loss: epochs_seen.append(e),
    )
    assert len(epochs_seen) > 0, "progress_callback was never invoked during training"

    decision = run_inference(bundle, series[-60:], nav_df=nav_df, price_dates=dates[-60:])
    assert decision.action is not None

    bt_progress = []
    result = run_backtest(
        bundle,
        series,
        nav_df=nav_df,
        price_dates=dates,
        progress_callback=lambda t, n_: bt_progress.append(t),
    )
    assert len(bt_progress) > 0, "backtest progress_callback was never invoked"
    assert result.equity_curve.shape[0] > 1


print("\n" + "=" * 70)
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"RESULTS: {n_pass} passed, {n_fail} failed (of {len(results)})")
for name, status, err in results:
    if status == "FAIL":
        print(f"  FAILED: {name} -> {err}")
print("=" * 70)
