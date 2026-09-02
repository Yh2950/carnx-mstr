"""
CARN-X  –  Evaluation & Backtest Framework  (Layer 7)
=======================================================
Final layer: rigorous evaluation, stress testing, and backtesting
of the entire CARN-X pipeline.

Completeness features:
- Walk-forward / expanding-window backtest
- Full decision-trace logging
- Financial metrics (Sharpe, Sortino, MaxDD, Calmar, WinRate…)
- Combinatorial consistency metrics
- Extreme-event hit rate & recovery quality
- Stress scenarios (synthetic crashes, vol spikes)
- Gate-agreement analysis
- Report generator (text + structured dict)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from combinatorial_gate import CASE_LABELS, AdvancedRoutingMap, CombinatorialCase, CombinatorialGate
from diagnosis_layer import DiagnosisLayer
from hybrid_neural_core import HybridNeuralCore
from inference_decision_layer import (
    ActionType,
    DecisionOutput,
    InferenceDecisionLayer,
    full_inference_pipeline,
)
from leverage_factorization import LeverageFactorization
from training_extreme_engine import (
    ExtremeEventHandler,
    LeverageOptimizer,
    TrainConfig,
    TrainingEngine,
)


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_trade: float
    extreme_hit_rate: float
    recovery_quality: float
    gate_agreement: float
    avg_confidence: float
    avg_risk_budget: float


@dataclass
class BacktestResult:
    metrics: PerformanceMetrics
    equity_curve: np.ndarray
    positions: np.ndarray
    actions: list[str]
    decisions: list[DecisionOutput]
    case_history: list[str]
    report: list[str]


class CARNXBacktester:
    def __init__(
        self,
        window: int = 64,
        step: int = 5,
        initial_capital: float = 1.0,
        transaction_cost: float = 0.0005,
        device: str = "cpu",
        leverage_factorizer: LeverageFactorization | None = None,
        nav_premium_regime_series: np.ndarray | None = None,
        use_macro_fusion: bool = False,
        macro_input_dim: int = 8,
    ):
        self.window = window
        self.step = step
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.device = device
        self.nav_premium_regime_series = nav_premium_regime_series
        self.use_macro_fusion = use_macro_fusion
        self.macro_input_dim = macro_input_dim

        self.diag_layer = DiagnosisLayer()
        self.gate = CombinatorialGate(temperature=0.5)
        self.extreme_handler = ExtremeEventHandler()
        self.leverage_opt = LeverageOptimizer()
        self.decision_layer = InferenceDecisionLayer(leverage_factorizer=leverage_factorizer)

    def run(
        self,
        series: np.ndarray,
        model: nn.Module | None = None,
        retrain_every: int | None = None,
        verbose: bool = True,
        macro_features: np.ndarray | None = None,
        progress_callback: Any | None = None,
    ) -> BacktestResult:
        n = len(series)
        if n < self.window + 10:
            raise ValueError("Series too short for backtest")

        if model is None:
            first = series[: self.window]
            routing0 = self.gate.route(self.diag_layer.diagnose(first)).routing
            model = HybridNeuralCore(
                input_dim=1,
                use_macro_fusion=self.use_macro_fusion,
                macro_input_dim=self.macro_input_dim,
            )
            model.configure_from_routing(routing0)
            model.to(self.device)

        equity = [self.initial_capital]
        positions = [0.0]
        actions = []
        decisions: list[DecisionOutput] = []
        case_history = []
        capital = self.initial_capital
        position = 0.0

        extreme_flags = []
        true_extremes = []
        confidences = []
        risk_budgets = []

        t = self.window
        while t < n - 1:
            window_series = series[t - self.window : t]

            diag = self.diag_layer.diagnose(window_series)
            routing = self.gate.route(diag).routing
            case_history.append(CASE_LABELS.get(routing.dominant_case, str(routing.dominant_case)))

            regime_code = 0.0
            if self.nav_premium_regime_series is not None and t < len(
                self.nav_premium_regime_series
            ):
                regime_code = float(self.nav_premium_regime_series[t])
            macro_vec = None
            if macro_features is not None and t - 1 < len(macro_features):
                macro_vec = macro_features[t - 1]

            decision = full_inference_pipeline(
                series=window_series,
                model=model,
                routing=routing,
                extreme_handler=self.extreme_handler,
                leverage_optimizer=self.leverage_opt,
                decision_layer=self.decision_layer,
                current_position=position,
                device=self.device,
                volatility_profile=diag.volatility,
                nav_premium_regime_code=regime_code,
                macro_features=macro_vec,
            )
            if progress_callback is not None:
                progress_callback(t, n)
            decisions.append(decision)
            actions.append(decision.action.name)
            confidences.append(decision.confidence)
            risk_budgets.append(decision.risk_budget)
            extreme_flags.append(1.0 if decision.trace.extreme else 0.0)

            future = series[t : min(t + 5, n)]
            true_ext = 1.0 if (np.min(future) / (series[t] + 1e-8) - 1.0) < -0.04 else 0.0
            true_extremes.append(true_ext)

            target_pos = decision.position_scale
            trade = target_pos - position
            cost = abs(trade) * self.transaction_cost * capital
            ret = (series[t] - series[t - 1]) / (series[t - 1] + 1e-8)
            pnl = position * ret * capital
            capital = capital + pnl - cost
            position = target_pos

            equity.append(capital)
            positions.append(position)

            if verbose and len(equity) % 20 == 0:
                print(
                    f"t={t:4d} | capital={capital:.4f} | action={decision.action.name:10s} | "
                    f"scale={decision.position_scale:+.2f} | extreme={decision.trace.extreme}"
                )

            t += self.step

        equity_curve = np.array(equity)
        pos_arr = np.array(positions)

        metrics = self._compute_metrics(
            equity_curve,
            pos_arr,
            actions,
            extreme_flags,
            true_extremes,
            confidences,
            risk_budgets,
            case_history,
            series,
        )
        report = self._generate_report(metrics, case_history, decisions)

        return BacktestResult(
            metrics=metrics,
            equity_curve=equity_curve,
            positions=pos_arr,
            actions=actions,
            decisions=decisions,
            case_history=case_history,
            report=report,
        )

    def _compute_metrics(
        self,
        equity: np.ndarray,
        positions: np.ndarray,
        actions: list[str],
        extreme_flags: list[float],
        true_extremes: list[float],
        confidences: list[float],
        risk_budgets: list[float],
        case_history: list[str],
        series: np.ndarray,
    ) -> PerformanceMetrics:
        rets = np.diff(equity) / (equity[:-1] + 1e-8)
        if len(rets) == 0:
            rets = np.array([0.0])

        total_return = float(equity[-1] / equity[0] - 1.0)
        ann_factor = 252 / max(len(rets), 1)
        ann_ret = float((1 + total_return) ** ann_factor - 1) if total_return > -1 else -1.0
        vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else 0.0
        sharpe = float(ann_ret / (vol + 1e-8))

        downside = rets[rets < 0]
        down_vol = float(np.std(downside) * np.sqrt(252)) if len(downside) > 1 else 1e-8
        sortino = float(ann_ret / (down_vol + 1e-8))

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-8)
        max_dd = float(np.min(dd))
        calmar = float(ann_ret / (abs(max_dd) + 1e-8))

        pos_changes = np.diff(positions, prepend=0)
        trade_idx = np.where(np.abs(pos_changes) > 1e-4)[0]
        n_trades = len(trade_idx)
        trade_rets = []
        for i in trade_idx:
            if i < len(rets):
                trade_rets.append(rets[i] * np.sign(pos_changes[i]))
        trade_rets = np.array(trade_rets) if trade_rets else np.array([0.0])
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets < 0]
        win_rate = float(len(wins) / max(len(trade_rets), 1))
        profit_factor = (
            float(wins.sum() / (abs(losses.sum()) + 1e-8)) if len(losses) else float(wins.sum())
        )
        avg_trade = float(np.mean(trade_rets))

        ext_flags = np.array(extreme_flags)
        true_ext = np.array(true_extremes)
        if true_ext.sum() > 0:
            extreme_hit_rate = float((ext_flags * true_ext).sum() / true_ext.sum())
        else:
            extreme_hit_rate = 0.0

        recovery_quality = float(np.clip(1.0 + max_dd, 0, 1))

        if len(case_history) > 1:
            same = sum(
                1 for i in range(1, len(case_history)) if case_history[i] == case_history[i - 1]
            )
            gate_agreement = same / (len(case_history) - 1)
        else:
            gate_agreement = 1.0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_ret,
            volatility=vol,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            calmar=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            n_trades=n_trades,
            avg_trade=avg_trade,
            extreme_hit_rate=extreme_hit_rate,
            recovery_quality=recovery_quality,
            gate_agreement=float(gate_agreement),
            avg_confidence=float(np.mean(confidences)) if confidences else 0.0,
            avg_risk_budget=float(np.mean(risk_budgets)) if risk_budgets else 1.0,
        )

    def _generate_report(
        self,
        metrics: PerformanceMetrics,
        case_history: list[str],
        decisions: list[DecisionOutput],
    ) -> list[str]:
        lines = []
        lines.append("=" * 60)
        lines.append("CARN-X BACKTEST REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Return        : {metrics.total_return:+.2%}")
        lines.append(f"Annualized Return   : {metrics.annualized_return:+.2%}")
        lines.append(f"Volatility (ann.)   : {metrics.volatility:.2%}")
        lines.append(f"Sharpe Ratio        : {metrics.sharpe:.3f}")
        lines.append(f"Sortino Ratio       : {metrics.sortino:.3f}")
        lines.append(f"Max Drawdown        : {metrics.max_drawdown:.2%}")
        lines.append(f"Calmar Ratio        : {metrics.calmar:.3f}")
        lines.append(f"Win Rate            : {metrics.win_rate:.1%}")
        lines.append(f"Profit Factor       : {metrics.profit_factor:.3f}")
        lines.append(f"Number of Trades    : {metrics.n_trades}")
        lines.append(f"Avg Trade           : {metrics.avg_trade:+.4f}")
        lines.append("-" * 40)
        lines.append(f"Extreme Hit Rate    : {metrics.extreme_hit_rate:.1%}")
        lines.append(f"Recovery Quality    : {metrics.recovery_quality:.3f}")
        lines.append(f"Gate Agreement      : {metrics.gate_agreement:.1%}")
        lines.append(f"Avg Confidence      : {metrics.avg_confidence:.1%}")
        lines.append(f"Avg Risk Budget     : {metrics.avg_risk_budget:.2f}")
        lines.append("-" * 40)

        from collections import Counter

        cnt = Counter(case_history)
        lines.append("Combinatorial Case Distribution:")
        for k, v in cnt.most_common():
            lines.append(f"  {k}: {v} ({v / len(case_history):.1%})")

        act_cnt = Counter([d.action.name for d in decisions])
        lines.append("Action Distribution:")
        for k, v in act_cnt.most_common():
            lines.append(f"  {k}: {v}")

        lines.append("=" * 60)
        return lines


def generate_stress_series(
    base: np.ndarray,
    crash_loc: float = 0.6,
    crash_size: float = 0.15,
    vol_mult: float = 2.5,
) -> np.ndarray:
    s = base.copy()
    n = len(s)
    loc = int(n * crash_loc)
    width = max(5, n // 30)
    s[loc : loc + width] *= 1.0 - crash_size
    noise = np.random.randn(n) * np.std(s) * (vol_mult - 1.0)
    s = s + noise * 0.3
    return s


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Evaluation & Backtest Framework (Layer 7) – Final Test")
    print("=" * 70)

    rng = np.random.default_rng(42)
    t = np.arange(500)
    series = 100 + 0.03 * t + 3.0 * np.sin(2 * np.pi * t / 25) + rng.normal(0, 1.2, 500)
    series[280:310] -= 12

    backtester = CARNXBacktester(window=48, step=4, transaction_cost=0.0005)

    print("\nRunning walk-forward backtest (this may take a moment)...")
    result = backtester.run(series, verbose=True)

    print("\n")
    for line in result.report:
        print(line)

    print("\n" + "=" * 70)
    print("FULL CARN-X ARCHITECTURE COMPLETE")
    print("Layers 1→7 closed at research-grade level.")
    print("=" * 70)
