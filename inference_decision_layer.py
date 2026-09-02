"""
CARN-X  –  Inference & Decision Layer  (Layer 6)
=================================================
Turns raw model outputs + Gate signals into actionable,
explainable, risk-aware decisions.

Crazy-level features:
- Multi-objective decision policy (forecast + extreme + leverage + regime)
- Confidence-calibrated action thresholds
- Counterfactual “what-if” probes (egg-drop / balance-scale inspired)
- Natural-language style explanation generator (structured)
- Position sizing under dynamic risk budget
- Regime-aware policy switching
- Soft combinatorial constraint enforcement
- Decision trace for auditability
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combinatorial_gate import (
    CASE_LABELS,
    AdvancedRoutingMap,
    AttentionGates,
    CombinatorialCase,
)
from leverage_factorization import LeverageFactorization, LeverageRecommendation
from training_extreme_engine import ExtremeDecision, LeverageSuggestion


class ActionType(Enum):
    HOLD = auto()
    INCREASE = auto()
    DECREASE = auto()
    HEDGE = auto()
    CASH = auto()
    PROBE = auto()
    REBALANCE = auto()


@dataclass
class DecisionTrace:
    timestamp: str | None = None
    dominant_case: str = ""
    forecast_value: float = 0.0
    forecast_direction: str = ""
    regime: str = ""
    extreme: bool = False
    severity: float = 0.0
    risk_budget: float = 1.0
    chosen_action: ActionType = ActionType.HOLD
    position_scale: float = 0.0
    confidence: float = 0.0
    explanation: list[str] = field(default_factory=list)
    counterfactuals: list[str] = field(default_factory=list)
    gates_snapshot: dict[str, float] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionOutput:
    action: ActionType
    position_scale: float
    confidence: float
    risk_budget: float
    explanation: list[str]
    trace: DecisionTrace
    leverage_suggestion: LeverageSuggestion | None = None
    extreme_decision: ExtremeDecision | None = None
    leverage_recommendation: LeverageRecommendation | None = None


class DecisionPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_actions: int = 7):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )
        self.n_actions = n_actions

    def forward(self, features: Tensor) -> Tensor:
        return self.scorer(features)


class InferenceDecisionLayer:
    def __init__(
        self,
        long_threshold: float = 0.15,
        short_threshold: float = -0.15,
        extreme_severity_cut: float = 0.55,
        min_confidence_to_act: float = 0.35,
        max_position: float = 1.0,
        leverage_factorizer: LeverageFactorization | None = None,
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.extreme_severity_cut = extreme_severity_cut
        self.min_confidence_to_act = min_confidence_to_act
        self.max_position = max_position
        self.leverage_factorizer = leverage_factorizer
        self.policy = DecisionPolicy(hidden_dim=128)

    def decide(
        self,
        model_out: dict[str, Any],
        routing: AdvancedRoutingMap,
        extreme: ExtremeDecision | None = None,
        leverage: LeverageSuggestion | None = None,
        current_position: float = 0.0,
        volatility_profile: Any | None = None,
        nav_premium_regime_code: float = 0.0,
    ) -> DecisionOutput:
        forecast = self._extract_forecast(model_out)
        regime = self._extract_regime(model_out)
        severity = (
            float(model_out.get("severity", torch.tensor(0.0)).mean())
            if "severity" in model_out
            else 0.0
        )
        confidence = float(routing.info_metrics.decision_confidence)

        is_extreme = False
        risk_budget = 1.0
        if extreme is not None:
            is_extreme = extreme.is_extreme
            severity = max(severity, extreme.severity)
            risk_budget = extreme.risk_budget

        scores = self._score_actions(
            forecast, regime, severity, is_extreme, confidence, routing, current_position
        )

        action, position_scale = self._select_action(
            scores, confidence, risk_budget, is_extreme, severity
        )

        leverage_recommendation = None
        if self.leverage_factorizer is not None and volatility_profile is not None:
            leverage_recommendation = self.leverage_factorizer.recommend(
                position_scale=position_scale,
                confidence=confidence,
                volatility_profile=volatility_profile,
                extreme_severity=severity,
                nav_premium_regime_code=nav_premium_regime_code,
            )
            position_scale = leverage_recommendation.risk_adjusted_position_scale

        explanation = self._build_explanation(
            action,
            forecast,
            regime,
            severity,
            is_extreme,
            confidence,
            routing,
            risk_budget,
            position_scale,
        )
        counterfactuals = self._build_counterfactuals(forecast, severity, routing, action)

        trace = DecisionTrace(
            dominant_case=CASE_LABELS.get(routing.dominant_case, str(routing.dominant_case)),
            forecast_value=forecast,
            forecast_direction="up" if forecast > 0 else ("down" if forecast < 0 else "flat"),
            regime=regime,
            extreme=is_extreme,
            severity=severity,
            risk_budget=risk_budget,
            chosen_action=action,
            position_scale=position_scale,
            confidence=confidence,
            explanation=explanation,
            counterfactuals=counterfactuals,
            gates_snapshot={
                "extreme": routing.gates.extreme_event_handler,
                "leverage": routing.gates.leverage_optimizer,
                "fibonacci": routing.gates.fibonacci_paths,
                "egg_drop": routing.gates.egg_drop_sequential,
                "fourier": routing.gates.fourier_spectral,
            },
            raw_scores=scores,
        )

        return DecisionOutput(
            action=action,
            position_scale=position_scale,
            confidence=confidence,
            risk_budget=risk_budget,
            explanation=explanation,
            trace=trace,
            leverage_suggestion=leverage,
            extreme_decision=extreme,
            leverage_recommendation=leverage_recommendation,
        )

    def _extract_forecast(self, model_out: dict[str, Any]) -> float:
        f = model_out.get("forecast")
        if f is None:
            return 0.0
        if torch.is_tensor(f):
            return float(f[:, -1].mean().cpu())
        return float(np.mean(f[..., -1]))

    def _extract_regime(self, model_out: dict[str, Any]) -> str:
        logits = model_out.get("regime_logits")
        if logits is None:
            return "unknown"
        if torch.is_tensor(logits):
            idx = int(logits.mean(dim=0).argmax().cpu())
        else:
            idx = int(np.argmax(np.mean(logits, axis=0)))
        names = ["bull", "bear", "sideways", "volatile"]
        return names[idx] if idx < len(names) else "unknown"

    def _score_actions(
        self, forecast, regime, severity, is_extreme, confidence, routing, current_position
    ):
        scores = {a.name: 0.0 for a in ActionType}

        if forecast > self.long_threshold:
            scores["INCREASE"] += 1.2 * min(abs(forecast), 1.0)
        elif forecast < self.short_threshold:
            scores["DECREASE"] += 1.2 * min(abs(forecast), 1.0)
        else:
            scores["HOLD"] += 0.8

        if regime == "bull":
            scores["INCREASE"] += 0.4
        elif regime == "bear":
            scores["DECREASE"] += 0.4
            scores["HEDGE"] += 0.3
        elif regime == "volatile":
            scores["HEDGE"] += 0.5
            scores["PROBE"] += 0.3

        if is_extreme or severity > self.extreme_severity_cut:
            scores["CASH"] += 1.5 * severity
            scores["HEDGE"] += 1.0 * severity
            scores["DECREASE"] += 0.8 * severity
            scores["INCREASE"] -= 1.0 * severity
            scores["PROBE"] += 0.6 * routing.gates.egg_drop_sequential

        if confidence < self.min_confidence_to_act:
            for aggressive in ["INCREASE", "DECREASE"]:
                scores[aggressive] *= 0.5
            scores["HOLD"] += 0.6
            scores["PROBE"] += 0.4

        case = routing.dominant_case
        if (
            case in (CombinatorialCase.ORDER_WITH_REP, CombinatorialCase.ORDER_NO_REP)
            and confidence < 0.5
        ):
            scores["PROBE"] += 0.5
        if case in (CombinatorialCase.NO_ORDER_WITH_REP, CombinatorialCase.NO_ORDER_NO_REP):
            scores["REBALANCE"] += 0.3

        if routing.gates.leverage_optimizer > 0.6 and not is_extreme:
            scores["INCREASE"] += 0.3
            scores["REBALANCE"] += 0.4

        vals = np.array(list(scores.values()))
        vals = vals - vals.max()
        exp = np.exp(vals)
        probs = exp / (exp.sum() + 1e-8)
        return {k: float(p) for k, p in zip(scores.keys(), probs)}

    def _select_action(self, scores, confidence, risk_budget, is_extreme, severity):
        best = max(scores.items(), key=lambda kv: kv[1])
        action = ActionType[best[0]]

        if action == ActionType.HOLD:
            scale = 0.0
        elif action == ActionType.INCREASE:
            scale = min(self.max_position, 0.4 + 0.6 * confidence) * risk_budget
        elif action == ActionType.DECREASE:
            scale = -min(self.max_position, 0.4 + 0.6 * confidence) * risk_budget
        elif action == ActionType.HEDGE:
            scale = -0.3 * risk_budget
        elif action == ActionType.CASH:
            scale = 0.0
        elif action == ActionType.PROBE:
            scale = 0.1 * risk_budget * (1.0 if not is_extreme else 0.5)
        elif action == ActionType.REBALANCE:
            scale = 0.2 * risk_budget
        else:
            scale = 0.0

        if is_extreme and severity > 0.7:
            scale *= 0.4

        return action, float(np.clip(scale, -self.max_position, self.max_position))

    def _build_explanation(
        self,
        action,
        forecast,
        regime,
        severity,
        is_extreme,
        confidence,
        routing,
        risk_budget,
        position_scale,
    ):
        lines = []
        lines.append(f"Action chosen: {action.name} (scale={position_scale:+.3f})")
        lines.append(f"Forecast signal: {forecast:+.4f} | Regime: {regime}")
        lines.append(f"Confidence: {confidence:.1%} | Risk budget: {risk_budget:.2f}")
        lines.append(
            f"Combinatorial case: {CASE_LABELS.get(routing.dominant_case, str(routing.dominant_case))}"
        )
        if is_extreme:
            lines.append(f"⚠ EXTREME regime active (severity={severity:.2f}) → defensive posture")
        if routing.gates.egg_drop_sequential > 0.5 and action == ActionType.PROBE:
            lines.append("Egg-drop gate high → sequential probing mode engaged")
        if routing.gates.leverage_optimizer > 0.6:
            lines.append("Leverage optimizer gate elevated → recovery bias present")
        if confidence < self.min_confidence_to_act:
            lines.append("Low decision confidence → reduced aggressiveness")
        return lines

    def _build_counterfactuals(self, forecast, severity, routing, chosen):
        cfs = []
        if chosen != ActionType.INCREASE and forecast > self.long_threshold:
            cfs.append(
                "Counterfactual: if confidence were higher, INCREASE would have been preferred"
            )
        if chosen != ActionType.CASH and severity > 0.6:
            cfs.append("Counterfactual: under even higher severity, CASH becomes dominant")
        if routing.gates.fibonacci_paths > 0.6:
            cfs.append(
                "Counterfactual: strong Fibonacci gate → path-counting recovery sequences possible"
            )
        if routing.gates.balance_scale_search > 0.5:
            cfs.append(
                "Counterfactual: balance-scale gate → ternary search style position adjustment available"
            )
        return cfs


def full_inference_pipeline(
    series: np.ndarray,
    model: nn.Module,
    routing: AdvancedRoutingMap,
    extreme_handler=None,
    leverage_optimizer=None,
    decision_layer: InferenceDecisionLayer | None = None,
    current_position: float = 0.0,
    device: str = "cpu",
    volatility_profile: Any | None = None,
    nav_premium_regime_code: float = 0.0,
    macro_features: Any | None = None,
) -> DecisionOutput:
    decision_layer = decision_layer or InferenceDecisionLayer()
    model.eval()
    model.to(device)

    x = torch.tensor(series, dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(-1)

    macro_t = None
    if macro_features is not None:
        macro_t = torch.as_tensor(macro_features, dtype=torch.float32, device=device)
        if macro_t.dim() == 1:
            macro_t = macro_t.unsqueeze(0)

    with torch.no_grad():
        out = model(x, routing=routing, macro_features=macro_t, return_aux=True)

    extreme = None
    leverage = None
    if extreme_handler is not None:
        sev = float(out["severity"].mean().cpu()) if "severity" in out else None
        extreme = extreme_handler.evaluate(series, routing, sev)
    if leverage_optimizer is not None and "leverage_action" in out:
        leverage = leverage_optimizer.suggest(out["leverage_action"], extreme, routing)

    return decision_layer.decide(
        model_out=out,
        routing=routing,
        extreme=extreme,
        leverage=leverage,
        current_position=current_position,
        volatility_profile=volatility_profile,
        nav_premium_regime_code=nav_premium_regime_code,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Inference & Decision Layer (Layer 6) – Self Test")
    print("=" * 70)

    from combinatorial_gate import CombinatorialGate
    from diagnosis_layer import DiagnosisLayer
    from hybrid_neural_core import HybridNeuralCore
    from training_extreme_engine import ExtremeEventHandler, LeverageOptimizer

    rng = np.random.default_rng(42)
    t = np.arange(200)
    series = 0.012 * t + 2.2 * np.sin(2 * np.pi * t / 15) + rng.normal(0, 0.7, 200)
    series[120:145] -= 9.5

    diag = DiagnosisLayer().diagnose(series)
    routing = CombinatorialGate(temperature=0.5).route(diag).routing

    model = HybridNeuralCore(input_dim=1)
    model.configure_from_routing(routing)

    decision = full_inference_pipeline(
        series=series,
        model=model,
        routing=routing,
        extreme_handler=ExtremeEventHandler(),
        leverage_optimizer=LeverageOptimizer(),
        decision_layer=InferenceDecisionLayer(),
        current_position=0.3,
    )

    print(f"\nAction          : {decision.action.name}")
    print(f"Position scale  : {decision.position_scale:+.3f}")
    print(f"Confidence      : {decision.confidence:.1%}")
    print(f"Risk budget     : {decision.risk_budget:.2f}")
    print(f"Extreme         : {decision.trace.extreme} (severity={decision.trace.severity:.2f})")
    print("\nExplanation:")
    for line in decision.explanation:
        print(f"  • {line}")
    if decision.trace.counterfactuals:
        print("\nCounterfactuals:")
        for cf in decision.trace.counterfactuals:
            print(f"  ↳ {cf}")

    print("\n" + "=" * 70)
    print("Inference & Decision Layer ready.")
    print("=" * 70)
