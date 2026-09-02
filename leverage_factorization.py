"""
CARN-X – Leverage Factorization Module (Domain Layer, standalone)
====================================================================
Real financial leverage/margin math for the *user's own trading account*
(NOT MSTR's corporate balance sheet / convertible-note structure).
Advisory only — produces recommendations, never executes trades.

Distinct from training_extreme_engine.LeverageOptimizer, which stays
untouched — that module is the generic combinatorial "leverage" metaphor
consumed by HybridNeuralCore's leverage_action head.

Ceiling shrinkage: each risk factor (volatility, extreme severity, NAV
premium extremity) floors at a positive fraction so leverage is only ever
capped down toward 1.0x, never forced below it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class LeverageRecommendation:
    recommended_leverage_ratio: float  # 1.0 = no leverage, 2.0 = 2x gross exposure/equity
    max_leverage_ceiling: float  # regime/vol/premium-gated ceiling, >= 1.0
    margin_call_distance_pct: float  # adverse price-move fraction that triggers a margin call
    financing_cost_annual_pct_of_equity: float
    financing_cost_estimate_usd: float | None
    risk_adjusted_position_scale: float  # replaces the naive 1.0x-capped position_scale
    notes: list[str] = field(default_factory=list)


class LeverageFactorization:
    """
    Computes a regime-aware leverage ceiling and applies it to a base
    position_scale coming from InferenceDecisionLayer, entirely advisory.
    """

    def __init__(
        self,
        base_max_leverage: float = 3.0,
        maintenance_margin_pct: float = 0.25,
        annual_financing_rate: float = 0.07,
        vol_ceiling_kappa: float = 1.0,
        severity_ceiling_kappa: float = 1.0,
        premium_extreme_ceiling_kappa: float = 1.0,
    ):
        self.base_max_leverage = base_max_leverage
        self.maintenance_margin_pct = maintenance_margin_pct
        self.annual_financing_rate = annual_financing_rate
        self.vol_ceiling_kappa = vol_ceiling_kappa
        self.severity_ceiling_kappa = severity_ceiling_kappa
        self.premium_extreme_ceiling_kappa = premium_extreme_ceiling_kappa

    def compute_ceiling(
        self,
        volatility_profile,  # diagnosis_layer.VolatilityProfile
        extreme_severity: float,
        nav_premium_regime_code: float,
    ) -> float:
        normalized_vol = _clamp(volatility_profile.historical_vol / 0.75, 0.0, 1.0)
        severity = _clamp(extreme_severity, 0.0, 1.0)
        premium_extremity = _clamp(abs(nav_premium_regime_code), 0.0, 1.0)

        vol_factor = _clamp(1.0 - self.vol_ceiling_kappa * normalized_vol, 0.2, 1.0)
        severity_factor = _clamp(1.0 - self.severity_ceiling_kappa * severity, 0.1, 1.0)
        premium_factor = _clamp(
            1.0 - self.premium_extreme_ceiling_kappa * premium_extremity, 0.3, 1.0
        )

        ceiling = self.base_max_leverage * vol_factor * severity_factor * premium_factor
        return max(ceiling, 1.0)

    def recommend(
        self,
        position_scale: float,
        confidence: float,
        volatility_profile,
        extreme_severity: float,
        nav_premium_regime_code: float,
        position_notional_usd: float | None = None,
    ) -> LeverageRecommendation:
        ceiling = self.compute_ceiling(
            volatility_profile, extreme_severity, nav_premium_regime_code
        )
        confidence = _clamp(confidence, 0.0, 1.0)

        leverage = 1.0 + (ceiling - 1.0) * confidence
        risk_adjusted_position_scale = _clamp(position_scale * leverage, -ceiling, ceiling)

        if leverage > 1.0:
            margin_call_distance_pct = max(
                0.0,
                (1.0 / leverage - self.maintenance_margin_pct)
                / (1.0 - self.maintenance_margin_pct),
            )
        else:
            margin_call_distance_pct = 1.0  # unlevered — cannot be margin-called

        borrowed_fraction = leverage - 1.0
        financing_cost_pct = borrowed_fraction * self.annual_financing_rate
        financing_cost_usd = (
            financing_cost_pct * position_notional_usd
            if position_notional_usd is not None
            else None
        )

        notes: list[str] = []
        if leverage > 1.05:
            notes.append(
                f"Leverage {leverage:.2f}x recommended (ceiling {ceiling:.2f}x, confidence {confidence:.1%})."
            )
        else:
            notes.append(
                "No meaningful leverage recommended — near 1.0x (low confidence or elevated risk)."
            )
        if extreme_severity > 0.5:
            notes.append(
                f"Extreme-event severity {extreme_severity:.2f} is compressing the leverage ceiling."
            )
        if abs(nav_premium_regime_code) > 0.5:
            notes.append(
                "NAV premium at an extreme regime — leverage ceiling reduced for mNAV-compression risk."
            )
        if margin_call_distance_pct < 0.15 and leverage > 1.0:
            notes.append(
                f"⚠ Margin-call distance only {margin_call_distance_pct:.1%} — thin buffer."
            )

        return LeverageRecommendation(
            recommended_leverage_ratio=leverage,
            max_leverage_ceiling=ceiling,
            margin_call_distance_pct=margin_call_distance_pct,
            financing_cost_annual_pct_of_equity=financing_cost_pct,
            financing_cost_estimate_usd=financing_cost_usd,
            risk_adjusted_position_scale=risk_adjusted_position_scale,
            notes=notes,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Leverage Factorization Module – Self Test")
    print("=" * 70)

    from diagnosis_layer import DiagnosisLayer

    rng = np.random.default_rng(42)
    t = np.arange(200)
    series = 100 + 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 1.0, 200)
    series[120:135] -= 15.0  # crash -> high volatility + severity

    diag = DiagnosisLayer().diagnose(series)

    factorizer = LeverageFactorization()
    rec = factorizer.recommend(
        position_scale=0.6,
        confidence=0.7,
        volatility_profile=diag.volatility,
        extreme_severity=0.4,
        nav_premium_regime_code=0.6,  # elevated premium regime
        position_notional_usd=50_000.0,
    )

    print(f"\nrecommended_leverage_ratio        = {rec.recommended_leverage_ratio:.3f}x")
    print(f"max_leverage_ceiling               = {rec.max_leverage_ceiling:.3f}x")
    print(f"margin_call_distance_pct           = {rec.margin_call_distance_pct:.2%}")
    print(f"financing_cost_annual_pct_of_equity = {rec.financing_cost_annual_pct_of_equity:.2%}")
    print(f"financing_cost_estimate_usd        = ${rec.financing_cost_estimate_usd:,.2f}")
    print(f"risk_adjusted_position_scale        = {rec.risk_adjusted_position_scale:+.3f}")
    print("\nNotes:")
    for n in rec.notes:
        print(f"  • {n}")

    print("\n" + "=" * 70)
    print("Leverage Factorization Module ready.")
    print("=" * 70)
