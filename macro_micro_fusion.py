"""
CARN-X  –  Multi-Modal / Macro-Micro Fusion  (Layer 5)
=======================================================
Fuses primary series (micro) with external macro features
under the control of the Combinatorial Gate.

Key ideas:
- Cross-attention between micro sequence and macro context
- Gated residual fusion controlled by Gate signals
- Macro can be static (per-sample) or time-aligned
- Optional missing-macro robustness
- Outputs enriched hidden states for the Core / Decision layers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from combinatorial_gate import AdvancedRoutingMap, CombinatorialCase

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class CrossAttention(nn.Module):
    """Micro queries attend to Macro keys/values."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, micro: Tensor, macro: Tensor) -> Tensor:
        """
        micro: [B, T, D]
        macro: [B, M, D]   (M = number of macro tokens)
        """
        B, T, D = micro.shape
        M = macro.shape[1]

        q = self.q_proj(micro).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(macro).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(macro).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class GatedFusion(nn.Module):
    """Learn how much macro information to inject into micro."""

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(dim, dim)

    def forward(self, micro: Tensor, macro_context: Tensor) -> Tensor:
        if macro_context.dim() == 2:
            macro_context = macro_context.unsqueeze(1).expand_as(micro)
        combined = torch.cat([micro, macro_context], dim=-1)
        g = self.gate(combined)
        return micro + g * self.proj(macro_context)


# ---------------------------------------------------------------------------
# Main Fusion Module
# ---------------------------------------------------------------------------


class MacroMicroFusion(nn.Module):
    """
    Layer 5: enriches micro sequence with macro information.
    Controlled by AdvancedRoutingMap.
    """

    def __init__(
        self,
        micro_dim: int = 256,
        macro_input_dim: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_macro_tokens: int = 16,
    ):
        super().__init__()
        self.micro_dim = micro_dim
        self.macro_input_dim = macro_input_dim

        self.macro_proj = nn.Sequential(
            nn.Linear(macro_input_dim, micro_dim),
            nn.SiLU(),
            nn.Linear(micro_dim, micro_dim),
        )

        self.norm_micro = RMSNorm(micro_dim)
        self.norm_macro = RMSNorm(micro_dim)

        self.cross_attn = CrossAttention(micro_dim, num_heads=num_heads, dropout=dropout)
        self.fusion = GatedFusion(micro_dim)

        self.macro_self_attn = nn.MultiheadAttention(
            micro_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.out_norm = RMSNorm(micro_dim)
        self.dropout = nn.Dropout(dropout)

        self.missing_macro_token = nn.Parameter(torch.randn(1, 1, micro_dim) * 0.02)

    def forward(
        self,
        micro_hidden: Tensor,
        macro_features: Tensor | None = None,
        routing: AdvancedRoutingMap | None = None,
        macro_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        micro_hidden : [B, T, D]
        macro_features : [B, M, F] or [B, F] or None
        """
        B, T, D = micro_hidden.shape
        device = micro_hidden.device

        fusion_strength = 0.5
        if routing is not None:
            fusion_strength = float(
                np.clip(
                    0.3
                    + 0.3 * routing.gates.extreme_event_handler
                    + 0.2 * routing.gates.garch_volatility
                    + 0.2 * (1.0 - getattr(routing.info_metrics, "decision_confidence", 0.5)),
                    0.1,
                    1.0,
                )
            )

        if macro_features is None:
            macro_tokens = self.missing_macro_token.expand(B, 1, D)
        else:
            if macro_features.dim() == 2:
                macro_features = macro_features.unsqueeze(1)
            macro_tokens = self.macro_proj(macro_features)

            if macro_tokens.shape[1] > 1:
                macro_tokens = self.norm_macro(macro_tokens)
                attn_out, _ = self.macro_self_attn(macro_tokens, macro_tokens, macro_tokens)
                macro_tokens = macro_tokens + self.dropout(attn_out)

        micro_n = self.norm_micro(micro_hidden)
        cross = self.cross_attn(micro_n, macro_tokens)

        fused = self.fusion(micro_hidden, cross)
        fused = micro_hidden + fusion_strength * (fused - micro_hidden)
        fused = self.out_norm(fused)

        return {
            "fused_hidden": fused,
            "fusion_strength": torch.tensor(fusion_strength, device=device),
            "macro_tokens": macro_tokens,
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


class MacroAwareCore(nn.Module):
    """
    Wraps HybridNeuralCore + MacroMicroFusion.
    """

    def __init__(self, core: nn.Module, fusion: MacroMicroFusion):
        super().__init__()
        self.core = core
        self.fusion = fusion

    def forward(
        self,
        x: Tensor,
        macro: Tensor | None = None,
        routing: AdvancedRoutingMap | None = None,
        return_aux: bool = False,
    ) -> dict[str, Tensor]:
        core_out = self.core(x, routing=routing, return_aux=True)
        hidden = core_out["hidden"]

        fusion_out = self.fusion(hidden, macro_features=macro, routing=routing)
        fused = fusion_out["fused_hidden"]

        out = {
            "forecast": core_out.get("forecast"),
            "hidden": fused,
            "fusion_strength": fusion_out["fusion_strength"],
            "severity": core_out.get("severity"),
            "leverage_action": core_out.get("leverage_action"),
            "regime_logits": core_out.get("regime_logits"),
            "dist_params": core_out.get("dist_params"),
        }
        if return_aux:
            out["macro_tokens"] = fusion_out["macro_tokens"]
            out["routing_case"] = core_out.get("routing_case")
            out["gates"] = core_out.get("gates")
        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Macro-Micro Fusion (Layer 5) – Self Test")
    print("=" * 70)

    from combinatorial_gate import CombinatorialGate
    from diagnosis_layer import DiagnosisLayer
    from hybrid_neural_core import HybridNeuralCore

    torch.manual_seed(42)
    B, T, D = 4, 64, 256
    F_macro = 6

    micro = torch.randn(B, T, D)
    macro = torch.randn(B, 3, F_macro)

    series = np.cumsum(np.random.randn(120)) + 100
    diag = DiagnosisLayer().diagnose(series)
    routing = CombinatorialGate().route(diag).routing

    fusion = MacroMicroFusion(micro_dim=D, macro_input_dim=F_macro, num_heads=4)
    out = fusion(micro, macro_features=macro, routing=routing)

    print(f"Input micro:  {tuple(micro.shape)}")
    print(f"Input macro:  {tuple(macro.shape)}")
    print(f"Fused hidden: {tuple(out['fused_hidden'].shape)}")
    print(f"Fusion strength (Gate-controlled): {out['fusion_strength'].item():.3f}")
    print(f"Dominant case: {routing.dominant_case}")

    core = HybridNeuralCore(input_dim=1, default_dim=D)
    core.configure_from_routing(routing)
    wrapper = MacroAwareCore(core, fusion)

    x = torch.randn(B, T, 1)
    full_out = wrapper(x, macro=macro, routing=routing, return_aux=True)
    print(f"\nWrapper forecast shape: {tuple(full_out['forecast'].shape)}")
    print(f"Wrapper fused hidden:   {tuple(full_out['hidden'].shape)}")

    print("\n" + "=" * 70)
    print("Macro-Micro Fusion ready.")
    print("=" * 70)
