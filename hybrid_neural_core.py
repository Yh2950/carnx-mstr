"""
CARN-X  –  Hybrid Neural Core  (Layer 3)
==========================================
Research-grade hybrid backbone that consumes AdvancedRoutingMap
from the Combinatorial Gate and assembles a dynamic network.

Design principles (public SOTA 2025-2026):
- Soft / hard routing driven by the Gate
- Transformer + optional State-Space flavour
- Gated expert modules (Fibonacci, Extreme, Fourier, …)
- Mixture-of-Pathways
- Combinatorial-aware loss terms
- Extreme-event specialized pathway
- Ready for financial time-series + multi-variate input
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import from previous layers
from combinatorial_gate import (
    AdvancedRoutingMap,
    AttentionGates,
    CombinatorialCase,
    GateResult,
    LossSchedule,
)
from macro_micro_fusion import MacroMicroFusion

# ---------------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: Tensor, seq_len: int | None = None) -> tuple[Tensor, Tensor]:
        if seq_len is None:
            seq_len = x.shape[2] if x.dim() == 4 else x.shape[1]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Attention & Transformer block
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, rope: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = rope
        if rope:
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            cos, sin = self.rotary(q, seq_len=T)
            q, k = apply_rotary(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1, mlp_ratio: float = 8 / 3):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Specialized gated modules (inspired by the strategic puzzles + math)
# ---------------------------------------------------------------------------


class FibonacciPathModule(nn.Module):
    """Recurrent path-counting inductive bias (stairs / Fibonacci)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.recurrent = nn.GRU(dim, dim, batch_first=True)
        self.proj_out = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, 1)

    def forward(self, x: Tensor, strength: float = 1.0) -> Tensor:
        h = self.proj_in(x)
        out, _ = self.recurrent(h)
        out = self.proj_out(out)
        g = torch.sigmoid(self.gate(x.mean(dim=1, keepdim=True))) * strength
        return x + g * out


class FourierFeatureModule(nn.Module):
    def __init__(self, dim: int, num_freqs: int = 32):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_proj = nn.Linear(dim, num_freqs * 2)
        self.out = nn.Linear(num_freqs * 2, dim)

    def forward(self, x: Tensor, strength: float = 1.0) -> Tensor:
        f = self.freq_proj(x)
        sin, cos = f.chunk(2, dim=-1)
        feat = torch.cat([torch.sin(sin), torch.cos(cos)], dim=-1)
        return x + strength * self.out(feat)


class ExtremeEventModule(nn.Module):
    """Specialized pathway for crashes / heavy tails / drawdowns."""

    def __init__(self, dim: int):
        super().__init__()
        self.down = nn.Linear(dim, dim * 2)
        self.up = nn.Linear(dim * 2, dim)
        self.severity = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 1),
        )
        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor, strength: float = 1.0) -> tuple[Tensor, Tensor]:
        h = F.silu(self.down(self.norm(x)))
        h = self.up(h)
        severity = torch.sigmoid(self.severity(x.mean(dim=1)))
        out = x + strength * severity.unsqueeze(1) * h
        return out, severity.squeeze(-1)


class LeverageOptimizerModule(nn.Module):
    """Suggests recovery / rebalancing actions under stress."""

    def __init__(self, dim: int, action_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, action_dim),
        )

    def forward(self, x: Tensor, strength: float = 1.0) -> Tensor:
        pooled = x.mean(dim=1)
        return strength * self.net(pooled)


class GatedExpert(nn.Module):
    """Generic expert that can be soft-gated."""

    def __init__(self, dim: int, expert_type: str = "mlp"):
        super().__init__()
        self.expert_type = expert_type
        if expert_type == "mlp":
            self.net = SwiGLU(dim)
        else:
            self.net = nn.Sequential(
                RMSNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
            )

    def forward(self, x: Tensor, gate: float) -> Tensor:
        return gate * self.net(x)


# ---------------------------------------------------------------------------
# Main Hybrid Core
# ---------------------------------------------------------------------------


class HybridNeuralCore(nn.Module):
    """
    Dynamic hybrid backbone controlled by AdvancedRoutingMap.
    """

    def __init__(
        self,
        input_dim: int = 1,
        max_seq_len: int = 2048,
        default_dim: int = 256,
        default_heads: int = 8,
        default_depth: int = 6,
        default_dropout: float = 0.1,
        use_macro_fusion: bool = False,
        macro_input_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.default_dim = default_dim

        self.dim = default_dim
        self.num_heads = default_heads
        self.depth = default_depth
        self.dropout = default_dropout

        self.input_proj = nn.Linear(input_dim, default_dim)
        self.pos_scale = nn.Parameter(torch.ones(1))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(default_dim, default_heads, dropout=default_dropout)
                for _ in range(default_depth)
            ]
        )
        self.final_norm = RMSNorm(default_dim)

        self.fib_module = FibonacciPathModule(default_dim)
        self.fourier_module = FourierFeatureModule(default_dim)
        self.extreme_module = ExtremeEventModule(default_dim)
        self.leverage_module = LeverageOptimizerModule(default_dim)
        self.garch_like = nn.GRU(default_dim, default_dim, batch_first=True)

        self.head_forecast = nn.Linear(default_dim, 1)
        self.head_dist = nn.Linear(default_dim, 2)
        self.head_classification = nn.Linear(default_dim, 4)

        self.case_embed = nn.Embedding(4, default_dim)

        self.macro_fusion = (
            MacroMicroFusion(
                micro_dim=default_dim,
                macro_input_dim=macro_input_dim,
                num_heads=4,
                dropout=default_dropout,
            )
            if use_macro_fusion
            else None
        )

        self._routing: AdvancedRoutingMap | None = None

    def configure_from_routing(self, routing: AdvancedRoutingMap) -> None:
        self._routing = routing

    def forward(
        self,
        x: Tensor,
        routing: AdvancedRoutingMap | None = None,
        macro_features: Tensor | None = None,
        return_aux: bool = False,
    ) -> dict[str, Tensor]:
        if routing is not None:
            self.configure_from_routing(routing)
        routing = routing or self._routing
        if routing is None:
            raise ValueError(
                "RoutingMap is required (pass it or call configure_from_routing first)"
            )

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        B, T, _ = x.shape
        h = self.input_proj(x)

        case_idx = list(CombinatorialCase).index(routing.dominant_case)
        case_emb = self.case_embed(torch.tensor(case_idx, device=x.device))
        h = h + case_emb * 0.1 * routing.preserve_order_strength

        if routing.positional_encoding_strength > 0.3:
            pos = torch.arange(T, device=x.device).float().unsqueeze(0).unsqueeze(-1) / T
            h = h + self.pos_scale * pos * routing.positional_encoding_strength * 0.05

        gates = routing.gates

        if gates.fourier_spectral > 0.2:
            h = self.fourier_module(h, strength=gates.fourier_spectral)

        for block in self.blocks:
            h = block(h)

        if gates.fibonacci_paths > 0.2:
            h = self.fib_module(h, strength=gates.fibonacci_paths)

        severity = None
        if gates.extreme_event_handler > 0.15:
            h, severity = self.extreme_module(h, strength=gates.extreme_event_handler)

        if gates.garch_volatility > 0.2:
            garch_out, _ = self.garch_like(h)
            h = h + gates.garch_volatility * garch_out

        fusion_strength = None
        if self.macro_fusion is not None:
            fusion_out = self.macro_fusion(h, macro_features=macro_features, routing=routing)
            h = fusion_out["fused_hidden"]
            fusion_strength = fusion_out["fusion_strength"]

        h = self.final_norm(h)

        forecast = self.head_forecast(h)
        dist_params = self.head_dist(h.mean(dim=1))
        regime = self.head_classification(h.mean(dim=1))

        leverage_action = None
        if gates.leverage_optimizer > 0.2:
            leverage_action = self.leverage_module(h, strength=gates.leverage_optimizer)

        out = {
            "forecast": forecast,
            "dist_params": dist_params,
            "regime_logits": regime,
            "hidden": h,
        }
        if severity is not None:
            out["severity"] = severity
        if leverage_action is not None:
            out["leverage_action"] = leverage_action
        if fusion_strength is not None:
            out["fusion_strength"] = fusion_strength

        if return_aux:
            out["routing_case"] = routing.dominant_case
            out["gates"] = gates
        return out


# ---------------------------------------------------------------------------
# Combinatorial-aware Loss
# ---------------------------------------------------------------------------


class CombinatorialLoss(nn.Module):
    def __init__(self, schedule: LossSchedule | None = None):
        super().__init__()
        self.schedule = schedule

    def forward(
        self,
        pred: dict[str, Tensor],
        target: Tensor,
        routing: AdvancedRoutingMap,
        severity_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if target.dim() == 2:
            target = target.unsqueeze(-1)

        forecast = pred["forecast"]
        min_t = min(forecast.shape[1], target.shape[1])
        forecast = forecast[:, :min_t]
        target = target[:, :min_t]

        task_loss = F.huber_loss(forecast, target, reduction="mean")

        dist_loss = torch.tensor(0.0, device=forecast.device)
        if "dist_params" in pred:
            mean, log_std = pred["dist_params"][:, 0], pred["dist_params"][:, 1]
            y = target[:, -1, 0]
            std = log_std.exp().clamp(min=1e-4)
            dist_loss = 0.5 * (torch.log(2 * math.pi * std**2) + ((y - mean) / std) ** 2).mean()

        extreme_loss = torch.tensor(0.0, device=forecast.device)
        if "severity" in pred and severity_target is not None:
            extreme_loss = F.binary_cross_entropy(pred["severity"], severity_target.float())

        struct_loss = torch.tensor(0.0, device=forecast.device)
        if routing.preserve_order_strength > 0.4 and "hidden" in pred:
            h = pred["hidden"]
            diff = h[:, 1:] - h[:, :-1]
            struct_loss = diff.pow(2).mean() * routing.preserve_order_strength

        sch = routing.loss if self.schedule is None else self.schedule
        total = (
            sch.w_task * task_loss
            + sch.w_combinatorial * dist_loss
            + sch.w_structure * struct_loss
            + sch.w_extreme * extreme_loss
            + sch.w_leverage * (pred.get("leverage_action", torch.tensor(0.0)).abs().mean() * 0.01)
        )

        return {
            "total": total,
            "task": task_loss.detach(),
            "dist": dist_loss.detach(),
            "extreme": extreme_loss.detach(),
            "structure": struct_loss.detach(),
        }


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------


def build_and_run_core(
    series: np.ndarray,
    routing: AdvancedRoutingMap,
    input_dim: int = 1,
    device: str = "cpu",
) -> dict[str, Any]:
    model = HybridNeuralCore(input_dim=input_dim)
    model.configure_from_routing(routing)
    model.to(device)
    model.eval()

    x = torch.tensor(series, dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(-1)
    elif x.dim() == 2:
        x = x.unsqueeze(0)

    with torch.no_grad():
        out = model(x, routing=routing, return_aux=True)
    return {k: v.cpu() if torch.is_tensor(v) else v for k, v in out.items()}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Hybrid Neural Core – Self Test")
    print("=" * 70)

    from combinatorial_gate import CombinatorialGate
    from diagnosis_layer import DiagnosisLayer

    rng = np.random.default_rng(42)
    t = np.arange(180)
    series = 0.015 * t + 2.5 * np.sin(2 * np.pi * t / 14) + rng.normal(0, 0.8, 180)
    series[90:105] -= 7.0

    diag = DiagnosisLayer().diagnose(series)
    gate_result = CombinatorialGate(temperature=0.5).route(diag)
    routing = gate_result.routing

    print(f"Routing dominant case: {routing.dominant_case}")
    print(f"Extreme gate: {routing.gates.extreme_event_handler:.3f}")
    print(f"Embed dim hint: {routing.embed_dim}, pathways: {routing.num_pathways}")

    model = HybridNeuralCore(input_dim=1)
    model.configure_from_routing(routing)

    x = torch.tensor(series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    out = model(x, routing=routing, return_aux=True)

    print("\nForward pass shapes:")
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {type(v).__name__}")

    loss_fn = CombinatorialLoss()
    target = x.clone()
    losses = loss_fn(out, target, routing)
    print("\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")

    print("\n" + "=" * 70)
    print("Hybrid Neural Core ready.")
    print("=" * 70)
