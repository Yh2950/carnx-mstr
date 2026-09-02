"""
CARN-X  --  Probabilistic Forecast Network
==========================================
The actual neural network. A hybrid temporal + tabular model that outputs a
full predictive *distribution* of the vol-normalized forward log-return for
each horizon, not a point estimate.

    seq path :  ~12 curated causal series  -> [B, W, C_seq]
                RoPE Transformer encoder (small, heavily regularized)
                -> last-token + attention-pooled summary

    tab path :  full engineered feature vector at time t -> [B, F]
                MLP -> compressed embedding

    fusion   :  concat(seq_summary, tab_embed) -> trunk MLP

    heads    :  per horizon h in {1, 5, 20}
                - Student-t params (mu, log_sigma, log_nu)   [distributional]
                - direction logit                            [calibrated sign]
                - forward realized-vol (softplus)            [vol forecast]

Optional gated expert modules (Fourier / recurrent / extreme) can be switched
on from a CombinatorialGate routing map; they are ablatable via
``ModelConfig.use_experts`` so their value can be measured rather than assumed.

Loss = w_nll * Student-t NLL(y_scaled)
     + w_pin * pinball over the t-quantiles
     + w_dir * BCE(direction)
     + w_vol * Huber(forward vol)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# curated series reused for the temporal encoder (must exist in features.py output)
DEFAULT_SEQ_FEATURES: tuple[str, ...] = (
    "mstr_ret_1",
    "mstr_rv_20",
    "mstr_ewmavol",
    "mstr_dd_60",
    "mstr_pricez_20",
    "btc_ret_1",
    "btc_rv_20",
    "ibit_ret_1",
    "mstr_btc_resid_20",
    "mstr_btc_beta_60",
    "vix_level",
    "tnx_chg_5",
)

PINBALL_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

# bump whenever ForecastNet's parameter set changes -- guards stale checkpoints
NET_ARCH_VERSION = 2  # v2: full 15-gate Routing + wavelet/garch/changepoint experts + hidden out


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    seq_feature_names: tuple[str, ...] = DEFAULT_SEQ_FEATURES
    window: int = 64
    horizons: tuple[int, ...] = (1, 5, 20)

    d_model: int = 96
    n_heads: int = 4
    depth: int = 3
    mlp_ratio: float = 2.0
    dropout: float = 0.2
    tab_hidden: int = 128
    tab_embed: int = 64
    trunk_hidden: int = 128

    use_experts: bool = True
    expert_fourier_freqs: int = 16

    nu_min: float = 2.2  # keep the Student-t variance finite
    nu_max: float = 60.0
    sigma_min: float = 0.05
    sigma_max: float = 6.0

    w_nll: float = 1.0
    w_pinball: float = 0.3
    w_direction: float = 0.4
    w_vol: float = 0.5


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device, dtype) -> tuple[Tensor, Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype)[None, None], emb.sin().to(dtype)[None, None]


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = self.rope(T, x.device, x.dtype)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop.p if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, dropout)
        self.n2 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.drop(self.mlp(self.n2(x)))
        return x


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim**-0.5

    def forward(self, x: Tensor) -> Tensor:
        attn = (self.q @ x.transpose(-2, -1)) * self.scale  # [B,1,T]
        w = attn.softmax(dim=-1)
        return (w @ x).squeeze(1)  # [B,D]


# ---------------------------------------------------------------------------
# optional gated experts
# ---------------------------------------------------------------------------


class FourierExpert(nn.Module):
    def __init__(self, dim: int, n_freqs: int):
        super().__init__()
        self.freq = nn.Linear(dim, n_freqs * 2)
        self.out = nn.Linear(n_freqs * 2, dim)

    def forward(self, x: Tensor, strength: float) -> Tensor:
        s, c = self.freq(x).chunk(2, dim=-1)
        return x + strength * self.out(torch.cat([torch.sin(s), torch.cos(c)], dim=-1))


class RecurrentExpert(nn.Module):
    """GRU path -- a Fibonacci-style sequential recurrence inductive bias."""

    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.gate = nn.Linear(dim, 1)

    def forward(self, x: Tensor, strength: float) -> Tensor:
        out, _ = self.gru(x)
        g = torch.sigmoid(self.gate(x.mean(dim=1, keepdim=True))) * strength
        return x + g * out


class WaveletExpert(nn.Module):
    """Multi-resolution: parallel depthwise dilated convs (à trous) -- a learned
    stand-in for the wavelet multi-scale decomposition the Gate asks for."""

    def __init__(self, dim: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(dim, dim, 3, padding=d, dilation=d, groups=dim) for d in dilations
        )
        self.mix = nn.Linear(dim * len(dilations), dim)

    def forward(self, x: Tensor, strength: float) -> Tensor:
        xt = x.transpose(1, 2)  # [B, D, T]
        feats = torch.cat([c(xt).transpose(1, 2) for c in self.convs], dim=-1)
        return x + strength * self.mix(feats)


class GarchExpert(nn.Module):
    """Volatility-clustering path: a GRU that carries a conditional-variance-like
    state and modulates the hidden stream (the Gate's garch_volatility)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x: Tensor, strength: float) -> Tensor:
        out, _ = self.gru(x)
        return x + strength * out


class ChangePointExpert(nn.Module):
    """Adaptive to regime shifts: attends more to the recent tail when local
    variability jumps (the Gate's change_point_adaptive)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, strength: float) -> Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1).unsqueeze(-1)  # [B,T,1]
        ctx = (w * x).sum(dim=1, keepdim=True)
        return x + strength * self.proj(ctx.expand_as(x))


# ---------------------------------------------------------------------------
# main network
# ---------------------------------------------------------------------------

_GATE_NAMES = (
    "fibonacci_paths",
    "euclid_gcd_cycles",
    "balance_scale_search",
    "egg_drop_sequential",
    "poor_pigs_parallel_info",
    "monte_carlo_uncertainty",
    "graph_algorithms",
    "neural_ode_flow",
    "fourier_spectral",
    "extreme_event_handler",
    "game_theoretic_nash",
    "garch_volatility",
    "wavelet_multiresolution",
    "change_point_adaptive",
    "leverage_optimizer",
)


@dataclass
class Routing:
    """The full CARN-X routing map the Combinatorial Gate (L2) hands to the core:
    all 15 attention gates + the LossSchedule weights (L6) + the dominant
    combinatorial case. Plain floats -- no torch, safe to pickle / log."""

    # -- 15 attention gates --
    fibonacci_paths: float = 0.0
    euclid_gcd_cycles: float = 0.0
    balance_scale_search: float = 0.0
    egg_drop_sequential: float = 0.0
    poor_pigs_parallel_info: float = 0.0
    monte_carlo_uncertainty: float = 0.0
    graph_algorithms: float = 0.0
    neural_ode_flow: float = 0.0
    fourier_spectral: float = 0.0
    extreme_event_handler: float = 0.0
    game_theoretic_nash: float = 0.0
    garch_volatility: float = 0.0
    wavelet_multiresolution: float = 0.0
    change_point_adaptive: float = 0.0
    leverage_optimizer: float = 0.0
    # -- L6 loss schedule (from the Gate) --
    w_task: float = 1.0
    w_combinatorial: float = 0.20
    w_structure: float = 0.10
    w_extreme: float = 0.15
    w_leverage: float = 0.08
    w_consistency: float = 0.05
    w_information: float = 0.08
    risk_aversion: float = 0.30
    # -- combinatorial case --
    dominant_case: str = ""
    preserve_order_strength: float = 0.5

    # back-compat: the 3 names the old ExpertStrengths exposed
    @property
    def fourier(self) -> float:
        return self.fourier_spectral

    @property
    def recurrent(self) -> float:
        return self.fibonacci_paths

    @property
    def extreme(self) -> float:
        return self.extreme_event_handler

    def gate_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in _GATE_NAMES}

    @classmethod
    def from_routing(cls, rmap) -> Routing:
        g, L = rmap.gates, rmap.loss
        return cls(
            **{n: float(getattr(g, n)) for n in _GATE_NAMES},
            w_task=float(L.w_task),
            w_combinatorial=float(L.w_combinatorial),
            w_structure=float(L.w_structure),
            w_extreme=float(L.w_extreme),
            w_leverage=float(L.w_leverage),
            w_consistency=float(L.w_consistency),
            w_information=float(L.w_information),
            risk_aversion=float(L.risk_aversion),
            dominant_case=str(getattr(rmap, "dominant_case", "")),
            preserve_order_strength=float(getattr(rmap, "preserve_order_strength", 0.5)),
        )


ExpertStrengths = Routing  # back-compat alias


class ForecastNet(nn.Module):
    def __init__(self, cfg: ModelConfig, n_tab_features: int):
        super().__init__()
        self.cfg = cfg
        self.n_horizons = len(cfg.horizons)
        c_seq = len(cfg.seq_feature_names)

        self.seq_in = nn.Linear(c_seq, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.seq_norm = RMSNorm(cfg.d_model)
        self.pool = AttentionPool(cfg.d_model)

        # L3 strategic gated modules (each ablatable via the Gate strength)
        self.fourier_expert = FourierExpert(cfg.d_model, cfg.expert_fourier_freqs)
        self.recurrent_expert = RecurrentExpert(cfg.d_model)  # fibonacci_paths
        self.wavelet_expert = WaveletExpert(cfg.d_model)  # wavelet_multiresolution
        self.garch_expert = GarchExpert(cfg.d_model)  # garch_volatility
        self.changepoint_expert = ChangePointExpert(cfg.d_model)  # change_point_adaptive
        self.extreme_gate = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.SiLU())
        self.mc_dropout = nn.Dropout(cfg.dropout)  # monte_carlo_uncertainty

        self.tab_mlp = nn.Sequential(
            nn.LayerNorm(n_tab_features),
            nn.Linear(n_tab_features, cfg.tab_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.tab_hidden, cfg.tab_embed),
            nn.GELU(),
        )

        trunk_in = cfg.d_model * 2 + cfg.tab_embed
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, cfg.trunk_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.trunk_hidden, cfg.trunk_hidden),
            nn.GELU(),
        )

        # per-horizon heads
        self.head_dist = nn.ModuleList(
            [nn.Linear(cfg.trunk_hidden, 3) for _ in cfg.horizons]  # mu, log_sigma, log_nu
        )
        self.head_dir = nn.ModuleList([nn.Linear(cfg.trunk_hidden, 1) for _ in cfg.horizons])
        self.head_vol = nn.ModuleList([nn.Linear(cfg.trunk_hidden, 1) for _ in cfg.horizons])

        self.apply(self._init)
        # bias the distribution heads so training starts near sigma~1, nu~12
        for lin in self.head_dist:
            with torch.no_grad():
                lin.bias[:] = torch.tensor([0.0, -1.66, -1.30])

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.7)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_seq: Tensor,  # [B, W, C_seq]
        x_tab: Tensor,  # [B, F]
        experts: ExpertStrengths | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.cfg
        rt = experts or Routing()

        def _on(v: float) -> bool:
            return cfg.use_experts and v > 0.15

        h = self.seq_in(x_seq)
        if _on(rt.fourier_spectral):
            h = self.fourier_expert(h, min(rt.fourier_spectral, 1.0))
        if _on(rt.wavelet_multiresolution):
            h = self.wavelet_expert(h, min(rt.wavelet_multiresolution, 1.0))
        for blk in self.blocks:
            h = blk(h)
        if _on(rt.fibonacci_paths):
            h = self.recurrent_expert(h, min(rt.fibonacci_paths, 1.0))
        if _on(rt.garch_volatility):
            h = self.garch_expert(h, min(rt.garch_volatility, 1.0))
        if _on(rt.change_point_adaptive):
            h = self.changepoint_expert(h, min(rt.change_point_adaptive, 1.0))
        h = self.seq_norm(h)
        if _on(rt.extreme_event_handler):
            h = h + rt.extreme_event_handler * self.extreme_gate(h)
        if _on(rt.monte_carlo_uncertainty):
            h = self.mc_dropout(h)

        seq_summary = torch.cat([h[:, -1], self.pool(h)], dim=-1)  # [B, 2*d_model]
        tab_embed = self.tab_mlp(x_tab)
        z = self.trunk(torch.cat([seq_summary, tab_embed], dim=-1))

        mu, log_sigma, log_nu, dir_logit, fwd_vol = [], [], [], [], []
        for i in range(self.n_horizons):
            d = self.head_dist[i](z)
            mu.append(d[:, 0])
            sig = cfg.sigma_min + (cfg.sigma_max - cfg.sigma_min) * torch.sigmoid(d[:, 1])
            log_sigma.append(sig.log())
            nu = cfg.nu_min + (cfg.nu_max - cfg.nu_min) * torch.sigmoid(d[:, 2])
            log_nu.append(nu.log())
            dir_logit.append(self.head_dir[i](z).squeeze(-1))
            # forward-vol head predicts log( realized_fwd_vol / trailing_daily_vol ),
            # a quantity centred near 0 -- far easier to learn than a raw vol level
            fwd_vol.append(2.5 * torch.tanh(self.head_vol[i](z).squeeze(-1)))

        return {
            "mu": torch.stack(mu, dim=1),  # [B, H]
            "sigma": torch.stack(log_sigma, dim=1).exp(),
            "nu": torch.stack(log_nu, dim=1).exp(),
            "dir_logit": torch.stack(dir_logit, dim=1),
            "fwd_vol_lr": torch.stack(fwd_vol, dim=1),  # log-ratio to trailing vol
            "hidden": h,  # [B, W, d_model]  (L6 structure term)
        }


# ---------------------------------------------------------------------------
# distribution math
# ---------------------------------------------------------------------------


def student_t_nll(y: Tensor, mu: Tensor, sigma: Tensor, nu: Tensor) -> Tensor:
    """Negative log-likelihood of y under Student-t(mu, sigma, nu). Elementwise."""
    z = (y - mu) / sigma
    log_c = (
        torch.lgamma((nu + 1.0) / 2.0)
        - torch.lgamma(nu / 2.0)
        - 0.5 * torch.log(nu * math.pi)
        - torch.log(sigma)
    )
    log_p = log_c - (nu + 1.0) / 2.0 * torch.log1p(z * z / nu)
    return -log_p


def student_t_quantile(q: float, mu: Tensor, sigma: Tensor, nu: Tensor) -> Tensor:
    """Approximate Student-t quantile (Cornish-Fisher on the normal quantile).

    Good enough for a pinball penalty; exact icdf is not available in torch.
    """
    from math import sqrt

    # normal quantile via erfinv
    zq = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * q - 1.0, device=mu.device))
    g1 = (zq**3 + zq) / 4.0
    g2 = (5 * zq**5 + 16 * zq**3 + 3 * zq) / 96.0
    t = zq + g1 / nu + g2 / (nu**2)
    return mu + sigma * t


def pinball_loss(
    y: Tensor,
    mu: Tensor,
    sigma: Tensor,
    nu: Tensor,
    quantiles: tuple[float, ...] = PINBALL_QUANTILES,
) -> Tensor:
    total = 0.0
    for q in quantiles:
        pred_q = student_t_quantile(q, mu, sigma, nu)
        err = y - pred_q
        total = total + torch.maximum(q * err, (q - 1.0) * err)
    return total / len(quantiles)


# ---------------------------------------------------------------------------
# combined loss
# ---------------------------------------------------------------------------


def forecast_loss(
    out: dict[str, Tensor],
    y_scaled: Tensor,  # [B, H]
    y_sign: Tensor,  # [B, H]  in {0,1}
    y_fwdvol: Tensor,  # [B, H]  log-ratio  log(fwd_vol / trailing_daily_vol)
    cfg: ModelConfig,
    sample_w: Tensor | None = None,  # [B]
    routing: Routing | None = None,  # L6: pull the loss weights from the Gate
) -> dict[str, Tensor]:
    mu, sigma, nu = out["mu"], out["sigma"], out["nu"]

    # mask any non-finite target entry so it contributes nothing
    m_ret = torch.isfinite(y_scaled).float()
    m_sign = torch.isfinite(y_sign).float()
    m_vol = torch.isfinite(y_fwdvol).float()
    y_scaled = torch.nan_to_num(y_scaled)
    y_sign = torch.nan_to_num(y_sign)
    y_fwdvol = torch.nan_to_num(y_fwdvol)

    nll = student_t_nll(y_scaled, mu, sigma, nu) * m_ret  # [B, H]
    pin = pinball_loss(y_scaled, mu, sigma, nu) * m_ret  # [B, H]
    dir_bce = (
        F.binary_cross_entropy_with_logits(out["dir_logit"], y_sign, reduction="none") * m_sign
    )  # [B, H]
    vol_h = F.mse_loss(out["fwd_vol_lr"], y_fwdvol, reduction="none") * m_vol

    # L2/L5-structure term: penalise jagged hidden trajectories when the Gate
    # says order matters (uses the aux "hidden" if the net returned it)
    struct = torch.tensor(0.0, device=nll.device)
    if routing is not None and "hidden" in out and routing.w_structure > 0:
        hdn = out["hidden"]
        struct = (hdn[:, 1:] - hdn[:, :-1]).pow(2).mean() * float(routing.preserve_order_strength)

    # weights: from the Gate's LossSchedule if provided, else the fixed cfg ones
    if routing is not None and getattr(cfg, "use_experts", True):
        w_nll, w_pin, w_vol = routing.w_task, routing.w_combinatorial, cfg.w_vol
        w_dir, w_struct = cfg.w_direction, routing.w_structure
        # risk-aversion up-weights the extreme (large-|y|) samples in the NLL
        ra = 1.0 + 2.0 * float(routing.risk_aversion) * y_scaled.abs().mean(dim=1)
    else:
        w_nll, w_pin, w_dir, w_vol, w_struct = (
            cfg.w_nll,
            cfg.w_pinball,
            cfg.w_direction,
            cfg.w_vol,
            0.0,
        )
        ra = torch.ones(nll.shape[0], device=nll.device)

    per_sample = (
        ra
        * (
            w_nll * nll.mean(dim=1)
            + w_pin * pin.mean(dim=1)
            + w_dir * dir_bce.mean(dim=1)
            + w_vol * vol_h.mean(dim=1)
        )
        + w_struct * struct
    )  # [B]

    if sample_w is not None:
        w = sample_w / sample_w.mean().clamp_min(1e-6)
        total = (per_sample * w).mean()
    else:
        total = per_sample.mean()

    return {
        "total": total,
        "nll": nll.mean().detach(),
        "pinball": pin.mean().detach(),
        "direction_bce": dir_bce.mean().detach(),
        "vol_huber": vol_h.mean().detach(),
        "structure": struct.detach() if torch.is_tensor(struct) else torch.tensor(0.0),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X ForecastNet -- Self Test")
    print("=" * 70)

    torch.manual_seed(0)
    cfg = ModelConfig()
    B, W, F_tab = 8, cfg.window, 120
    net = ForecastNet(cfg, n_tab_features=F_tab)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\nparameters       : {n_params:,}")

    x_seq = torch.randn(B, W, len(cfg.seq_feature_names))
    x_tab = torch.randn(B, F_tab)
    out = net(
        x_seq,
        x_tab,
        Routing(
            fourier_spectral=0.6,
            fibonacci_paths=0.5,
            extreme_event_handler=0.7,
            wavelet_multiresolution=0.4,
            garch_volatility=0.5,
            change_point_adaptive=0.3,
            w_task=1.0,
            w_combinatorial=0.3,
            w_structure=0.15,
            risk_aversion=0.4,
            preserve_order_strength=0.6,
        ),
    )
    for k, v in out.items():
        print(f"  {k:<10} {tuple(v.shape)}  range=[{v.min():+.2f}, {v.max():+.2f}]")

    y_scaled = torch.randn(B, len(cfg.horizons))
    y_sign = (y_scaled > 0).float()
    y_fwdvol = torch.rand(B, len(cfg.horizons)) * 0.05
    losses = forecast_loss(out, y_scaled, y_sign, y_fwdvol, cfg, sample_w=torch.rand(B))
    print("\nlosses:")
    for k, v in losses.items():
        print(f"  {k:<14} {v.item():.5f}")

    losses["total"].backward()
    grad_norm = math.sqrt(
        sum(p.grad.pow(2).sum().item() for p in net.parameters() if p.grad is not None)
    )
    print(f"\nbackward OK, grad-norm={grad_norm:.3f}")

    print("\n" + "=" * 70)
    print("ForecastNet ready.")
    print("=" * 70)
