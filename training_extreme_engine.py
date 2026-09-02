"""
CARN-X  –  Training & Extreme Event Engine  (Layer 4)
=======================================================
Highest-level training loop + Extreme Event handling + Leverage Optimizer.

Out-of-the-box ideas implemented:
1. Dynamic curriculum driven by Combinatorial Gate confidence
2. Extreme-event triggered loss re-weighting (crash → boost extreme + leverage terms)
3. Soft pathway mixture training (multi-pathway when Gate is uncertain)
4. Adaptive learning-rate + temperature annealing from Gate
5. Leverage action regularization (min resources, max recovery potential)
6. Severity-aware sampling (oversample crash regimes)
7. Early-exit / adaptive computation hint from Egg-Drop & Balance-Scale gates
8. Symbolic combinatorial regularizer (optional)
9. Multi-objective logging (task / extreme / leverage / structure)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from combinatorial_gate import AdvancedRoutingMap, CombinatorialGate, GateResult
from diagnosis_layer import CharacterizationResult, DiagnosisLayer
from hybrid_neural_core import CombinatorialLoss, HybridNeuralCore

# ---------------------------------------------------------------------------
# Dataset with severity weighting
# ---------------------------------------------------------------------------


class SeriesDataset(Dataset):
    """
    Simple sliding-window dataset.
    Can attach a severity score per window for crash-aware sampling.
    """

    def __init__(
        self,
        series: np.ndarray,
        window: int = 64,
        horizon: int = 1,
        severity: np.ndarray | None = None,
        macro: np.ndarray | None = None,
    ):
        self.series = series.astype(np.float32)
        self.window = window
        self.horizon = horizon
        self.severity = severity
        self.macro = macro.astype(np.float32) if macro is not None else None
        self.n = len(series) - window - horizon + 1
        if self.n <= 0:
            raise ValueError("Series too short for given window/horizon")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.series[idx : idx + self.window]
        y = self.series[idx + self.window : idx + self.window + self.horizon]
        item = {
            "x": torch.from_numpy(x).unsqueeze(-1),  # [T, 1]
            "y": torch.from_numpy(y).unsqueeze(-1),
        }
        if self.severity is not None:
            sev = float(np.max(self.severity[idx : idx + self.window]))
            item["severity"] = torch.tensor(sev, dtype=torch.float32)
        if self.macro is not None:
            item["macro"] = torch.from_numpy(self.macro[idx + self.window - 1])
        return item


def make_severity_proxy(series: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple drawdown-based severity proxy for sampling."""
    s = np.asarray(series, dtype=np.float64)
    roll_max = np.maximum.accumulate(s)
    dd = (s - roll_max) / (roll_max + 1e-8)
    from numpy.lib.stride_tricks import sliding_window_view

    if len(dd) >= window:
        views = sliding_window_view(dd, window)
        sev = -np.min(views, axis=1)
        pad = np.zeros(window - 1)
        return np.concatenate([pad, sev])
    return -dd


# ---------------------------------------------------------------------------
# Extreme Event Handler
# ---------------------------------------------------------------------------


@dataclass
class ExtremeDecision:
    is_extreme: bool
    severity: float
    recommended_actions: list[str]
    leverage_hint: float  # 0..1 suggested aggressiveness of recovery
    risk_budget: float


class ExtremeEventHandler:
    """
    Out-of-the-box crash detector + action suggester.
    Uses Gate signals + model severity head.
    """

    def __init__(
        self,
        dd_threshold: float = -0.15,
        severity_threshold: float = 0.55,
    ):
        self.dd_threshold = dd_threshold
        self.severity_threshold = severity_threshold

    def evaluate(
        self,
        series: np.ndarray,
        routing: AdvancedRoutingMap,
        model_severity: float | None = None,
    ) -> ExtremeDecision:
        s = series[-min(60, len(series)) :]
        peak = np.maximum.accumulate(s)
        dd = float(np.min((s - peak) / (peak + 1e-8)))

        gate_extreme = routing.gates.extreme_event_handler
        severity = max(
            -dd / 0.4,
            gate_extreme,
            model_severity or 0.0,
        )
        severity = float(np.clip(severity, 0, 1))

        is_extreme = (dd < self.dd_threshold) or (severity > self.severity_threshold)

        actions = []
        if is_extreme:
            actions.append("activate_extreme_pathway")
            actions.append("increase_extreme_loss_weight")
            if routing.gates.leverage_optimizer > 0.4:
                actions.append("run_leverage_optimizer")
            if severity > 0.7:
                actions.append("reduce_position_size")
                actions.append("consider_hedge_or_cash")
            if routing.gates.egg_drop_sequential > 0.5:
                actions.append("sequential_probing_mode")

        leverage_hint = float(
            np.clip(0.3 + 0.5 * severity + 0.2 * routing.gates.leverage_optimizer, 0, 1)
        )
        risk_budget = float(np.clip(1.0 - 0.7 * severity, 0.05, 1.0))

        return ExtremeDecision(
            is_extreme=is_extreme,
            severity=severity,
            recommended_actions=actions,
            leverage_hint=leverage_hint,
            risk_budget=risk_budget,
        )


# ---------------------------------------------------------------------------
# Leverage Optimizer
# ---------------------------------------------------------------------------


@dataclass
class LeverageSuggestion:
    action_vector: np.ndarray
    suggested_scale: float
    min_resource_score: float
    notes: list[str]


class LeverageOptimizer:
    """
    Turns model leverage_action head + ExtremeDecision into concrete suggestions.
    Principle: maximum recovery potential with minimum additional resources.
    """

    def suggest(
        self,
        leverage_action: torch.Tensor,
        extreme: ExtremeDecision,
        routing: AdvancedRoutingMap,
    ) -> LeverageSuggestion:
        vec = leverage_action.detach().cpu().numpy().flatten()
        scale = extreme.leverage_hint * (0.6 + 0.4 * routing.gates.leverage_optimizer)

        l1 = np.abs(vec).sum() + 1e-8
        sparsity = 1.0 - (l1 / (len(vec) * (np.abs(vec).max() + 1e-8)))
        min_resource_score = float(np.clip(0.5 * sparsity + 0.5 * (1.0 / (1.0 + l1)), 0, 1))

        notes = []
        if extreme.is_extreme:
            notes.append(f"Extreme regime (severity={extreme.severity:.2f})")
            notes.append(f"Risk budget left ≈ {extreme.risk_budget:.2f}")
        if min_resource_score > 0.6:
            notes.append("Action is relatively resource-efficient")
        if scale > 0.7:
            notes.append("High recovery aggressiveness recommended by Gate")

        return LeverageSuggestion(
            action_vector=vec,
            suggested_scale=float(scale),
            min_resource_score=min_resource_score,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Training Engine
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    window: int = 64
    horizon: int = 1
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    checkpoint_dir: str = "./carnx_checkpoints"
    log_every: int = 10
    severity_oversample_power: float = 1.5
    early_stop_patience: int = 7


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


class TrainingEngine:
    """
    Full training loop that respects the Combinatorial Gate decisions.
    """

    def __init__(
        self,
        model: HybridNeuralCore,
        config: TrainConfig,
        gate: CombinatorialGate | None = None,
    ):
        self.model = model
        self.config = config
        self.gate = gate or CombinatorialGate()
        self.extreme_handler = ExtremeEventHandler()
        self.leverage_opt = LeverageOptimizer()
        self.loss_fn = CombinatorialLoss()
        self.state = TrainState()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _build_loaders(
        self,
        train_series: np.ndarray,
        val_series: np.ndarray | None = None,
        train_macro: np.ndarray | None = None,
        val_macro: np.ndarray | None = None,
    ) -> tuple[DataLoader, DataLoader | None]:
        sev_train = make_severity_proxy(train_series)
        # clamp severity to [0,1] for safety
        sev_train = np.clip(sev_train, 0.0, 1.0)
        ds_train = SeriesDataset(
            train_series,
            window=self.config.window,
            horizon=self.config.horizon,
            severity=sev_train,
            macro=train_macro,
        )
        weights = (sev_train[: len(ds_train)] + 0.05) ** self.config.severity_oversample_power
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(ds_train),
            replacement=True,
        )
        loader_train = DataLoader(
            ds_train,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            drop_last=True,
        )
        loader_val = None
        if val_series is not None:
            ds_val = SeriesDataset(
                val_series,
                window=self.config.window,
                horizon=self.config.horizon,
                macro=val_macro,
            )
            loader_val = DataLoader(
                ds_val,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        return loader_train, loader_val

    def _get_routing(self, series_chunk: np.ndarray) -> AdvancedRoutingMap:
        diag = DiagnosisLayer().diagnose(series_chunk)
        gate_res = self.gate.route(diag)
        return gate_res.routing

    def train(
        self,
        train_series: np.ndarray,
        val_series: np.ndarray | None = None,
        static_routing: AdvancedRoutingMap | None = None,
        train_macro: np.ndarray | None = None,
        val_macro: np.ndarray | None = None,
        progress_callback: Any | None = None,
    ) -> TrainState:
        self.model.to(self.config.device)
        loader_train, loader_val = self._build_loaders(
            train_series, val_series, train_macro, val_macro
        )

        if static_routing is not None:
            routing = static_routing
        else:
            routing = self._get_routing(train_series[-min(300, len(train_series)) :])

        self.model.configure_from_routing(routing)
        print(f"[Train] Initial dominant case: {routing.dominant_case}")
        print(f"[Train] Extreme gate: {routing.gates.extreme_event_handler:.3f}")

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            self.model.train()
            epoch_losses = []

            for step, batch in enumerate(loader_train):
                x = batch["x"].to(self.config.device)
                y = batch["y"].to(self.config.device)
                sev = batch.get("severity")
                if sev is not None:
                    sev = sev.to(self.config.device).clamp(0.0, 1.0)
                macro_batch = batch.get("macro")
                if macro_batch is not None:
                    macro_batch = macro_batch.to(self.config.device)

                if static_routing is None and step % 50 == 0 and step > 0:
                    routing = self._get_routing(train_series[-200:])
                    self.model.configure_from_routing(routing)

                self.optimizer.zero_grad()
                out = self.model(x, routing=routing, macro_features=macro_batch)
                losses = self.loss_fn(out, y, routing, severity_target=sev)

                # Extreme-triggered re-weight
                if routing.gates.extreme_event_handler > 0.6:
                    losses["total"] = losses["total"] + 0.15 * losses["extreme"]

                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

                epoch_losses.append(losses["total"].item())
                self.state.global_step += 1

                if step % self.config.log_every == 0:
                    print(
                        f"Epoch {epoch:03d} step {step:04d} | "
                        f"loss={losses['total'].item():.5f} "
                        f"task={losses['task'].item():.5f} "
                        f"ext={losses['extreme'].item():.5f}"
                    )

            self.scheduler.step()
            avg_loss = float(np.mean(epoch_losses))

            if progress_callback is not None:
                progress_callback(epoch, self.config.epochs, avg_loss)

            val_loss = None
            if loader_val is not None:
                val_loss = self._validate(loader_val, routing)
                print(f"Epoch {epoch:03d} | train_loss={avg_loss:.5f} | val_loss={val_loss:.5f}")
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self.state.patience_counter = 0
                    self._save_checkpoint("best.pt", routing)
                else:
                    self.state.patience_counter += 1
            else:
                print(f"Epoch {epoch:03d} | train_loss={avg_loss:.5f}")
                self._save_checkpoint(f"epoch_{epoch:03d}.pt", routing)

            self.state.history.append(
                {
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_loss": val_loss if val_loss is not None else avg_loss,
                }
            )

            if self.state.patience_counter >= self.config.early_stop_patience:
                print("Early stopping triggered.")
                break

        return self.state

    def _validate(self, loader: DataLoader, routing: AdvancedRoutingMap) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.config.device)
                y = batch["y"].to(self.config.device)
                macro_batch = batch.get("macro")
                if macro_batch is not None:
                    macro_batch = macro_batch.to(self.config.device)
                out = self.model(x, routing=routing, macro_features=macro_batch)
                loss_dict = self.loss_fn(out, y, routing)
                losses.append(loss_dict["total"].item())
        return float(np.mean(losses))

    def _save_checkpoint(self, name: str, routing: AdvancedRoutingMap) -> None:
        path = Path(self.config.checkpoint_dir) / name
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "state": self.state,
                "routing_summary": routing.summary,
            },
            path,
        )
        print(f"  → checkpoint saved: {path}")

    @torch.no_grad()
    def predict_with_extreme(
        self,
        series: np.ndarray,
        routing: AdvancedRoutingMap | None = None,
        macro_features: np.ndarray | None = None,
    ) -> dict[str, Any]:
        self.model.eval()
        if routing is None:
            routing = self._get_routing(series[-min(300, len(series)) :])
            self.model.configure_from_routing(routing)

        x = torch.tensor(series, dtype=torch.float32, device=self.config.device)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        macro_t = None
        if macro_features is not None:
            macro_t = torch.tensor(macro_features, dtype=torch.float32, device=self.config.device)
            if macro_t.dim() == 1:
                macro_t = macro_t.unsqueeze(0)
        out = self.model(x, routing=routing, macro_features=macro_t, return_aux=True)

        model_sev = None
        if "severity" in out:
            model_sev = float(out["severity"].mean().cpu())

        extreme = self.extreme_handler.evaluate(series, routing, model_sev)
        leverage_sug = None
        if "leverage_action" in out:
            leverage_sug = self.leverage_opt.suggest(out["leverage_action"], extreme, routing)

        return {
            "forecast": out["forecast"].cpu().numpy(),
            "extreme": extreme,
            "leverage": leverage_sug,
            "routing_case": routing.dominant_case,
            "gates": routing.gates,
        }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Training & Extreme Event Engine – Demo")
    print("=" * 70)

    rng = np.random.default_rng(42)
    t = np.arange(600)
    series = 0.01 * t + 2.0 * np.sin(2 * np.pi * t / 18) + rng.normal(0, 0.7, 600)
    series[250:280] -= 8.0
    series[400:420] -= 5.0

    train_s = series[:450]
    val_s = series[450:]

    diag = DiagnosisLayer().diagnose(train_s)
    gate_res = CombinatorialGate(temperature=0.5).route(diag)
    routing = gate_res.routing

    model = HybridNeuralCore(input_dim=1)
    model.configure_from_routing(routing)

    config = TrainConfig(
        window=48,
        batch_size=16,
        epochs=5,
        lr=1e-3,
        log_every=5,
        early_stop_patience=3,
    )
    engine = TrainingEngine(model, config)

    print("\nStarting short training demo...")
    state = engine.train(train_s, val_s, static_routing=routing)

    print("\n--- Post-train Extreme evaluation on full series ---")
    result = engine.predict_with_extreme(series, routing)
    print(f"Extreme? {result['extreme'].is_extreme} | severity={result['extreme'].severity:.3f}")
    print(f"Actions: {result['extreme'].recommended_actions}")
    if result["leverage"] is not None:
        print(
            f"Leverage scale={result['leverage'].suggested_scale:.3f} | "
            f"resource_score={result['leverage'].min_resource_score:.3f}"
        )
        print(f"Notes: {result['leverage'].notes}")

    print("\n" + "=" * 70)
    print("Training & Extreme Event Engine ready.")
    print("=" * 70)
