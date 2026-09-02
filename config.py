"""
CARN-X  --  Central Configuration  ("הקונפיגורציה מוברגת על הארכיטקטורה")
======================================================================
One object, ``CarnxConfig``, aggregates every sub-config in the system and the
shared magic numbers, so a single edit (or a ``carnx.toml``) reconfigures the
whole 8-layer pipeline instead of touching ten dataclasses in ten files.

Layer map (from `ארכיטקטורה והתגלמות לפריסת עבודה.txt`):

    L1  Diagnosis & Characterisation   diagnosis_layer.DiagnosisLayer
    L2  Combinatorial Gate             combinatorial_gate.CombinatorialGate
    L3  Strategic modules (gated)      forecast_net expert modules  (Fibonacci /
                                       Fourier / Wavelet / GARCH / ChangePoint /
                                       Extreme / MC-uncertainty ...)
    L4  Hybrid neural core             forecast_net.ForecastNet
    L5  Macro-Micro fusion             macro_micro_fusion.MacroMicroFusion
                                       (+ nav_premium features)
    L6  Combinatorial + financial loss forecast_net.forecast_loss  (weights from
                                       the Gate's LossSchedule)
    L7  Inference & decision           inference.py  (+ leverage_factorization,
                                       inference_decision_layer)
    L8  Evaluation & backtest          walk_forward + evaluation + baselines

The probability lab (prob_models) and the BTC 4-year cycle (btc_cycle) are
analytic companions, not part of the neural path.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict

from data_layer import DataConfig
from features import FeatureConfig
from targets import TargetConfig
from forecast_net import ModelConfig
from walk_forward import WFConfig
from baselines import BaselineConfig
from evaluation import EconConfig
from inference import InferenceConfig


# ---------------------------------------------------------------------------
# shared constants  (previously magic numbers scattered across modules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constants:
    trading_days_year: int = 252
    lookback_2y: int = 504
    riskmetrics_lambda: float = 0.94
    z_80: float = 1.2816  # one-sided 90% normal quantile (80% band)
    z_90: float = 1.6449
    z_95: float = 1.9600
    mc_paths: int = 20_000  # default Monte-Carlo path count
    btc_cycle_days: float = 1400.0


CONST = Constants()


# ---------------------------------------------------------------------------
# the aggregate
# ---------------------------------------------------------------------------


@dataclass
class CarnxConfig:
    """Every layer's config in one place. `carnx.build_pipeline(cfg)` threads it."""

    data: DataConfig = field(default_factory=DataConfig)  # L0/L1 input
    features: FeatureConfig = field(default_factory=FeatureConfig)  # L1
    targets: TargetConfig = field(default_factory=TargetConfig)  # L1
    model: ModelConfig = field(default_factory=ModelConfig)  # L3/L4
    walk_forward: WFConfig = field(default_factory=WFConfig)  # L8
    baselines: BaselineConfig = field(default_factory=BaselineConfig)  # L8
    economic: EconConfig = field(default_factory=EconConfig)  # L8
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # L7

    # cross-cutting toggles
    use_combinatorial_gate: bool = True  # L2 -> feeds L3 gates + L6 loss weights
    use_macro_fusion: bool = False  # L5
    use_loss_schedule: bool = True  # L6: take loss weights from the Gate
    seed: int = 0

    # ------------------------------------------------------------------
    def sync(self) -> CarnxConfig:
        """Push shared fields down so the sub-configs never disagree."""
        self.walk_forward.model = self.model
        self.walk_forward.targets = self.targets
        self.inference.model = self.model
        self.inference.targets = self.targets
        self.walk_forward.use_experts = self.use_combinatorial_gate
        self.inference.use_experts = self.use_combinatorial_gate
        self.model.use_experts = self.use_combinatorial_gate
        return self

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        def _d(v):
            if is_dataclass(v) and not isinstance(v, type):
                return {f.name: _d(getattr(v, f.name)) for f in fields(v)}
            if isinstance(v, (list, tuple)):
                return [_d(x) for x in v]
            if isinstance(v, dict):
                return {k: _d(x) for k, x in v.items()}
            return v

        return _d(self)

    def save(self, path: str = "./carnx_config.json") -> str:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str = "./carnx_config.json") -> CarnxConfig:
        raw = json.loads(Path(path).read_text())
        cfg = cls()
        for name, sub in raw.items():
            cur = getattr(cfg, name, None)
            if is_dataclass(cur) and isinstance(sub, dict):
                for k, v in sub.items():
                    if hasattr(cur, k) and not is_dataclass(getattr(cur, k)):
                        try:
                            setattr(
                                cur,
                                k,
                                type(getattr(cur, k))(v)
                                if not isinstance(getattr(cur, k), (list, tuple, dict))
                                else v,
                            )
                        except Exception:
                            setattr(cur, k, v)
            elif not is_dataclass(cur) and hasattr(cfg, name):
                setattr(cfg, name, sub)
        return cfg.sync()


# ready-made presets ---------------------------------------------------------


def default() -> CarnxConfig:
    return CarnxConfig().sync()


def fast() -> CarnxConfig:
    c = CarnxConfig()
    c.walk_forward = WFConfig(
        initial_train=504,
        step=90,
        max_folds=4,
        epochs=15,
        ensemble=1,
        seeds=(0,),
        retrain_every=1,
        tag="wf_fast",
    )
    c.inference = InferenceConfig(ensemble=2, seeds=(0, 1), epochs=15, patience=4)
    return c.sync()


def production() -> CarnxConfig:
    c = CarnxConfig()
    c.inference = InferenceConfig(ensemble=5, seeds=(0, 1, 2, 3, 4), epochs=45, patience=8)
    return c.sync()


def production_xl() -> CarnxConfig:
    """Highest-quality production training: 9-seed ensemble, long schedule with
    generous early-stopping patience.  Architecture is unchanged on purpose --
    on ~1.5k daily rows the reliable lever is ensembling + calibration, not
    capacity (a bigger net just overfits and hurts PIT calibration)."""
    c = CarnxConfig()
    c.inference = InferenceConfig(
        ensemble=9,
        seeds=tuple(range(9)),
        epochs=90,
        patience=14,
        lr=1.0e-3,
        weight_decay=2.0e-4,
        val_tail=140,
    )
    c.walk_forward = WFConfig(
        initial_train=504,
        step=21,
        embargo=20,
        retrain_every=3,
        ensemble=3,
        seeds=(0, 1, 2),
        epochs=60,
        patience=8,
        num_threads=4,
        tag="wf_xl",
    )
    return c.sync()


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Central Config -- Self Test")
    print("=" * 70)
    cfg = default()
    d = cfg.to_dict()
    print(f"\naggregated {len(d)} top-level keys:")
    for k, v in d.items():
        n = len(v) if isinstance(v, dict) else 1
        print(f"  {k:18} {'(' + str(n) + ' fields)' if isinstance(v, dict) else repr(v)}")
    p = cfg.save("/tmp/_carnx_cfg_test.json")
    back = CarnxConfig.load(p)
    assert back.model.d_model == cfg.model.d_model
    assert back.walk_forward.model is back.model, "sync() must alias model into walk_forward"
    import os

    os.remove(p)
    print("\nround-trip save/load OK · sync() aliases shared configs")
    print("presets: default() / fast() / production()")
    print("=" * 70)
