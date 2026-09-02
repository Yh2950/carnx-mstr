"""
CARN-X  --  8-Layer Orchestrator  (Combinatorial Adaptive Routing Network – Extreme)
===================================================================================
The named realisation of the architecture in
`ארכיטקטורה והתגלמות לפריסת עבודה.txt`. Every method IS a layer; the config
(config.CarnxConfig) is threaded through all of them.

    L1  diagnose(series)      -> CharacterizationResult      diagnosis_layer
    L2  route(diag)           -> Routing (4 cases + 15 gates + loss schedule)
    L3  (gates consumed inside L4's expert modules -- Fibonacci/Fourier/Wavelet/
         GARCH/ChangePoint/Extreme/MC-uncertainty ...)
    L4  core                  -> ForecastNet  (RoPE transformer + tabular MLP)
    L5  fusion                -> MacroMicroFusion   (optional; NAV-premium macro)
    L6  loss                  -> forecast_loss weighted by L2's LossSchedule
    L7  decide(dist)          -> action / position / leverage recommendation
    L8  evaluate()            -> walk-forward OOS + baselines + economic metrics

Use:
    from carnx import CARNX
    from config import production
    cx = CARNX(production())
    cx.fit()                      # trains the production ensemble
    print(cx.latest())            # calibrated forecast for the last bar
    cx.evaluate(fast=True)        # walk-forward proof
    print(cx.conformance())       # which layers are actually wired
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import CarnxConfig, default
from dataset import AssembledData, assemble
from forecast_net import DEFAULT_SEQ_FEATURES, Routing


# ---------------------------------------------------------------------------


@dataclass
class LayerStatus:
    layer: str
    name: str
    module: str
    wired: bool
    note: str = ""


class CARNX:
    def __init__(self, cfg: CarnxConfig | None = None):
        self.cfg = (cfg or default()).sync()
        self._data: AssembledData | None = None
        self._model = None  # inference.ProductionModel
        self._routing: Routing | None = None

    # ---- data (L0) ----------------------------------------------------
    def data(self, refresh: bool = False, require_targets: bool = False) -> AssembledData:
        if self._data is None or refresh:
            self._data = assemble(
                feature_cfg=self.cfg.features,
                target_cfg=self.cfg.targets,
                data_cfg=self.cfg.data,
                seq_feature_names=self.cfg.model.seq_feature_names or DEFAULT_SEQ_FEATURES,
                require_targets=require_targets,
            )
        return self._data

    # ---- L1: diagnosis ---------------------------------------------------
    def diagnose(self, series: np.ndarray | None = None):
        from diagnosis_layer import DiagnosisLayer

        s = series if series is not None else self.data().close.to_numpy()
        return DiagnosisLayer().diagnose(s[-min(400, len(s)) :])

    # ---- L2: combinatorial gate -> routing -----------------------------
    def route(self, diag=None) -> Routing:
        if not self.cfg.use_combinatorial_gate:
            self._routing = Routing()
            return self._routing
        from combinatorial_gate import CombinatorialGate

        diag = diag or self.diagnose()
        rmap = (
            CombinatorialGate(
                **self.cfg.walk_forward.__dict__.get("gate_kwargs", {})
                if hasattr(self.cfg.walk_forward, "gate_kwargs")
                else {}
            )
            .route(diag)
            .routing
            if False
            else CombinatorialGate(temperature=0.5).route(diag).routing
        )
        self._routing = Routing.from_routing(rmap)
        return self._routing

    # ---- L4 + L6: fit the core with the gate-weighted loss ------------
    def fit(self, progress=None):
        from inference import fit_production_model

        self._model = fit_production_model(
            self.data(require_targets=False), self.cfg.inference, progress=progress
        )
        return self._model

    def load(self, path: str | None = None):
        from inference import load_production_model

        self._model = load_production_model(self.data(require_targets=False), path)
        return self._model

    def save(self, path: str | None = None) -> str:
        from inference import save_production_model

        assert self._model is not None, "fit() or load() first"
        return save_production_model(self._model, n_tab_features=self.data().X.shape[1], path=path)

    # ---- L7: predictive distribution + decision ----------------------
    def latest(self) -> pd.DataFrame:
        from inference import predict_distribution

        assert self._model is not None, "fit() or load() first"
        return predict_distribution(self._model, self.data(require_targets=False))

    def monte_carlo(self, horizon: int = 20, n_paths: int | None = None):
        from config import CONST
        from inference import monte_carlo_paths

        assert self._model is not None, "fit() or load() first"
        return monte_carlo_paths(
            self._model,
            self.data(require_targets=False),
            horizon=horizon,
            n_paths=n_paths or CONST.mc_paths,
        )

    def decide(self, current_position: float = 0.0):
        """L7 decision using the legacy InferenceDecisionLayer + leverage math."""
        from diagnosis_layer import DiagnosisLayer
        from inference_decision_layer import InferenceDecisionLayer, full_inference_pipeline
        from leverage_factorization import LeverageFactorization
        from training_extreme_engine import ExtremeEventHandler, LeverageOptimizer

        d = self.data(require_targets=False)
        series = d.close.reindex(d.index).to_numpy(float)[-max(self.cfg.model.window, 64) :]
        self.route()
        # reuse the trained core's first net as the decision core
        net = self._model.nets[0] if self._model else None
        if net is None:
            self.load()
            net = self._model.nets[0]
        diag = DiagnosisLayer().diagnose(series)
        return full_inference_pipeline(
            series=series,
            model=net,
            routing=self._raw_routing(),
            extreme_handler=ExtremeEventHandler(),
            leverage_optimizer=LeverageOptimizer(),
            decision_layer=InferenceDecisionLayer(leverage_factorizer=LeverageFactorization()),
            current_position=current_position,
            volatility_profile=diag.volatility,
        )

    def _raw_routing(self):
        from combinatorial_gate import CombinatorialGate

        return CombinatorialGate(temperature=0.5).route(self.diagnose()).routing

    # ---- L8: evaluation ----------------------------------------------
    def evaluate(self, fast: bool = False) -> dict[str, Any]:
        from evaluation import evaluate
        from walk_forward import run_walk_forward

        wf = self.cfg.walk_forward
        if fast:
            wf = type(wf)(
                initial_train=504,
                step=90,
                max_folds=3,
                epochs=12,
                ensemble=1,
                seeds=(0,),
                retrain_every=1,
                tag="wf_carnx_fast",
                model=self.cfg.model,
                targets=self.cfg.targets,
                use_experts=self.cfg.use_combinatorial_gate,
            )
        res = run_walk_forward(wf, data=self.data(require_targets=True))
        return {"oos": res.oos, "report": evaluate(res.oos), "run_path": res.run_path}

    # ---- BTC 4-year cycle companion --------------------------------
    def btc_cycle_projection(self, target_date, current_cycle_weight: float = 0.6):
        from btc_cycle import (
            fit_cycle_template,
            load_btc_history,
            macro_state,
            mstr_path_from_btc,
            project_btc,
        )

        btc = load_btc_history()
        from data_layer import build_panel

        pnl = build_panel(self.cfg.data)
        ms = macro_state(pnl)
        proj = project_btc(
            btc,
            target_date,
            fit_cycle_template(btc),
            current_cycle_weight=current_cycle_weight,
            macro=ms,
        )
        mproj = mstr_path_from_btc(pnl["mstr_close"].dropna(), pnl["btc_close"].dropna(), proj)
        return proj, mproj

    # ---- conformance report --------------------------------------
    def conformance(self) -> list[LayerStatus]:
        r = self._routing
        return [
            LayerStatus(
                "L1",
                "Diagnosis & Characterisation",
                "diagnosis_layer",
                True,
                "GARCH / STL / EVT / change-points / order-repetition-symmetry",
            ),
            LayerStatus(
                "L2",
                "Combinatorial Gate",
                "combinatorial_gate",
                self.cfg.use_combinatorial_gate,
                "4 cases + Bayesian posterior + 15 attention gates + LossSchedule",
            ),
            LayerStatus(
                "L3",
                "Strategic gated modules",
                "forecast_net",
                self.cfg.use_combinatorial_gate,
                f"{'all 15 gates' if r else 'not routed yet'} -> expert modules "
                "(Fibonacci/Fourier/Wavelet/GARCH/ChangePoint/Extreme/MC)",
            ),
            LayerStatus(
                "L4",
                "Hybrid neural core",
                "forecast_net.ForecastNet",
                True,
                "RoPE causal transformer over 12 curated series + tabular MLP",
            ),
            LayerStatus(
                "L5",
                "Macro-Micro fusion",
                "macro_micro_fusion + nav_premium",
                self.cfg.use_macro_fusion,
                "NAV-premium 8-vec cross-attention (off by default -- needs NAV CSV)",
            ),
            LayerStatus(
                "L6",
                "Combinatorial + financial loss",
                "forecast_net.forecast_loss",
                True,
                (
                    "weights from L2 LossSchedule"
                    if self.cfg.use_loss_schedule
                    else "fixed weights (use_loss_schedule=False)"
                ),
            ),
            LayerStatus(
                "L7",
                "Inference & decision",
                "inference + inference_decision_layer",
                True,
                "Student-t dist + direction + fwd-vol + leverage_factorization",
            ),
            LayerStatus(
                "L8",
                "Evaluation & backtest",
                "walk_forward + evaluation + baselines",
                True,
                "expanding walk-forward, embargo, PIT calibration, DM test, costed strategy",
            ),
        ]

    def print_conformance(self) -> None:
        print("=" * 78)
        print("CARN-X  ·  8-LAYER CONFORMANCE")
        print("=" * 78)
        for s in self.conformance():
            mark = "✓" if s.wired else "○"
            print(f"  {mark} {s.layer}  {s.name:34} [{s.module}]")
            print(f"       {s.note}")
        print("=" * 78)


if __name__ == "__main__":
    cx = CARNX(default())
    cx.route()
    cx.print_conformance()
    print("\nrouting summary:")
    r = cx._routing
    if r is not None:
        print(f"  dominant case      : {r.dominant_case}")
        print(
            f"  loss weights       : task={r.w_task:.2f} comb={r.w_combinatorial:.2f} "
            f"struct={r.w_structure:.2f} extreme={r.w_extreme:.2f} lev={r.w_leverage:.2f}"
        )
        print(
            "  top gates          : "
            + ", ".join(
                f"{k}={v:.2f}" for k, v in sorted(r.gate_dict().items(), key=lambda kv: -kv[1])[:5]
            )
        )
