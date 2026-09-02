"""
CARN-X  --  Walk-Forward Training & Out-of-Sample Prediction
===========================================================
The real evaluation harness. Expanding-window walk-forward:

    |<---------------- train ---------------->|  embargo  |<- test ->|
    0                                      t_tr       t_tr+E     t_tr+E+S

- The model is (re)trained from scratch on an ensemble of seeds every
  ``retrain_every`` folds; between retrains the last ensemble is reused.
- ``embargo`` (= max horizon) rows are skipped between train and test so a
  training label cannot overlap the test window.
- Fold scaling statistics come from the training rows only.
- Every test row gets a genuine out-of-sample predictive distribution.

Output: a tidy DataFrame of OOS predictions (one row per test day per horizon)
plus the assembled fold log, cached to ``runs/``.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import json
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import AssembledData, FoldScaler, WindowedDataset, assemble
from forecast_net import (
    DEFAULT_SEQ_FEATURES,
    ExpertStrengths,
    ForecastNet,
    ModelConfig,
    forecast_loss,
)
from targets import TargetConfig, descale

warnings.filterwarnings("ignore")


@dataclass
class WFConfig:
    window: int = 64
    initial_train: int = 504  # ~2y before the first prediction
    step: int = 21  # test-block length (≈1 trading month)
    embargo: int = 20  # = max horizon
    retrain_every: int = 2  # retrain the ensemble every k folds
    max_folds: int | None = None  # cap folds (smoke tests)

    epochs: int = 60
    patience: int = 8
    batch_size: int = 64
    lr: float = 1.2e-3
    weight_decay: float = 1.5e-4
    grad_clip: float = 1.0
    val_frac: float = 0.15  # tail of train used for early stopping
    ensemble: int = 3
    seeds: tuple[int, ...] = (0, 1, 2)

    use_experts: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_threads: int = 0  # 0 -> leave torch default
    verbose: bool = True
    run_dir: str = "./runs"
    tag: str = "wf"

    model: ModelConfig = field(default_factory=ModelConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)


# ---------------------------------------------------------------------------
# routing (combinatorial gate -> expert strengths), one call per fold
# ---------------------------------------------------------------------------


def _route_experts(close_window: np.ndarray, use_experts: bool) -> ExpertStrengths:
    if not use_experts:
        return ExpertStrengths()
    try:
        from diagnosis_layer import DiagnosisLayer
        from combinatorial_gate import CombinatorialGate

        diag = DiagnosisLayer().diagnose(close_window[-min(400, len(close_window)) :])
        routing = CombinatorialGate(temperature=0.5).route(diag).routing
        return ExpertStrengths.from_routing(routing)
    except Exception as e:  # noqa: BLE001
        print(f"[walk_forward] routing failed ({e}); experts off this fold")
        return ExpertStrengths()


# ---------------------------------------------------------------------------
# single-model training
# ---------------------------------------------------------------------------


def _train_one(
    cfg: WFConfig,
    data: AssembledData,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    scaler: FoldScaler,
    experts: ExpertStrengths,
    seed: int,
) -> ForecastNet:
    torch.manual_seed(seed)
    np.random.seed(seed)

    mcfg = ModelConfig(
        **{
            **asdict(cfg.model),
            "use_experts": cfg.use_experts,
            "window": cfg.window,
            "horizons": data.tb.horizons,
            "seq_feature_names": tuple(data.seq_cols),
        }
    )
    net = ForecastNet(mcfg, n_tab_features=data.X.shape[1]).to(cfg.device)

    train_ds = WindowedDataset(data, train_pos, cfg.window, scaler)
    val_ds = WindowedDataset(data, val_pos, cfg.window, scaler)
    if len(train_ds) < cfg.batch_size:
        raise RuntimeError(f"fold too small: {len(train_ds)} train samples")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False) if len(val_ds) else None

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val, best_state, bad = float("inf"), None, 0
    for _epoch in range(cfg.epochs):
        net.train()
        for batch in train_dl:
            opt.zero_grad()
            out = net(batch["x_seq"].to(cfg.device), batch["x_tab"].to(cfg.device), experts)
            losses = forecast_loss(
                out,
                batch["y_scaled"].to(cfg.device),
                batch["y_sign"].to(cfg.device),
                batch["y_fwdvol_lr"].to(cfg.device),
                mcfg,
                batch["sample_w"].to(cfg.device),
                routing=experts,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
        sched.step()

        if val_dl is not None:
            net.eval()
            vtot, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_dl:
                    out = net(batch["x_seq"].to(cfg.device), batch["x_tab"].to(cfg.device), experts)
                    vl = forecast_loss(
                        out,
                        batch["y_scaled"].to(cfg.device),
                        batch["y_sign"].to(cfg.device),
                        batch["y_fwdvol_lr"].to(cfg.device),
                        mcfg,
                        routing=experts,
                    )["total"].item()
                    vtot += vl * len(batch["pos"])
                    vn += len(batch["pos"])
            vloss = vtot / max(vn, 1)
            if vloss < best_val - 1e-4:
                best_val, best_state, bad = (
                    vloss,
                    {k: v.detach().clone() for k, v in net.state_dict().items()},
                    0,
                )
            else:
                bad += 1
                if bad >= cfg.patience:
                    break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# ensemble prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def _predict(
    nets: list[ForecastNet],
    data: AssembledData,
    test_pos: np.ndarray,
    window: int,
    scaler: FoldScaler,
    experts: ExpertStrengths,
    device: str,
) -> pd.DataFrame:
    ds = WindowedDataset(data, test_pos, window, scaler)
    if len(ds) == 0:
        return pd.DataFrame()
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    horizons = list(data.tb.horizons)

    rows: list[dict] = []
    for batch in dl:
        xs, xt = batch["x_seq"].to(device), batch["x_tab"].to(device)
        mus, sigmas, nus, pups, fvols = [], [], [], [], []
        for net in nets:
            o = net(xs, xt, experts)
            mus.append(o["mu"])
            sigmas.append(o["sigma"])
            nus.append(o["nu"])
            pups.append(torch.sigmoid(o["dir_logit"]))
            fvols.append(o["fwd_vol_lr"])
        mu = torch.stack(mus).mean(0)
        # mixture variance = mean(sigma^2) + var(mu)
        sig = torch.sqrt(
            torch.stack([s**2 for s in sigmas]).mean(0)
            + torch.stack(mus).var(0, unbiased=False)
            + 1e-8
        )
        nu = torch.stack(nus).mean(0)
        pup = torch.stack(pups).mean(0)
        fvol = torch.stack(fvols).mean(0)

        pos = batch["pos"].numpy()
        dates = data.index[pos]
        sigma_ewma = batch["sigma"].numpy()
        drift = batch["y_drift"].numpy()
        for b in range(len(pos)):
            for hi, h in enumerate(horizons):
                mu_raw = float(descale(mu[b, hi].item(), sigma_ewma[b], h, drift[b, hi]))
                sig_raw = float(descale(sig[b, hi].item(), sigma_ewma[b], h))  # scale, no drift
                rows.append(
                    {
                        "date": dates[b],
                        "pos": int(pos[b]),
                        "horizon": h,
                        "mu_scaled": mu[b, hi].item(),
                        "sigma_scaled": sig[b, hi].item(),
                        "nu": nu[b, hi].item(),
                        "p_up": pup[b, hi].item(),
                        "fwd_vol_pred": float(sigma_ewma[b] * np.exp(fvol[b, hi].item())),
                        "fwd_vol_logratio": fvol[b, hi].item(),
                        "mu_raw": mu_raw,
                        "sigma_raw": sig_raw,
                        "sigma_ewma": float(sigma_ewma[b]),
                        "y_scaled": batch["y_scaled"][b, hi].item(),
                        "y_raw": batch["y_raw"][b, hi].item(),
                        "y_sign": batch["y_sign"][b, hi].item(),
                        "y_fwdvol": batch["y_fwdvol"][b, hi].item(),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


@dataclass
class WFResult:
    oos: pd.DataFrame
    fold_log: pd.DataFrame
    config: dict
    run_path: str


def run_walk_forward(cfg: WFConfig | None = None, data: AssembledData | None = None) -> WFResult:
    cfg = cfg or WFConfig()
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    if data is None:
        data = assemble(seq_feature_names=cfg.model.seq_feature_names, target_cfg=cfg.targets)

    n = len(data.index)
    starts = list(range(cfg.initial_train, n - cfg.embargo - 1, cfg.step))
    if cfg.max_folds:
        starts = starts[: cfg.max_folds]
    if not starts:
        raise RuntimeError(
            f"no folds: n={n}, need > initial_train+embargo+step={cfg.initial_train + cfg.embargo + cfg.step}"
        )

    close_arr = data.close.reindex(data.index).to_numpy(float)
    all_oos: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    nets: list[ForecastNet] = []
    scaler: FoldScaler | None = None
    experts = ExpertStrengths()
    t0 = time.time()

    for fi, t_tr in enumerate(starts):
        test_lo = t_tr + cfg.embargo
        test_hi = min(test_lo + cfg.step, n)
        test_pos = np.arange(test_lo, test_hi)

        retrain = (fi % cfg.retrain_every == 0) or not nets
        if retrain:
            all_pos = np.arange(0, t_tr)
            n_val = max(cfg.window + cfg.batch_size, int(len(all_pos) * cfg.val_frac))
            train_pos, val_pos = all_pos[:-n_val], all_pos[-n_val:]

            scaler = FoldScaler().fit(data.X.iloc[:t_tr], data.seq_cols)
            experts = _route_experts(close_arr[:t_tr], cfg.use_experts)

            nets = []
            for s in cfg.seeds[: cfg.ensemble]:
                net = _train_one(cfg, data, train_pos, val_pos, scaler, experts, seed=s)
                nets.append(net)

        pred = _predict(nets, data, test_pos, cfg.window, scaler, experts, cfg.device)
        if len(pred):
            all_oos.append(pred)

        h1 = pred[pred.horizon == data.tb.horizons[0]] if len(pred) else pred
        da = float(((h1.mu_raw > 0) == (h1.y_raw > 0)).mean()) if len(h1) else float("nan")
        fold_rows.append(
            {
                "fold": fi,
                "train_end": str(data.index[t_tr - 1].date()),
                "test_start": str(data.index[test_lo].date()) if test_lo < n else "-",
                "test_end": str(data.index[min(test_hi, n) - 1].date()),
                "n_test": len(test_pos),
                "retrained": retrain,
                "experts": (experts.fourier, experts.recurrent, experts.extreme),
                "h1_dir_acc": da,
            }
        )
        if cfg.verbose:
            print(
                f"fold {fi:02d} | train->{fold_rows[-1]['train_end']} "
                f"test {fold_rows[-1]['test_start']}..{fold_rows[-1]['test_end']} "
                f"| n={len(test_pos)} retrain={retrain} h1_dir={da:.3f} "
                f"| {time.time() - t0:.0f}s"
            )

    oos = (
        pd.concat(all_oos, ignore_index=True).sort_values(["horizon", "date"])
        if all_oos
        else pd.DataFrame()
    )
    fold_log = pd.DataFrame(fold_rows)

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_path = run_dir / f"{cfg.tag}_{stamp}"
    run_path.mkdir()
    oos.to_parquet(run_path / "oos_predictions.parquet")
    fold_log.to_csv(run_path / "fold_log.csv", index=False)
    cfg_dump = {k: (v if _jsonable(v) else str(v)) for k, v in asdict(cfg).items()}
    (run_path / "config.json").write_text(json.dumps(cfg_dump, indent=2, default=str))

    print(
        f"\n[walk_forward] {len(starts)} folds, {len(oos)} OOS rows, "
        f"{time.time() - t0:.0f}s -> {run_path}"
    )
    return WFResult(oos=oos, fold_log=fold_log, config=cfg_dump, run_path=str(run_path))


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    import sys

    quick = "--full" not in sys.argv
    print("=" * 70)
    print(f"CARN-X Walk-Forward -- {'QUICK smoke' if quick else 'FULL run'}")
    print("=" * 70)

    if quick:
        cfg = WFConfig(
            initial_train=500,
            step=90,
            max_folds=3,
            epochs=12,
            patience=4,
            ensemble=1,
            seeds=(0,),
            retrain_every=1,
            tag="wf_smoke",
        )
    else:
        cfg = WFConfig(tag="wf_full")

    res = run_walk_forward(cfg)
    if len(res.oos):
        for h in sorted(res.oos.horizon.unique()):
            d = res.oos[res.oos.horizon == h]
            dir_acc = ((d.mu_raw > 0) == (d.y_raw > 0)).mean()
            rmse = np.sqrt(((d.mu_raw - d.y_raw) ** 2).mean())
            base = np.sqrt((d.y_raw**2).mean())
            print(
                f"  h={h:2d} | n={len(d):4d} | dir_acc={dir_acc:.3f} | "
                f"RMSE(ret)={rmse:.4f} vs zero-pred {base:.4f}"
            )
    print("\n" + "=" * 70)
    print("Walk-Forward ready.")
    print("=" * 70)
