"""
CARN-X  --  Production Inference
===============================
The walk-forward harness (walk_forward.py) is for *evaluation*. This module is
the *use* path: train once on all history available today, then produce a
calibrated predictive distribution for the latest bar, plus a Monte-Carlo price
cone derived from that distribution.

    fit_production_model(data, cfg)      -> ProductionModel   (ensemble + scaler + experts)
    predict_distribution(pm, data, pos)  -> DataFrame  (one row per horizon)
    monte_carlo_paths(pm, data, ...)     -> MonteCarloResult
    latest_forecast(cfg=None)            -> everything the app needs, cached-friendly

Nothing here re-implements the model; it composes walk_forward._train_one and
walk_forward._predict so the training recipe stays identical to the evaluation.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from dataset import AssembledData, FoldScaler, assemble
from forecast_net import DEFAULT_SEQ_FEATURES, ExpertStrengths, ForecastNet, ModelConfig
from targets import TargetConfig, descale
from walk_forward import WFConfig, _predict, _route_experts, _train_one

import os as _os

_BUILTIN_MODEL_PATH = "./carnx_checkpoints/production_model.pt"


def default_model_path() -> str:
    """Resolved live each call so ``CARNX_MODEL_PATH`` (set by tests / experiments
    to avoid clobbering the real model) always takes effect."""
    return _os.environ.get("CARNX_MODEL_PATH") or _BUILTIN_MODEL_PATH


DEFAULT_MODEL_PATH = default_model_path()  # back-compat snapshot for imports


@dataclass
class InferenceConfig:
    window: int = 64
    val_tail: int = 120  # last N rows held out for early stopping
    ensemble: int = 5
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 60
    patience: int = 8
    lr: float = 1.2e-3
    weight_decay: float = 1.5e-4
    batch_size: int = 64
    use_experts: bool = True
    num_threads: int = 4
    model: ModelConfig = field(default_factory=ModelConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)

    def to_wfconfig(self) -> WFConfig:
        return WFConfig(
            window=self.window,
            epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            ensemble=self.ensemble,
            seeds=self.seeds,
            use_experts=self.use_experts,
            num_threads=self.num_threads,
            model=self.model,
            targets=self.targets,
            val_frac=0.15,
        )


@dataclass
class ProductionModel:
    nets: list
    scaler: FoldScaler
    experts: ExpertStrengths
    wf_cfg: WFConfig
    trained_at: str
    train_rows: int
    data_date_max: str


@dataclass
class MonteCarloResult:
    horizon: int
    n_paths: int
    last_price: float
    paths: np.ndarray  # [n_paths, horizon+1]  price paths incl. t0
    percentiles: dict[str, np.ndarray]  # label -> [horizon+1] price band
    terminal_prices: np.ndarray  # [n_paths]
    prob_up: float
    expected_price: float
    var_5_price: float  # 5th percentile terminal price
    es_5_price: float  # expected shortfall below the 5th pct
    # --- extended (all optional so older callers keep working) ---------------
    drift_mode: str = "model"
    mu_annual: float = 0.0  # annualised log-drift actually used
    sigma_annual: float = 0.0  # annualised vol actually used
    nu: float = 0.0  # Student-t dof of the daily innovations
    var_1_price: float = 0.0
    es_1_price: float = 0.0
    median_price: float = 0.0
    terminal_skew: float = 0.0
    terminal_kurt: float = 0.0  # excess kurtosis
    p_touch_up: float = 0.0  # P(path ever >= target)   (target > S0)
    p_touch_down: float = 0.0  # P(path ever <= target)   (target < S0)
    p_close_above: float = 0.0  # P(S_T >= target)
    expected_hit_day: float = 0.0  # E[first-passage day | hit], trading days
    max_drawdown_p50: float = 0.0  # median of per-path max drawdown
    target_price: float = 0.0


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def fit_production_model(
    data: AssembledData,
    cfg: InferenceConfig | None = None,
    progress: callable | None = None,
) -> ProductionModel:
    cfg = cfg or InferenceConfig()
    if cfg.num_threads > 0:
        torch.set_num_threads(cfg.num_threads)
    wf = cfg.to_wfconfig()

    # only rows whose every-horizon target is defined can train / early-stop
    trainable = data.trainable_positions()
    if len(trainable) < cfg.window + 4 * cfg.batch_size:
        raise RuntimeError(f"not enough trainable rows: {len(trainable)}")
    n_val = max(cfg.window + cfg.batch_size, cfg.val_tail)
    train_pos, val_pos = trainable[:-n_val], trainable[-n_val:]

    # scaler statistics from the training rows only (never the val tail or the
    # target-less bars we will predict on)
    train_cutoff = int(train_pos[-1]) + 1
    scaler = FoldScaler().fit(data.X.iloc[:train_cutoff], data.seq_cols)
    close_arr = data.close.reindex(data.index).to_numpy(float)
    experts = _route_experts(close_arr, cfg.use_experts)

    nets = []
    for i, seed in enumerate(cfg.seeds[: cfg.ensemble]):
        if progress:
            progress(i, cfg.ensemble)
        nets.append(_train_one(wf, data, train_pos, val_pos, scaler, experts, seed=seed))
    if progress:
        progress(cfg.ensemble, cfg.ensemble)

    return ProductionModel(
        nets=nets,
        scaler=scaler,
        experts=experts,
        wf_cfg=wf,
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        train_rows=len(train_pos),
        data_date_max=str(data.index.max().date()),
    )


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def save_production_model(pm: ProductionModel, n_tab_features: int, path: str | None = None) -> str:
    path = path or default_model_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    from forecast_net import NET_ARCH_VERSION

    torch.save(
        {
            "net_states": [n.state_dict() for n in pm.nets],
            "n_tab_features": n_tab_features,
            "net_arch_version": NET_ARCH_VERSION,
            "seq_cols": list(pm.nets[0].cfg.seq_feature_names) if pm.nets else None,
            "scaler": pickle.dumps(pm.scaler),
            "experts": pickle.dumps(pm.experts),
            "wf_cfg": pickle.dumps(pm.wf_cfg),
            "meta": {
                "trained_at": pm.trained_at,
                "train_rows": pm.train_rows,
                "data_date_max": pm.data_date_max,
            },
        },
        path,
    )
    return path


class ModelStale(RuntimeError):
    """The saved model does not match the current feature set / data schema."""


def load_production_model(data: AssembledData, path: str | None = None) -> ProductionModel:
    path = path or default_model_path()
    blob = torch.load(path, map_location="cpu", weights_only=False)

    from forecast_net import NET_ARCH_VERSION

    if int(blob.get("net_arch_version", 1)) != NET_ARCH_VERSION:
        raise ModelStale(
            f"checkpoint network architecture is v{blob.get('net_arch_version', 1)}, "
            f"code is v{NET_ARCH_VERSION}. Retrain the model (Settings → אמן מודל עכשיו)."
        )
    n_saved = int(blob.get("n_tab_features", -1))
    n_now = int(data.X.shape[1])
    if n_saved != n_now:
        raise ModelStale(
            f"model was trained on {n_saved} features, the pipeline now produces "
            f"{n_now}. Retrain the model (Settings → אמן מודל עכשיו)."
        )
    saved_seq = blob.get("seq_cols")
    if saved_seq is not None and list(saved_seq) != list(data.seq_cols):
        raise ModelStale("the sequence-feature set changed since the model was trained; retrain.")
    wf_cfg: WFConfig = pickle.loads(blob["wf_cfg"])
    scaler = pickle.loads(blob["scaler"])
    experts = pickle.loads(blob["experts"])

    from dataclasses import asdict

    mcfg = ModelConfig(
        **{
            **asdict(wf_cfg.model),
            "use_experts": wf_cfg.use_experts,
            "window": wf_cfg.window,
            "horizons": data.tb.horizons,
            "seq_feature_names": tuple(data.seq_cols),
        }
    )
    nets = []
    for state in blob["net_states"]:
        net = ForecastNet(mcfg, n_tab_features=blob["n_tab_features"])
        net.load_state_dict(state)
        net.eval()
        nets.append(net)

    m = blob["meta"]
    return ProductionModel(
        nets=nets,
        scaler=scaler,
        experts=experts,
        wf_cfg=wf_cfg,
        trained_at=m["trained_at"],
        train_rows=m["train_rows"],
        data_date_max=m["data_date_max"],
    )


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def predict_distribution(
    pm: ProductionModel, data: AssembledData, positions: np.ndarray | None = None
) -> pd.DataFrame:
    if positions is None:
        positions = np.array([len(data.index) - 1])
    df = _predict(
        pm.nets,
        data,
        np.asarray(positions),
        pm.wf_cfg.window,
        pm.scaler,
        pm.experts,
        pm.wf_cfg.device,
    )
    # add human-friendly columns
    if len(df):
        df["expected_move_pct"] = np.expm1(df["mu_raw"]) * 100.0
        df["ann_vol_implied"] = df["sigma_ewma"] * np.sqrt(252.0) * 100.0
        for q, name in [(0.05, "q05"), (0.25, "q25"), (0.75, "q75"), (0.95, "q95")]:
            from scipy.stats import t as _t

            z = _t.ppf(q, df["nu"].to_numpy())
            df[f"ret_{name}_pct"] = (
                np.expm1(df["mu_raw"].to_numpy() + z * df["sigma_raw"].to_numpy()) * 100.0
            )
    return df


def tail_probability(row: pd.Series, threshold_pct: float) -> float:
    """P(forward return <= threshold_pct) under the predicted Student-t."""
    from scipy.stats import t as _t

    thr_log = np.log1p(threshold_pct / 100.0)
    z = (thr_log - row["mu_raw"]) / row["sigma_raw"]
    return float(_t.cdf(z, df=row["nu"]))


# ---------------------------------------------------------------------------
# Monte-Carlo cone
# ---------------------------------------------------------------------------

_TRADING_DAYS = 252.0
DRIFT_MODES = ("model", "risk_neutral", "custom", "historical")


def _resolve_daily_drift(
    drift_mode: str,
    *,
    sigma_daily: float,
    custom_drift_annual: float,
    risk_free_rate: float,
    hist_daily_logret: float,
) -> tuple[float | None, float]:
    """Return (fixed_daily_log_drift | None, mu_annual_reported).

    ``None`` means "use the model's own per-step location" (mode ``model``).
    For the closed-form modes the Ito term -0.5 sigma^2 is applied so that
    ``mu_annual`` is the *arithmetic* expected annual log-growth the user dialled
    in, and E[S_T] = S_0 * exp(mu_annual * T) holds for the Gaussian limit."""
    if drift_mode == "model":
        return None, float("nan")
    if drift_mode == "risk_neutral":
        mu_a = float(risk_free_rate)
    elif drift_mode == "custom":
        mu_a = float(custom_drift_annual)
    elif drift_mode == "historical":
        mu_a = float(hist_daily_logret * _TRADING_DAYS)
    else:
        raise ValueError(f"unknown drift_mode {drift_mode!r}; pick one of {DRIFT_MODES}")
    # the -0.5 sigma^2 convexity term is applied later as an *empirical* per-step
    # correction (exact for fat tails), so here we only return the raw target.
    return mu_a / _TRADING_DAYS, mu_a


def monte_carlo_paths(
    pm: ProductionModel,
    data: AssembledData,
    horizon: int = 252,
    n_paths: int = 20000,
    seed: int = 0,
    *,
    drift_mode: str = "model",
    custom_drift_annual: float = 0.0,
    risk_free_rate: float = 0.045,
    target_price: float | None = None,
    antithetic: bool = True,
    hist_lookback: int = 504,
) -> MonteCarloResult:
    """Daily Student-t Monte-Carlo price cone.

    Volatility and fat-tails (nu) always come from the model's 1-day predictive
    distribution.  The *drift* is user-selectable (``drift_mode``):

    ``model``         the model's own per-step location (its directional view).
    ``risk_neutral``  mu = r  (risk-free rate) -- the martingale benchmark.
    ``custom``        mu = ``custom_drift_annual``  (scenario / bull-case).
    ``historical``    mu = annualised mean daily log-return over ``hist_lookback``
                      (carries the full volatility drag -- shown with a warning).

    Innovations are standardised Student-t: eta ~ t_nu * sqrt((nu-2)/nu) so that
    Var(eta)=1 and ``sigma_daily`` is the true daily std at every step
    (dt = 1/252).  Antithetic variates halve the sampling error for free.
    """
    if drift_mode not in DRIFT_MODES:
        raise ValueError(f"drift_mode must be one of {DRIFT_MODES}, got {drift_mode!r}")
    horizon = max(int(horizon), 1)
    n_paths = max(int(n_paths), 2)
    rng = np.random.default_rng(seed)
    pos = len(data.index) - 1
    row = predict_distribution(pm, data, np.array([pos]))
    h1 = row[row.horizon == data.tb.horizons[0]].iloc[0]

    mu1 = float(h1["mu_scaled"])  # per-step model location, in vol units
    sig1 = float(h1["sigma_scaled"])  # per-step model scale, in vol units
    nu1 = max(float(h1["nu"]), 2.1)  # keep finite variance
    sigma_daily = float(h1["sigma_ewma"])
    drift1 = float(data.tb.y_drift.loc[data.index[pos], f"h{data.tb.horizons[0]}"])

    # S_0 = latest split-adjusted close (yfinance auto_adjust=True upstream)
    last_price = float(data.close.reindex(data.index).iloc[pos])

    close = data.close.reindex(data.index).astype(float)
    hist_logret = np.log(close).diff().tail(int(hist_lookback)).dropna()
    hist_daily = float(hist_logret.mean()) if len(hist_logret) else 0.0

    fixed_daily, mu_annual = _resolve_daily_drift(
        drift_mode,
        sigma_daily=sigma_daily,
        custom_drift_annual=custom_drift_annual,
        risk_free_rate=risk_free_rate,
        hist_daily_logret=hist_daily,
    )

    # --- innovations -------------------------------------------------------
    n_draw = n_paths
    if antithetic:
        n_half = (n_paths + 1) // 2
        raw = rng.standard_t(nu1, size=(n_half, horizon))
        t_draws = np.concatenate([raw, -raw], axis=0)[:n_paths]
        n_draw = t_draws.shape[0]
    else:
        t_draws = rng.standard_t(nu1, size=(n_paths, horizon))
    if fixed_daily is None:  # mode == "model"
        # the model's *calibrated* predictive Student-t: location mu1, scale sig1
        # (both in trailing-vol units) -- left exactly as fitted (PIT-checked).
        step_logret = (mu1 + sig1 * t_draws) * sigma_daily + drift1
        mu_annual = float((mu1 * sigma_daily + drift1) * _TRADING_DAYS)
    else:
        # closed-form drift modes: standardise the innovation so sigma_daily is
        # the true daily std at dt = 1/252 regardless of nu, then apply the
        # empirical convexity correction  c = log E[exp(sigma * eta)]  so that
        # E[exp(step)] = exp(mu_annual / 252) exactly (fat-tail safe martingale).
        eta = t_draws * np.sqrt((nu1 - 2.0) / nu1)  # Var(eta) = 1
        shock = sigma_daily * eta
        from scipy.special import logsumexp as _lse  # overflow-safe E[e^shock]

        c = float(_lse(shock) - np.log(shock.size))
        step_logret = (fixed_daily - c) + shock

    step_logret = np.nan_to_num(step_logret, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    log_paths = np.concatenate(
        [np.zeros((n_draw, 1), np.float32), np.cumsum(step_logret, axis=1)], axis=1
    )
    log_paths = np.clip(log_paths, -80.0, 80.0)  # keep exp() finite in float32
    price_paths = (last_price * np.exp(log_paths)).astype(np.float32)

    pct_labels = {
        "p01": 1,
        "p05": 5,
        "p10": 10,
        "p25": 25,
        "p50": 50,
        "p75": 75,
        "p90": 90,
        "p95": 95,
        "p99": 99,
    }
    percentiles = {n: np.percentile(price_paths, p, axis=0) for n, p in pct_labels.items()}

    terminal = price_paths[:, -1].astype(float)
    var5 = float(np.percentile(terminal, 5))
    var1 = float(np.percentile(terminal, 1))
    es5 = float(terminal[terminal <= var5].mean()) if (terminal <= var5).any() else var5
    es1 = float(terminal[terminal <= var1].mean()) if (terminal <= var1).any() else var1

    from scipy.stats import skew as _skew, kurtosis as _kurt

    running_max = np.maximum.accumulate(price_paths, axis=1)
    max_dd = 1.0 - (price_paths / running_max).min(axis=1)

    tp = float(target_price) if target_price is not None else last_price
    if tp >= last_price:
        hit = price_paths.max(axis=1) >= tp
        p_touch_up = float(hit.mean())
        p_touch_down = 0.0
    else:
        hit = price_paths.min(axis=1) <= tp
        p_touch_down = float(hit.mean())
        p_touch_up = 0.0
    if hit.any():
        cross = (price_paths >= tp) if tp >= last_price else (price_paths <= tp)
        first_day = np.argmax(cross, axis=1).astype(float)
        expected_hit_day = float(first_day[hit].mean())
    else:
        expected_hit_day = float("nan")
    p_close_above = float((terminal >= tp).mean())

    return MonteCarloResult(
        horizon=horizon,
        n_paths=n_draw,
        last_price=last_price,
        paths=price_paths,
        percentiles=percentiles,
        terminal_prices=terminal,
        prob_up=float((terminal > last_price).mean()),
        expected_price=float(terminal.mean()),
        var_5_price=var5,
        es_5_price=es5,
        drift_mode=drift_mode,
        mu_annual=float(mu_annual),
        sigma_annual=float(sigma_daily * np.sqrt(_TRADING_DAYS)),
        nu=nu1,
        var_1_price=var1,
        es_1_price=es1,
        median_price=float(np.median(terminal)),
        terminal_skew=float(_skew(terminal)),
        terminal_kurt=float(_kurt(terminal)),
        p_touch_up=p_touch_up,
        p_touch_down=p_touch_down,
        p_close_above=p_close_above,
        expected_hit_day=expected_hit_day,
        max_drawdown_p50=float(np.median(max_dd)),
        target_price=tp,
    )


# ---------------------------------------------------------------------------
# one-call convenience  (what the app asks for)
# ---------------------------------------------------------------------------


@dataclass
class LatestForecast:
    as_of: str
    last_price: float
    distribution: pd.DataFrame  # per-horizon predictive stats
    model: ProductionModel
    data: AssembledData


def latest_forecast(
    cfg: InferenceConfig | None = None,
    data: AssembledData | None = None,
    progress: callable | None = None,
) -> LatestForecast:
    cfg = cfg or InferenceConfig()
    if data is None:
        data = assemble(
            seq_feature_names=cfg.model.seq_feature_names,
            target_cfg=cfg.targets,
            require_targets=False,  # keep the latest target-less bars for prediction
        )
    pm = fit_production_model(data, cfg, progress=progress)
    dist = predict_distribution(pm, data)
    return LatestForecast(
        as_of=str(data.index.max().date()),
        last_price=float(data.close.reindex(data.index).iloc[-1]),
        distribution=dist,
        model=pm,
        data=data,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Production Inference -- Self Test (quick)")
    print("=" * 70)

    cfg = InferenceConfig(ensemble=2, seeds=(0, 1), epochs=12, patience=4)
    lf = latest_forecast(cfg, progress=lambda i, n: print(f"  trained {i}/{n}"))

    print(f"\nas of {lf.as_of}  last MSTR close = {lf.last_price:.2f}")
    print("\npredictive distribution:")
    cols = ["horizon", "expected_move_pct", "p_up", "ann_vol_implied", "ret_q05_pct", "ret_q95_pct"]
    print(lf.distribution[cols].round(3).to_string(index=False))

    for _, r in lf.distribution.iterrows():
        p_dd10 = tail_probability(r, -10.0)
        print(f"  h={int(r.horizon):2d}: P(return <= -10%) = {p_dd10:.1%}")

    mc = monte_carlo_paths(lf.model, lf.data, horizon=20, n_paths=5000)
    print(
        f"\nMonte-Carlo 20d (model drift): E[price]={mc.expected_price:.2f}  "
        f"P(up)={mc.prob_up:.1%}  VaR5 price={mc.var_5_price:.2f}  ES5={mc.es_5_price:.2f}"
    )

    # martingale property under the risk-neutral measure: E[S_T] ~ S_0 * e^{rT}
    r = 0.045
    mcq = monte_carlo_paths(
        lf.model, lf.data, horizon=252, n_paths=40000, drift_mode="risk_neutral", risk_free_rate=r
    )
    grow = mcq.expected_price / mcq.last_price
    expect = np.exp(r * 252 / 252.0)
    print(
        f"risk-neutral 252d: E[S_T]/S_0 = {grow:.4f}  (target e^r = {expect:.4f})  "
        f"P(up)={mcq.prob_up:.1%}"
    )
    assert abs(grow - expect) < 0.03, f"martingale check failed: {grow:.4f} vs {expect:.4f}"
    print("  martingale property OK")

    print("\n" + "=" * 70)
    print("Production Inference ready.")
    print("=" * 70)
