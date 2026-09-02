"""
CARN-X  --  Baselines
=====================
Honest yardsticks the neural model must beat, evaluated on the *same*
expanding walk-forward splits, targets and metrics as walk_forward.py.

    rw       : random walk           -> mu = 0
    drift    : trailing-mean return   -> mu = mean(ret, 252) * h
    ewma     : EWMA momentum          -> mu = ewma(ret) * h
    ar1      : AR(1) on daily returns, iterated h steps
    garch    : GARCH(1,1) vol forecast (vol-only; direction = drift)
    ridge    : Ridge regression on the full tab-feature vector -> y_scaled
    gbm      : HistGradientBoosting  on the full tab-feature vector -> y_scaled
    logit    : LogisticRegression on tab features -> P(up)   (direction head)

``run_baselines`` returns an OOS DataFrame per model, shaped exactly like the
neural model's so evaluation.evaluate / print_report work unchanged.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dataset import AssembledData, assemble
from forecast_net import DEFAULT_SEQ_FEATURES
from targets import descale

warnings.filterwarnings("ignore")


@dataclass
class BaselineConfig:
    initial_train: int = 504
    step: int = 21
    embargo: int = 20
    models: tuple = ("rw", "drift", "ewma", "ar1", "garch", "ridge", "gbm", "logit")
    ridge_alpha: float = 5.0
    gbm_max_iter: int = 300
    gbm_learning_rate: float = 0.03
    gbm_max_depth: int = 3


def _oos_row(
    date,
    pos,
    h,
    mu_scaled,
    sigma_scaled,
    nu,
    p_up,
    sigma_ewma,
    y_scaled,
    y_raw,
    y_sign,
    y_fwdvol,
    fwd_vol_pred,
    drift=0.0,
):
    mu_raw = float(descale(mu_scaled, sigma_ewma, h, drift))
    sig_raw = float(descale(sigma_scaled, sigma_ewma, h))
    return dict(
        date=date,
        pos=int(pos),
        horizon=int(h),
        mu_scaled=mu_scaled,
        sigma_scaled=sigma_scaled,
        nu=nu,
        p_up=p_up,
        fwd_vol_pred=fwd_vol_pred,
        mu_raw=mu_raw,
        sigma_raw=sig_raw,
        sigma_ewma=float(sigma_ewma),
        y_scaled=y_scaled,
        y_raw=y_raw,
        y_sign=y_sign,
        y_fwdvol=y_fwdvol,
    )


def run_baselines(
    cfg: BaselineConfig | None = None, data: AssembledData | None = None
) -> dict[str, pd.DataFrame]:
    cfg = cfg or BaselineConfig()
    if data is None:
        data = assemble(seq_feature_names=DEFAULT_SEQ_FEATURES)

    idx = data.index
    n = len(idx)
    horizons = list(data.tb.horizons)
    rets = np.log(data.close.reindex(idx)).diff().to_numpy()
    X = data.X.to_numpy(np.float64)
    y_scaled = data.tb.y_scaled.loc[idx].to_numpy()
    y_raw = data.tb.y_raw.loc[idx].to_numpy()
    y_sign = data.tb.y_sign.loc[idx].to_numpy()
    y_fwdvol = data.tb.y_fwdvol.loc[idx].to_numpy()
    y_drift = data.tb.y_drift.loc[idx].to_numpy()
    sigma_ewma = data.tb.sigma_ewma.loc[idx].to_numpy()

    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    try:
        from arch import arch_model

        HAS_ARCH = True
    except ImportError:
        HAS_ARCH = False

    out: dict[str, list[dict]] = {m: [] for m in cfg.models}

    starts = list(range(cfg.initial_train, n - cfg.embargo - 1, cfg.step))
    for t_tr in starts:
        te = np.arange(t_tr + cfg.embargo, min(t_tr + cfg.embargo + cfg.step, n))
        if len(te) == 0:
            continue

        r_tr = rets[1:t_tr]
        drift_daily = np.nanmean(r_tr)
        ewma_daily = pd.Series(rets[:t_tr]).ewm(span=20, adjust=False).mean().iloc[-1]
        # AR(1)
        r0, r1 = r_tr[:-1], r_tr[1:]
        ok = np.isfinite(r0) & np.isfinite(r1)
        phi = np.polyfit(r0[ok], r1[ok], 1)[0] if ok.sum() > 30 else 0.0
        phi = float(np.clip(phi, -0.5, 0.5))

        garch_sig_daily = None
        if HAS_ARCH and "garch" in cfg.models:
            try:
                am = arch_model(r_tr[np.isfinite(r_tr)] * 100, vol="GARCH", p=1, q=1, rescale=False)
                gres = am.fit(disp="off")
                garch_sig_daily = float(
                    np.sqrt(gres.forecast(horizon=1).variance.values[-1, 0]) / 100
                )
            except Exception:
                garch_sig_daily = None

        # supervised models: fit per horizon on y_scaled
        sup_models: dict[str, list] = {"ridge": [], "gbm": [], "logit": []}
        Xtr_raw = X[:t_tr]
        scaler = StandardScaler().fit(np.nan_to_num(Xtr_raw))
        Xtr = scaler.transform(np.nan_to_num(Xtr_raw))
        Xte = scaler.transform(np.nan_to_num(X[te]))
        for hi in range(len(horizons)):
            ytr = y_scaled[:t_tr, hi]
            m = np.isfinite(ytr)
            if "ridge" in cfg.models:
                rg = Ridge(alpha=cfg.ridge_alpha).fit(Xtr[m], ytr[m])
                sup_models["ridge"].append(rg.predict(Xte))
            if "gbm" in cfg.models:
                gb = HistGradientBoostingRegressor(
                    max_iter=cfg.gbm_max_iter,
                    learning_rate=cfg.gbm_learning_rate,
                    max_depth=cfg.gbm_max_depth,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=0,
                ).fit(Xtr[m], ytr[m])
                sup_models["gbm"].append(gb.predict(Xte))
            if "logit" in cfg.models:
                str_tr = y_sign[:t_tr, hi]
                ml = np.isfinite(str_tr)
                lg = LogisticRegression(C=0.3, max_iter=500).fit(Xtr[ml], str_tr[ml].astype(int))
                sup_models["logit"].append(lg.predict_proba(Xte)[:, 1])

        for j, pos in enumerate(te):
            for hi, h in enumerate(horizons):
                base = dict(
                    date=idx[pos],
                    pos=pos,
                    h=h,
                    sigma_ewma=sigma_ewma[pos],
                    y_scaled=y_scaled[pos, hi],
                    y_raw=y_raw[pos, hi],
                    y_sign=y_sign[pos, hi],
                    y_fwdvol=y_fwdvol[pos, hi],
                )
                sig_s = 1.0  # vol-normalized target has unit scale by construction
                nu = 6.0
                sd = sigma_ewma[pos] * np.sqrt(h)

                drift_val = float(y_drift[pos, hi])

                def emit(name, mu_raw, p_up, fwd_vol_pred, sig_scaled=sig_s, drift=0.0):
                    # mu_raw is a RAW-space forecast; store it losslessly and let
                    # _oos_row re-derive it (drift is only for de-trended models)
                    out[name].append(
                        _oos_row(
                            base["date"],
                            base["pos"],
                            base["h"],
                            mu_scaled=(mu_raw - drift) / (sd + 1e-12),
                            sigma_scaled=sig_scaled,
                            nu=nu,
                            p_up=p_up,
                            sigma_ewma=base["sigma_ewma"],
                            y_scaled=base["y_scaled"],
                            y_raw=base["y_raw"],
                            y_sign=base["y_sign"],
                            y_fwdvol=base["y_fwdvol"],
                            fwd_vol_pred=fwd_vol_pred,
                            drift=drift,
                        )
                    )

                if "rw" in cfg.models:
                    emit("rw", 0.0, 0.5, sigma_ewma[pos])
                if "drift" in cfg.models:
                    mu = drift_daily * h
                    emit("drift", mu, float(1 / (1 + np.exp(-mu / (sd + 1e-9)))), sigma_ewma[pos])
                if "ewma" in cfg.models:
                    mu = float(ewma_daily) * h
                    emit("ewma", mu, float(1 / (1 + np.exp(-mu / (sd + 1e-9)))), sigma_ewma[pos])
                if "ar1" in cfg.models:
                    last_r = rets[pos] if np.isfinite(rets[pos]) else 0.0
                    mu = sum(phi**k * last_r for k in range(1, h + 1))
                    emit("ar1", mu, float(1 / (1 + np.exp(-mu / (sd + 1e-9)))), sigma_ewma[pos])
                if "garch" in cfg.models:
                    gv = garch_sig_daily or sigma_ewma[pos]
                    emit(
                        "garch",
                        drift_daily * h,
                        float(1 / (1 + np.exp(-(drift_daily * h) / (gv * np.sqrt(h) + 1e-9)))),
                        gv,
                    )
                if "ridge" in cfg.models:
                    ms = float(sup_models["ridge"][hi][j])
                    emit(
                        "ridge",
                        descale(ms, sigma_ewma[pos], h, drift_val),
                        float(1 / (1 + np.exp(-ms))),
                        sigma_ewma[pos],
                        drift=drift_val,
                    )
                if "gbm" in cfg.models:
                    ms = float(sup_models["gbm"][hi][j])
                    emit(
                        "gbm",
                        descale(ms, sigma_ewma[pos], h, drift_val),
                        float(1 / (1 + np.exp(-ms))),
                        sigma_ewma[pos],
                        drift=drift_val,
                    )
                if "logit" in cfg.models:
                    p = float(sup_models["logit"][hi][j])
                    mu = descale((p - 0.5) * 2.0, sigma_ewma[pos], h, drift_val)
                    emit("logit", mu, p, sigma_ewma[pos], drift=drift_val)

    return {
        m: pd.DataFrame(rows).sort_values(["horizon", "date"]) for m, rows in out.items() if rows
    }


if __name__ == "__main__":
    from evaluation import evaluate

    print("=" * 74)
    print("CARN-X Baselines -- walk-forward OOS")
    print("=" * 74)
    frames = run_baselines()

    summary = []
    for name, df in frames.items():
        r = evaluate(df)
        for h, block in r["horizons"].items():
            s = block["statistical"]
            row = {
                "model": name,
                "h": h,
                "dir_acc": s["dir_acc_mean"],
                "rmse_skill": s["rmse_skill"],
                "nll": s["nll"],
                "pit_ks": s["pit_ks"],
            }
            if "economic" in block:
                row["sharpe"] = block["economic"]["strategy"].get("sharpe", float("nan"))
            summary.append(row)
    sm = pd.DataFrame(summary)
    print("\n", sm.pivot_table(index="model", columns="h", values="dir_acc").round(3), sep="")
    print("\ndirectional accuracy by model x horizon  (above)\n")
    print(
        sm[sm.h == sm.h.min()][["model", "dir_acc", "rmse_skill", "nll", "pit_ks", "sharpe"]]
        .round(3)
        .to_string(index=False)
    )
    print("\n" + "=" * 74)
