"""
CARN-X  --  Evaluation
======================
Scores an out-of-sample prediction frame (from walk_forward.run_walk_forward)
on statistical *and* economic criteria, per horizon.

Statistical
    - directional accuracy + binomial p-value vs 0.5
    - RMSE / MAE on raw forward log-returns, and skill vs the zero (random-walk)
      forecast  ( skill = 1 - RMSE_model / RMSE_rw )
    - Student-t NLL and CRPS-proxy of the predictive distribution
    - PIT calibration: histogram uniformity (KS statistic) + 90% interval coverage
    - Diebold-Mariano test of equal predictive accuracy vs the random walk

Economic
    - a transparent walk-forward strategy: position_t = clip(k * p_edge, -1, 1)
      with turnover-based costs; reports CAGR, vol, Sharpe, Sortino, MaxDD,
      Calmar, hit rate, turnover, and the same for buy&hold
    - regime split: performance in the worst-vol quintile vs the calmest

``evaluate`` returns a nested dict; ``print_report`` renders it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# statistical
# ---------------------------------------------------------------------------


def _student_t_nll(y, mu, sigma, nu):
    z = (y - mu) / sigma
    log_c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - np.log(sigma)
    return -(log_c - (nu + 1) / 2 * np.log1p(z**2 / nu))


def _pit(y, mu, sigma, nu):
    return stats.t.cdf((y - mu) / sigma, df=nu)


def _diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int) -> float:
    """DM statistic for squared-error loss; >0 means model 1 is worse. Returns p-value."""
    d = e1**2 - e2**2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return float("nan")
    dbar = d.mean()
    # Newey-West long-run variance with lag h-1
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for k in range(1, h):
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        lrv += 2 * (1 - k / h) * cov
    dm = dbar / np.sqrt(lrv / n + 1e-18)
    return float(2 * (1 - stats.norm.cdf(abs(dm))))


def _stat_block(d: pd.DataFrame, h: int) -> dict[str, float]:
    y = d.y_raw.to_numpy()
    mu = d.mu_raw.to_numpy()
    sig = d.sigma_raw.to_numpy()
    nu = d.nu.to_numpy()
    p_up = d.p_up.to_numpy()

    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sig)
    y, mu, sig, nu, p_up = y[ok], mu[ok], sig[ok], nu[ok], p_up[ok]
    n = len(y)

    # direction from the distribution mean and from the calibrated head
    dir_mu = (mu > 0) == (y > 0)
    dir_head = (p_up > 0.5) == (y > 0)
    n_correct = int(dir_mu.sum())
    binom_p = (
        float(stats.binomtest(n_correct, n, 0.5, alternative="greater").pvalue)
        if n
        else float("nan")
    )

    rmse_m = float(np.sqrt(np.mean((mu - y) ** 2)))
    rmse_rw = float(np.sqrt(np.mean(y**2)))
    mae_m = float(np.mean(np.abs(mu - y)))

    pit = _pit(y, mu, sig, nu)
    ks = float(stats.kstest(pit, "uniform").statistic)
    lo = stats.t.ppf(0.05, nu) * sig + mu
    hi = stats.t.ppf(0.95, nu) * sig + mu
    cover90 = float(np.mean((y >= lo) & (y <= hi)))

    nll = float(np.mean(_student_t_nll(y, mu, sig, nu)))
    dm_p = _diebold_mariano(y - 0.0, y - mu, h)  # rw error vs model error

    return {
        "n": n,
        "dir_acc_mean": float(dir_mu.mean()),
        "dir_acc_head": float(dir_head.mean()),
        "dir_binom_p": binom_p,
        "rmse_model": rmse_m,
        "rmse_rw": rmse_rw,
        "rmse_skill": float(1 - rmse_m / (rmse_rw + 1e-18)),
        "mae_model": mae_m,
        "nll": nll,
        "pit_ks": ks,
        "coverage_90": cover90,
        "dm_pvalue_vs_rw": dm_p,
    }


# ---------------------------------------------------------------------------
# economic
# ---------------------------------------------------------------------------


@dataclass
class EconConfig:
    edge_scale: float = 6.0  # position = clip(edge_scale * (p_up-0.5), -1, 1)
    max_leverage: float = 1.0
    cost_per_turnover: float = 0.0007  # round-trip ~7 bps
    trading_days: int = 252


def _perf(returns: np.ndarray, positions: np.ndarray, cfg: EconConfig) -> dict[str, float]:
    r = returns[np.isfinite(returns)]
    if len(r) < 5:
        return {}
    turnover = np.abs(np.diff(np.concatenate([[0.0], positions]))).sum()
    ann = cfg.trading_days
    mean, sd = r.mean(), r.std()
    downside = r[r < 0].std() if (r < 0).any() else 1e-9
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1).min())
    total = float(eq[-1] - 1)
    cagr = float(eq[-1] ** (ann / len(r)) - 1) if eq[-1] > 0 else -1.0
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": float(sd * math.sqrt(ann)),
        "sharpe": float(mean / (sd + 1e-12) * math.sqrt(ann)),
        "sortino": float(mean / (downside + 1e-12) * math.sqrt(ann)),
        "max_drawdown": mdd,
        "calmar": float(cagr / (abs(mdd) + 1e-12)),
        "hit_rate": float((r > 0).mean()),
        "avg_turnover_per_step": float(turnover / len(r)),
    }


def _econ_block(d: pd.DataFrame, h: int, cfg: EconConfig) -> dict[str, dict[str, float]]:
    """Only h==1 is used for a daily strategy; longer h -> overlapping, skipped."""
    d = d.sort_values("date")
    edge = np.clip(cfg.edge_scale * (d.p_up.to_numpy() - 0.5), -cfg.max_leverage, cfg.max_leverage)
    # realized next-period simple return for h==1
    fwd_simple = np.expm1(d.y_raw.to_numpy())
    gross = edge * fwd_simple
    turnover_cost = np.abs(np.diff(np.concatenate([[0.0], edge]))) * cfg.cost_per_turnover
    net = gross - turnover_cost

    strat = _perf(net, edge, cfg)
    bh = _perf(fwd_simple, np.ones_like(edge), cfg)

    # regime split by trailing vol (sigma_ewma)
    vq = pd.qcut(d.sigma_ewma, 5, labels=False, duplicates="drop")
    calm = net[vq == 0] if (vq == 0).any() else net[:0]
    storm = net[vq == vq.max()] if len(vq) else net[:0]
    return {
        "strategy": strat,
        "buy_hold": bh,
        "calmest_quintile": _perf(calm, edge[vq == 0] if (vq == 0).any() else edge[:0], cfg),
        "wildest_quintile": _perf(storm, edge[vq == vq.max()] if len(vq) else edge[:0], cfg),
    }


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def evaluate(oos: pd.DataFrame, econ_cfg: EconConfig | None = None) -> dict:
    econ_cfg = econ_cfg or EconConfig()
    result: dict = {"horizons": {}}
    for h in sorted(oos.horizon.unique()):
        d = oos[oos.horizon == h].copy()
        block = {"statistical": _stat_block(d, int(h))}
        if h == oos.horizon.min():
            block["economic"] = _econ_block(d, int(h), econ_cfg)
        result["horizons"][int(h)] = block
    result["n_oos_rows"] = int(len(oos))
    result["date_range"] = [str(oos.date.min())[:10], str(oos.date.max())[:10]]
    return result


def print_report(result: dict) -> None:
    print("=" * 74)
    print(
        f"CARN-X OOS EVALUATION   rows={result['n_oos_rows']}   "
        f"{result['date_range'][0]} -> {result['date_range'][1]}"
    )
    print("=" * 74)
    for h, block in result["horizons"].items():
        s = block["statistical"]
        print(f"\n---- horizon {h} day(s)  (n={s['n']}) " + "-" * 40)
        print(
            f"  directional accuracy (dist mean) : {s['dir_acc_mean']:.3f}   "
            f"(binom p vs 0.5 = {s['dir_binom_p']:.3f})"
        )
        print(f"  directional accuracy (dir head)  : {s['dir_acc_head']:.3f}")
        print(
            f"  RMSE  model / random-walk        : {s['rmse_model']:.4f} / {s['rmse_rw']:.4f}"
            f"   skill = {s['rmse_skill']:+.2%}"
        )
        print(f"  MAE model                        : {s['mae_model']:.4f}")
        print(f"  Student-t NLL (lower better)     : {s['nll']:.4f}")
        print(f"  PIT KS stat (0 = perfectly calib): {s['pit_ks']:.3f}")
        print(f"  90% interval coverage (target .90): {s['coverage_90']:.3f}")
        print(f"  Diebold-Mariano p vs RW          : {s['dm_pvalue_vs_rw']:.3f}")
        if "economic" in block:
            e = block["economic"]
            st, bh = e["strategy"], e["buy_hold"]
            print("\n  economic (daily strategy, edge->position, costs on):")
            print(f"    {'':16}{'strategy':>12}{'buy&hold':>12}")
            for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate"):
                sv = st.get(k, float("nan"))
                bv = bh.get(k, float("nan"))
                print(f"    {k:16}{sv:>12.3f}{bv:>12.3f}")
            print(f"    {'turnover/step':16}{st.get('avg_turnover_per_step', float('nan')):>12.3f}")
            cw, ww = e["calmest_quintile"], e["wildest_quintile"]
            print(
                f"    regime Sharpe  calm={cw.get('sharpe', float('nan')):+.2f}   "
                f"wild={ww.get('sharpe', float('nan')):+.2f}"
            )
    print("\n" + "=" * 74)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    runs = sorted(Path("runs").glob("*/oos_predictions.parquet"))
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    elif runs:
        path = runs[-1]
    else:
        raise SystemExit("no runs/ predictions found -- run walk_forward.py first")

    print(f"evaluating: {path}\n")
    oos = pd.read_parquet(path)
    print_report(evaluate(oos))
