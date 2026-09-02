# CARN-X — MSTR Probabilistic Forecasting

A research-grade console that forecasts the **full predictive distribution** of
MicroStrategy (MSTR) forward returns at 1 / 5 / 20-day horizons — grounded in real
historical data, fat-tailed probabilistic models and honest out-of-sample
evaluation. It is **not** a black box that claims to print money: every claim is
checked walk-forward against honest baselines, with calibration diagnostics.

> ⚠️ Not investment advice. A distribution / volatility / regime instrument that
> does **not** predict price direction and makes no claim of profitability.

**Live demo:** _deploy to [Streamlit Community Cloud](https://share.streamlit.io)
in ~2 min — see [Deploy](#deploy) below._

```bash
git clone <this-repo> && cd carnx-mstr
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
streamlit run mstr_app.py            # -> http://localhost:8501
```

The repo ships a trained 9-net production checkpoint
(`carnx_checkpoints/production_model.pt`, 5.8 MB) and a cached market panel, so the
app works on first launch; **הגדרות → אמן מודל עכשיו** retrains it on fresh data.

---

## The 8-layer architecture (`ארכיטקטורה והתגלמות לפריסת עבודה.txt`)

`carnx.CARNX` is the named orchestrator; `config.CarnxConfig` is threaded through
every layer (one edit reconfigures the whole pipeline). `python carnx.py` prints
the live conformance report.

| L | layer | module(s) | wired |
|---|-------|-----------|-------|
| L1 | Diagnosis & characterisation | `diagnosis_layer` (GARCH/STL/EVT/change-points/order-rep-sym) | ✓ |
| L2 | Combinatorial Gate | `combinatorial_gate` → `forecast_net.Routing` (4 cases + Bayesian posterior + **15 attention gates** + `LossSchedule`) | ✓ |
| L3 | Strategic gated modules | `forecast_net` experts: Fourier · Fibonacci(GRU) · **Wavelet(à-trous)** · **GARCH(GRU)** · **ChangePoint** · Extreme · MC-dropout | ✓ |
| L4 | Hybrid neural core | `forecast_net.ForecastNet` — RoPE causal transformer + tabular MLP → Student-t heads | ✓ |
| L5 | Macro-Micro fusion | `macro_micro_fusion` + `nav_premium` | ○ off by default (needs NAV CSV) |
| L6 | Combinatorial + financial loss | `forecast_net.forecast_loss` — weights from L2's `LossSchedule` (`w_task/w_combinatorial/w_structure`), risk-aversion up-weights extreme samples | ✓ |
| L7 | Inference & decision | `inference` + `inference_decision_layer` + `leverage_factorization` | ✓ |
| L8 | Evaluation & backtest | `walk_forward` + `evaluation` + `baselines` | ✓ |

Analytic companions (not the neural path): `prob_models` (25 distributions +
combinatorics + Markov/GARCH), `btc_cycle` (4-year halving cycle → MSTR scenario),
`math_structures` (the MSTR series re-expressed through named branches of
mathematics — Cox–Ross–Rubinstein binomial lattice = Pascal's triangle,
Catalan/ballot drawdown-survival, Koopman/DMD linear-operator spectrum,
Fokker–Planck density evolution, Merton jump-diffusion by EM, Wald–Wolfowitz /
arcsine path statistics, and an *honest* Fibonacci pivot-clustering test —
each checked against a closed form, Monte-Carlo, or the empirical record;
tested in `test_math_structures.py`, shown in the app's *מבנים מתמטיים* screen).


A research-grade pipeline that forecasts the **full predictive distribution** of
MicroStrategy (MSTR) forward log-returns at 1 / 5 / 20-day horizons, grounded in
real historical data, probabilistic modelling and statistical evaluation.

It is **not** a black box that claims to print money. Every claim is checked
out-of-sample with a walk-forward harness, against honest baselines, with
calibration diagnostics.

---

## Pipeline (new)

| stage | module | what it does |
|-------|--------|--------------|
| 1. data | `data_layer.py` | daily OHLCV for MSTR + BTC, IBIT, MSTU/MSTZ, GLD, ^VIX, ^TNX, DXY, S&P, Nasdaq via `yfinance`; aligned to the MSTR calendar, cached to parquet |
| 2. features | `features.py` | ~250 **strictly causal** features: multi-lag returns, realized/EWMA/Parkinson vol, rolling skew/kurt, drawdown, RSI/MACD, rolling MSTR↔BTC beta & idiosyncratic residual, leverage-ETF decay, macro levels/changes, calendar, missingness flags; winsorized |
| 3. targets | `targets.py` | vol-normalized forward log-returns `y = (logret_h − drift) / (σ_ewma·√h)`; also raw returns, forward realized vol, up/down labels, recency sample weights |
| 4. windowing | `dataset.py` | sequence windows for the temporal encoder + full tabular vector; **train-only** fold scaling |
| 5. model | `forecast_net.py` | hybrid net: RoPE causal Transformer over 12 curated series + MLP over the full feature vector → per-horizon **Student-t** head (μ, σ, ν) + calibrated direction head + forward-vol head. Optional Fourier / recurrent / extreme experts, gated by the legacy `CombinatorialGate` (ablatable) |
| 6. training | `walk_forward.py` | expanding-window walk-forward with embargo, seed ensembling, early stopping; emits genuine OOS predictive distributions |
| 7. baselines | `baselines.py` | random walk, drift, EWMA, AR(1), GARCH(1,1), Ridge, HistGradientBoosting, Logistic — same folds/targets/metrics |
| 8. evaluation | `evaluation.py` | directional accuracy + binomial p, RMSE skill vs RW, Student-t NLL, PIT calibration (KS) + 90% coverage, Diebold–Mariano vs RW, and a costed walk-forward strategy with regime split |
| 9. production inference | `inference.py` | train once on all history → `ProductionModel` (save/load), predict the latest bar's distribution, tail probabilities, and a Monte-Carlo price cone. MC horizon 5–504 trading days; **configurable drift** (`model` / `risk_neutral` r / `custom` scenario μ / `historical`) with an empirical fat-tail martingale correction; standardised Student-t innovations (Var=1, Δt=1/252); antithetic variates; first-passage / close / drawdown / VaR·CVaR metrics |
| 10. BTC cycle | `btc_cycle.py` | full BTC history + halving-cycle features (causal) + aligned cycle template + BTC→MSTR long-horizon structural scenario + macro overlay |
| 11. prob calculator | `prob_models.py` | classical distributions (normal/binomial/poisson/geometric/uniform) parameterised from 2y MSTR data, each with E/Var/SD; "next-day expected price". App screen is split one-distribution-per-page: two inputs only (target price + horizon) → each distribution's answer + a probability-vs-time-axis curve; one shared 40k-path sim; "מתקדם" tab keeps the full family tables |
| 11b. hybrid engine | `hybrid_engine.py` | **Hybrid Neural-Structural MSTR engine.** Values MSTR as the leveraged reflexive claim it is: `S_t = (BTC-per-share·S_t^BTC − Debt-per-share)·m_NAV`. Three layers — (1) a 3-state Gaussian regime classifier (bear/accumulation/expansion) with an empirical transition matrix, (2) a bivariate SDE: regime-switching **Merton jump-diffusion** for BTC × a bounded **Ornstein–Uhlenbeck** mNAV multiple, Cholesky-correlated, (3) a **reflexive ATM-accretion loop** (issuing above NAV grows sats-per-share). Scenario mode calibrates the BTC drift **drag-neutral** so the *median* terminal hits the assumed BTC price — killing the artificial −½σ² decay on long cyclical bull paths. Vectorised float32 + antithetic; `HybridResult.to_mc_result()` feeds the existing MC-screen UI. Tested in `test_hybrid_engine.py` (structural identity, drag-neutral convergence, reflexivity, antithetic variance reduction). Selectable on the app's Monte-Carlo screen (Mode 1 = hybrid, Mode 2 = pure statistical). |
| 12. app | `mstr_app.py` | Streamlit console: snapshot · probability calculator · diagnosis · probabilistic forecast · Monte-Carlo (hybrid **or** statistical engine) · mathematical structures · BTC-cycle scenario · OOS evidence · risk/leverage. Presentation split out into `theme.py` (neon-aurora skin, `hero()` banners) + `charts.py` (candlestick, donut/pie, gauge, gradient-area, diverging bars, ranked hbars, bullet, ridgeline, PIT reliability, complex-plane, surface-heat, stem) — visual layer only, no logic |

### Legacy CARN-X layers (kept)
`diagnosis_layer.py` (statistical characterization), `combinatorial_gate.py`
(routing → expert strengths), `nav_premium.py` (MSTR NAV-premium features from a
manual CSV), `leverage_factorization.py` (advisory account-leverage math),
`macro_micro_fusion.py`. `hybrid_neural_core.py` / `inference_decision_layer.py`
/ `training_extreme_engine.py` / `evaluation_backtest.py` / `pipeline.py` /
`app.py` are the earlier prototype and are being superseded by the modules above.

---

## Quick start

```bash
pip install -r requirements.txt

python data_layer.py         # fetch + cache the panel
python test_integrity.py     # data / leakage / embargo guard rails
python test_app.py           # heavy end-to-end: data, inference, every app screen, the train button
python stress_test.py        # adversarial: chart fuzz, MC-engine invariant sweep,
                             # calendar round-trip, every widget swept to its edges

python run_evaluation.py            # neural walk-forward + all baselines + comparison table
python run_evaluation.py --fast     # fewer folds/seeds for a quick read
python run_evaluation.py --no-experts   # ablate the combinatorial-gate experts

# or step by step:
python walk_forward.py           # quick 3-fold smoke   (--full for the whole history)
python baselines.py
python evaluation.py runs/<run>/oos_predictions.parquet
```

### The app

```bash
python inference.py              # trains a small model, prints the latest forecast (sanity check)
python train_production.py --eval   # DEEP retrain: 9-seed ensemble (config.production_xl)
                                    # -> save/reload/sanity -> promote checkpoint
                                    # -> acceptance report (MC cone x 4 drift modes, martingale)
                                    # -> full walk-forward OOS (calibration / skill)
streamlit run mstr_app.py        # the full console
```

Latest deep retrain OOS (49 folds, 2022-07 → 2026-07): PIT-KS 0.056 / 0.053 /
0.024 at h = 1 / 5 / 20, 90% coverage 0.93 / 0.94 / 0.90, directional accuracy
0.50 / 0.50 / 0.53 (only h20 marginally beats a coin, p = 0.022). Calibrated
distribution + volatility + regime — **not** a direction caller.

In the app: **הגדרות → אמן מודל עכשיו** trains the production model (all history,
~1–4 min) and caches it to `carnx_checkpoints/production_model.pt`; every other
screen then loads it instantly.

---

## Deploy

The repo is deploy-ready for **[Streamlit Community Cloud](https://share.streamlit.io)**
(free, permanent `*.streamlit.app` URL):

1. Push this repo to GitHub (private or public — Community Cloud handles both).
2. Go to **share.streamlit.io → Create app → Deploy a public app from GitHub**.
3. Repo = this repo · Branch = `main` · **Main file path = `mstr_app.py`**.
4. Click **Deploy**. First build ≈ 5–8 min (it installs the CPU-only PyTorch wheel
   pinned in `requirements.txt`); afterwards the app reloads in seconds.

`.python-version` pins Python 3.12; `requirements.txt` is the runtime set
(`requirements-dev.txt` adds `ruff` + `playwright` for local checks only). The
bundled checkpoint makes the deployed app usable immediately; it refreshes market
data from Yahoo on demand.

---

## Project layout

```
mstr_app.py            Streamlit entry point (11 screens)
carnx.py               CARN-X orchestrator — one method per layer L1–L8
config.py              CarnxConfig — the single config threaded through everything

data_layer.py          L-data   · yfinance panel + live quotes + freshness
features.py             feat    · ~265 strictly-causal features
targets.py              target  · vol-normalised de-trended forward log-returns
dataset.py              window  · sequence windows + train-only FoldScaler
diagnosis_layer.py     L1  · regime characterisation (GARCH / STL / EVT / change-points)
combinatorial_gate.py  L2  · diagnosis -> Routing (15 gates + LossSchedule)
forecast_net.py         L3+L4 · RoPE transformer ∥ tabular MLP -> Student-t heads
macro_micro_fusion.py  L5  · NAV-premium macro fusion (opt-in)
walk_forward.py        L6+L8 · expanding-window walk-forward + gate-weighted loss
baselines.py            L8  · 8 honest baselines on the same folds
evaluation.py           L8  · calibration / skill / Diebold-Mariano / costed strategy
inference.py           L7  · train-once production model + Monte-Carlo cone

prob_models.py               25+ classical distributions + GARCH + Markov + MC
btc_cycle.py                 4-year halving-cycle -> MSTR structural scenario
hybrid_engine.py             bivariate Merton-jump BTC × OU mNAV reflexivity engine
math_structures.py           the "מבנים מתמטיים" screen (DMD, puzzle map, …)

theme.py                     neon-aurora skin (global CSS) + hero() + Altair theme
charts.py                    16 themed Altair chart builders
tv_chart.py                  lightweight-charts trading terminal (rendered in an iframe)

test_*.py                    integrity · heavy end-to-end · prob-math · structures · hybrid
stress_test.py               adversarial: chart fuzz + MC sweep + widget edge sweep
mobile_audit.py              headless-Chrome iPhone-viewport overflow check
train_production.py          deep retrain runner (config.production_xl, --eval)
run_evaluation.py            neural walk-forward + all baselines + comparison table

pyproject.toml               ruff config (format + lint; the `import envcheck` guard is protected)
envcheck.py                  interpreter guard — a helpful message if deps are missing
```

---

## Anti-look-ahead guarantees (enforced by `test_integrity.py`)

- **feature causality** — features computed on a truncated panel are byte-identical
  to the full run sliced to the same date (no feature can peek forward)
- **no single-feature leak** — |corr(feature_t, next_return)| < 0.5 for every column
- **fold scaling** — winsor + standardize statistics come only from the training slice
- **embargo** — `max(horizon)` rows are skipped between train and test; a training
  label can never overlap the test window
- **exact target inverse** — `descale()` reproduces the raw forward return, drift included
- **panel integrity** — sorted, unique, weekday-only, no partial/today bar, split-adjusted

---

## What the data says (honest read)

- **Daily direction on MSTR is close to a coin flip for every method**, neural
  and classical alike. Any edge is small and horizon/regime dependent.
- The **predictable** parts are volatility, the return *distribution* (fat tails,
  Student-t ν ≈ 4–8) and **regime** — which is exactly what the probabilistic
  heads target.
- The costed strategy's value shows up as **drawdown control** (much lower vol /
  MaxDD than buy-and-hold), and it performs very differently in calm vs
  high-vol regimes — see the regime split in `evaluation.py`.

Treat outputs as **probabilistic scenarios for position sizing**, never as a
promise of returns.
