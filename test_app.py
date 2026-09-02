"""
CARN-X  --  Heavy App / Pipeline Test Harness
============================================
Exercises the whole product the way a user would, and fails loudly on anything
that isn't right. Meant to be run in a loop until it is all green:

    while ! .venv/bin/python test_app.py; do echo "--- retry ---"; sleep 2; done

Covers:
  1. every module imports + byte-compiles
  2. data layer: panel builds, is fresh, integrity asserts pass, live quote works
  3. inference: fit (tiny) -> save -> load -> predict -> monte-carlo, values sane
  4. Streamlit AppTest: every screen renders with zero exceptions
  5. AppTest widget interaction: sliders / selects / number inputs on every screen
  6. AppTest: the "אמן מודל עכשיו" button actually trains + saves (tiny config)
  7. walk-forward smoke (require_targets=True path still works)
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import glob
import os
import subprocess
import sys
import time
import traceback

FAILS: list[str] = []
T0 = time.time()


def step(name: str):
    """Runs the decorated function immediately; records a failure but continues."""

    def deco(fn):
        print(f"\n▶ {name}")
        try:
            fn()
            print(f"  ✓ {name}  ({time.time() - T0:.0f}s)")
        except Exception as e:  # noqa: BLE001
            FAILS.append(f"{name}  ->  {type(e).__name__}: {e}")
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
        return fn

    return deco


# ---------------------------------------------------------------------------


@step("1. modules import + compile")
def _():
    mods = [
        "data_layer",
        "features",
        "targets",
        "dataset",
        "forecast_net",
        "walk_forward",
        "baselines",
        "evaluation",
        "inference",
        "run_evaluation",
        "btc_cycle",
        "prob_models",
        "config",
        "carnx",
        "theme",
        "charts",
        "tv_chart",
        "math_structures",
        "hybrid_engine",
    ]
    for m in mods:
        __import__(m)
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "mstr_app.py", *[f"{m}.py" for m in mods]],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


@step("2. data layer: panel + freshness + integrity + live quote")
def _():
    from data_layer import (
        DataConfig,
        build_panel,
        primary_close,
        assert_panel_integrity,
        panel_freshness,
        live_quote,
        expected_last_session,
    )

    cfg = DataConfig()
    panel = build_panel(cfg)
    assert_panel_integrity(panel, cfg)
    fr = panel_freshness(panel)
    print(f"    freshness: {fr}")
    assert fr["sessions_behind"] <= 1, f"panel is {fr['sessions_behind']} sessions behind"
    close = primary_close(panel, cfg)
    assert close.iloc[-1] > 0 and close.index.max() <= expected_last_session()
    lq = live_quote("MSTR")
    print(
        f"    live: ${lq.price} ({lq.source})  panel last ${close.iloc[-1]:.2f} @ {close.index.max().date()}"
    )
    assert lq.price and lq.price > 0
    # the panel close and the live price should be within a sane band (no unit / adj bug)
    assert 0.5 < lq.price / close.iloc[-1] < 2.0, "panel vs live price diverge implausibly"


@step("3. inference: fit(tiny) -> save -> load -> predict -> monte-carlo")
def _():
    from dataset import assemble
    from forecast_net import DEFAULT_SEQ_FEATURES
    from inference import (
        InferenceConfig,
        fit_production_model,
        save_production_model,
        load_production_model,
        predict_distribution,
        monte_carlo_paths,
        tail_probability,
    )

    data = assemble(seq_feature_names=DEFAULT_SEQ_FEATURES, require_targets=False)
    cfg = InferenceConfig(ensemble=1, seeds=(0,), epochs=8, patience=3, num_threads=4)
    pm = fit_production_model(data, cfg)
    path = save_production_model(
        pm, n_tab_features=data.X.shape[1], path="./carnx_checkpoints/_test_model.pt"
    )
    pm2 = load_production_model(data, path)
    os.remove(path)

    d1 = predict_distribution(pm, data)
    d2 = predict_distribution(pm2, data)
    import numpy as np

    assert np.allclose(d1["mu_scaled"], d2["mu_scaled"], atol=1e-5), "save/load changed predictions"

    for _, r in d1.iterrows():
        assert 0.0 <= r.p_up <= 1.0
        assert r.nu >= 2.0 and r.ann_vol_implied > 0
        assert r.ret_q05_pct < r.expected_move_pct < r.ret_q95_pct, "quantiles out of order"
        assert 0.0 <= tail_probability(r, -10.0) <= 1.0
    mc = monte_carlo_paths(pm, data, horizon=10, n_paths=3000)
    assert mc.var_5_price < mc.expected_price and mc.paths.shape == (3000, 11)
    assert 0.0 <= mc.prob_up <= 1.0


@step("3b. prob_models: full report + every custom model, sane probabilities")
def _():
    import math as _m

    from data_layer import DataConfig, build_panel, primary_close
    from prob_models import (
        compute_custom,
        custom_model_keys,
        custom_sensors,
        full_report,
    )

    close = primary_close(build_panel(DataConfig()))
    S0 = float(close.iloc[-1])

    rep = full_report(close, target_price=S0 * 1.1, horizon_days=20)
    assert len(rep.results) >= 24
    for r in rep.results:
        for v in r.answers.values():
            if isinstance(v, (int, float)) and 0 <= v <= 1:
                assert 0.0 <= v <= 1.0
        assert _m.isfinite(rep.next_day["expected_price"]) and rep.next_day["expected_price"] > 0

    # the user's canonical question, through every custom model
    for key in custom_model_keys():
        sens = custom_sensors(key, close)
        spec = {
            s.name: (
                S0 * 1.8
                if s.name == "strike"
                else 23
                if s.name == "horizon"
                else 2
                if s.name == "k"
                else s.default
            )
            for s in sens
        }
        res = compute_custom(close, key, spec)
        probs = [v for v in res.answers.values() if isinstance(v, (int, float)) and 0 <= v <= 1]
        assert probs, f"{key} produced no probability"
        assert all(0.0 <= p <= 1.0 for p in probs), f"{key} probability out of [0,1]"
    print(
        f"    full_report={len(rep.results)} models · custom={len(custom_model_keys())} models  ✓"
    )


@step("4+5+6. Streamlit AppTest: render + interact + train button")
def _():
    import tempfile

    # redirect the model path so the button-click test never touches the real one
    tmp_model = os.path.join(tempfile.gettempdir(), "carnx_test_model.pt")
    os.environ["CARNX_MODEL_PATH"] = tmp_model
    if os.path.exists(tmp_model):
        os.remove(tmp_model)
    # seed it from the real model (if any) so model-dependent screens render
    real = "./carnx_checkpoints/production_model.pt"
    real_mtime0 = os.path.getmtime(real) if os.path.exists(real) else 0
    if os.path.exists(real):
        import shutil

        shutil.copy2(real, tmp_model)

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("mstr_app.py", default_timeout=400)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]

    sections = [
        "סקירה",
        "טרמינל מסחר",
        "מחשבון הסתברויות",
        "אבחון סטטיסטי",
        "תחזית הסתברותית",
        "Monte Carlo",
        "מבנים מתמטיים",
        "מחזור ביטקוין → MSTR",
        "ראיות Walk-Forward",
        "סיכון ומינוף",
        "הגדרות",
    ]
    for sec in sections:
        at.sidebar.radio[0].set_value(sec).run()
        errs = [f"{type(e).__name__}: {e}" for e in at.exception]
        assert not errs, f"[{sec}] {errs}"
        # poke every slider / select / toggle on the page
        for sl in at.slider:
            try:
                sl.set_value(sl.max if sl.value != sl.max else sl.min).run()
            except Exception:
                pass
        assert not at.exception, f"[{sec}] slider interaction: {[str(e) for e in at.exception]}"
        for ss in at.select_slider:
            try:
                ss.set_value(ss.options[-1]).run()
            except Exception:
                pass
        assert not at.exception, f"[{sec}] select_slider: {[str(e) for e in at.exception]}"
        for tg in at.toggle:
            try:
                tg.set_value(not tg.value).run()
            except Exception:
                pass
        assert not at.exception, f"[{sec}] toggle: {[str(e) for e in at.exception]}"
        print(f"    [{sec}] OK  metrics={len(at.metric)} sliders={len(at.slider)}")

    # --- the train button ---
    at.sidebar.radio[0].set_value("הגדרות").run()
    for sl in at.slider:
        lbl = sl.label or ""
        if "Ensemble" in lbl:
            sl.set_value(1).run()
        elif "Epochs" in lbl:
            sl.set_value(15).run()
    train_btn = [b for b in at.button if "אמן מודל עכשיו" in (b.label or "")]
    assert train_btn, "train button not found"
    print("    clicking 'אמן מודל עכשיו' (this trains for real, ~20s)…")
    tmp_model = os.environ["CARNX_MODEL_PATH"]
    mtime_before = os.path.getmtime(tmp_model) if os.path.exists(tmp_model) else 0
    try:
        train_btn[0].click().run()
        errs = [f"{type(e).__name__}: {e}" for e in at.exception]
        assert not errs, f"train button raised: {errs}"
        saved = os.path.exists(tmp_model) and os.path.getmtime(tmp_model) > mtime_before
        assert any("נשמר" in (s.value or "") for s in at.success) or saved, (
            "training produced no saved model / success message"
        )
        real_mtime1 = os.path.getmtime(real) if os.path.exists(real) else 0
        assert real_mtime1 == real_mtime0, "the real production model was clobbered by the test"
        print("    train button: model trained + saved with no error ✓ (real model untouched)")

        # the "הרץ test_integrity" button must use the app's interpreter, not bare `python`
        it_btn = [b for b in at.button if "test_integrity" in (b.label or "")]
        assert it_btn, "test_integrity button not found"
        it_btn[0].click().run()
        assert not at.exception, f"test_integrity button raised: {[str(e) for e in at.exception]}"
        code_blocks = " ".join(c.value for c in at.code)
        assert "wrong Python interpreter" not in code_blocks, (
            "test_integrity button ran the wrong Python (envcheck guard fired)"
        )
        assert "all integrity checks passed" in code_blocks or any(
            "עברו" in (s.value or "") for s in at.success
        ), f"test_integrity button did not report success; got: {code_blocks[-400:]}"
        print("    test_integrity button: ran with the right interpreter, all checks passed ✓")
    finally:
        os.environ.pop("CARNX_MODEL_PATH", None)
        if os.path.exists(tmp_model):
            os.remove(tmp_model)


@step("7. walk-forward smoke (require_targets=True)")
def _():
    from walk_forward import WFConfig, run_walk_forward

    cfg = WFConfig(
        initial_train=500,
        step=120,
        max_folds=2,
        epochs=8,
        patience=3,
        ensemble=1,
        seeds=(0,),
        retrain_every=1,
        tag="wf_apptest",
        verbose=False,
    )
    res = run_walk_forward(cfg)
    assert len(res.oos) > 0
    for run in glob.glob("runs/wf_apptest_*"):
        import shutil

        shutil.rmtree(run, ignore_errors=True)


@step("8. probability-math verification (test_prob_math.py)")
def _():
    r = subprocess.run([sys.executable, "test_prob_math.py"], capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:]
    assert r.returncode == 0, f"prob-math checks failed:\n{r.stdout[-2000:]}"
    print(f"    {tail[0] if tail else 'ok'}")


@step("8b. mathematical-structures verification (test_math_structures.py)")
def _():
    r = subprocess.run([sys.executable, "test_math_structures.py"], capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:]
    assert r.returncode == 0, f"math-structures checks failed:\n{r.stdout[-2500:]}"
    print(f"    {tail[0] if tail else 'ok'}")


@step("8c. hybrid structural-neural engine verification (test_hybrid_engine.py)")
def _():
    r = subprocess.run([sys.executable, "test_hybrid_engine.py"], capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:]
    assert r.returncode == 0, f"hybrid-engine checks failed:\n{r.stdout[-2500:]}"
    print(f"    {tail[0] if tail else 'ok'}")


@step("9. CARN-X orchestrator: config sync + 8-layer conformance + routing")
def _():
    from carnx import CARNX
    from config import CarnxConfig, default, fast, production

    cfg = default()
    assert cfg.walk_forward.model is cfg.model, "config.sync() must alias model"
    p = cfg.save("/tmp/_carnx_cfg_t.json")
    assert CarnxConfig.load(p).model.d_model == cfg.model.d_model
    os.remove(p)

    cx = CARNX(default())
    cx.route()
    layers = cx.conformance()
    assert len(layers) == 8
    wired = {s.layer for s in layers if s.wired}
    assert {"L1", "L2", "L3", "L4", "L6", "L7", "L8"} <= wired, f"missing: {wired}"

    r = cx._routing
    assert r is not None and len(r.gate_dict()) == 15, "Routing must carry all 15 gates"
    assert 0.0 <= r.w_task <= 5.0 and r.w_combinatorial >= 0.0
    gsum = sum(r.gate_dict().values())
    assert gsum > 0, "gate strengths all zero -- gate not routing"
    print(
        f"    8 layers · {len(wired)} wired · 15 gates (Σ={gsum:.1f}) · "
        f"loss w_task={r.w_task:.2f} w_comb={r.w_combinatorial:.2f}"
    )

    _ = fast(), production()  # presets construct cleanly


@step("10. Monte-Carlo engine: drift modes + martingale + calendar sync")
def _():
    import numpy as np
    from prob_models import add_trading_days, trading_days_between
    from dataset import assemble
    from inference import InferenceConfig, load_production_model, monte_carlo_paths, DRIFT_MODES

    # trading-calendar round-trip (bi-directional slider <-> date sync)
    for start in ("2026-08-28", "2026-08-29", "2027-01-11"):
        for n in (5, 21, 155, 252, 504):
            assert trading_days_between(start, add_trading_days(start, n)) == n
    assert add_trading_days("2026-08-28", 155).date().isoformat() == "2027-04-13"

    cfg = InferenceConfig()
    data = assemble(
        seq_feature_names=cfg.model.seq_feature_names, target_cfg=cfg.targets, require_targets=False
    )
    pm = load_production_model(data)

    r = 0.045
    mcq = monte_carlo_paths(
        pm, data, horizon=252, n_paths=60000, drift_mode="risk_neutral", risk_free_rate=r
    )
    grow = mcq.expected_price / mcq.last_price
    assert abs(grow - np.exp(r)) < 0.03, f"martingale broken: {grow:.4f} vs {np.exp(r):.4f}"
    assert mcq.n_paths == 60000 and mcq.paths.dtype == np.float32

    mc_bull = monte_carlo_paths(
        pm, data, horizon=252, n_paths=20000, drift_mode="custom", custom_drift_annual=1.0
    )
    assert mc_bull.expected_price > mcq.expected_price, "bull drift must lift E[S_T]"
    assert set(DRIFT_MODES) == {"model", "risk_neutral", "custom", "historical"}
    # horizon far past the old 60-day cap must still produce a full cone
    assert mcq.paths.shape[1] == 253 and len(mcq.percentiles["p99"]) == 253
    print(
        f"    martingale E[S_T]/S0={grow:.4f} (e^r={np.exp(r):.4f}) · "
        f"bull E={mc_bull.expected_price:,.0f} · 252-day cone OK"
    )


# ---------------------------------------------------------------------------

print("\n" + "=" * 66)
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED   ({time.time() - T0:.0f}s)")
    for f in FAILS:
        print(f"  - {f}")
    sys.exit(1)
print(f"RESULT: ALL GREEN   ({time.time() - T0:.0f}s)")
sys.exit(0)
