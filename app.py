"""
CARN-X – Streamlit UI
======================
Single-file app (tabs instead of a pages/ folder, to keep the UI compact):
Data Input | Diagnosis | Routing/Gate | Training | Inference & Decision | Backtest.

Expensive objects (pipeline_bundle, trained model, computed results) live in
st.session_state so a widget interaction elsewhere doesn't recompute them.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import PipelineConfig, build_pipeline, run_backtest, run_inference, run_training
from training_extreme_engine import TrainConfig

st.set_page_config(page_title="CARN-X", layout="wide")
st.title("CARN-X — תחזיות מתמטיות")

if "pipeline_bundle" not in st.session_state:
    st.session_state.pipeline_bundle = None
if "price_series" not in st.session_state:
    st.session_state.price_series = None
if "price_dates" not in st.session_state:
    st.session_state.price_dates = None
if "nav_df" not in st.session_state:
    st.session_state.nav_df = None

tabs = st.tabs(["קלט דאטה", "אבחון", "ניתוב / Gate", "אימון", "אינפרנס והחלטה", "Backtest"])

# ---------------------------------------------------------------------------
# Tab 1: Data Input
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("קלט דאטה")
    price_file = st.file_uploader(
        "קובץ מחירים (CSV, עמודה אחת של מחירי סגירה)", type="csv", key="price_csv"
    )
    if price_file is not None:
        raw = pd.read_csv(io.BytesIO(price_file.getvalue()))
        col = raw.columns[0] if raw.shape[1] == 1 else st.selectbox("עמודת מחיר", raw.columns)
        st.session_state.price_series = raw[col].astype(float).values
        st.session_state.price_dates = pd.date_range(
            end=pd.Timestamp.now(), periods=len(st.session_state.price_series)
        )
        st.success(f"נטענו {len(st.session_state.price_series)} תצפיות מחיר.")

    nav_file = st.file_uploader(
        "קובץ NAV Premium (date, btc_holdings, btc_price_usd, shares_outstanding, stock_price_usd)",
        type="csv",
        key="nav_csv",
    )
    if nav_file is not None:
        from nav_premium import load_nav_csv

        tmp_path = "/tmp/_carnx_nav_upload.csv"
        with open(tmp_path, "wb") as f:
            f.write(nav_file.getvalue())
        st.session_state.nav_df = load_nav_csv(tmp_path)
        st.success(f"נטענו {len(st.session_state.nav_df)} שורות NAV.")
        st.dataframe(st.session_state.nav_df.tail(10))

    use_nav_macro = st.checkbox(
        "הפעל חיווט NAV premium אל הרשת הנוירונית (macro fusion)", value=False
    )

    if st.button("בנה Pipeline"):
        config = PipelineConfig(use_nav_macro=use_nav_macro and st.session_state.nav_df is not None)
        st.session_state.pipeline_bundle = build_pipeline(config)
        st.success("Pipeline נבנה.")

    if st.session_state.price_series is None:
        st.info("אין דאטה — ניתן להעלות CSV, או להשתמש בכפתור הדמו הסינתטי בהמשך.")
        if st.button("טען דאטה סינתטית לדמו"):
            rng = np.random.default_rng(42)
            t = np.arange(300)
            s = 100 + 0.02 * t + 3.0 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 1.1, 300)
            s[150:170] -= 12.0
            st.session_state.price_series = s
            st.session_state.price_dates = pd.date_range(end=pd.Timestamp.now(), periods=len(s))
            st.rerun()

# ---------------------------------------------------------------------------
# Tab 2: Diagnosis
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("שכבת אבחון")
    if st.session_state.price_series is None:
        st.warning("טען דאטה בטאב הראשון.")
    else:
        from diagnosis_layer import DiagnosisLayer

        diag = DiagnosisLayer().diagnose(st.session_state.price_series)
        st.session_state.diagnosis_result = diag
        c1, c2, c3 = st.columns(3)
        c1.metric("Historical Vol", f"{diag.volatility.historical_vol:.2%}")
        c2.metric("Max Drawdown", f"{diag.extremes.max_drawdown:.2%}")
        c3.metric("Skewness", f"{diag.basic.skewness:.2f}")
        st.line_chart(pd.Series(st.session_state.price_series, name="price"))
        st.write("**המלצות:**")
        for r in diag.recommendations:
            st.write(f"- {r}")

# ---------------------------------------------------------------------------
# Tab 3: Routing / Gate
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("שער ניתוב קומבינטורי")
    if "diagnosis_result" not in st.session_state:
        st.warning("עבור קודם בטאב האבחון.")
    else:
        from combinatorial_gate import CASE_LABELS, CombinatorialGate

        routing = CombinatorialGate().route(st.session_state.diagnosis_result).routing
        st.session_state.routing = routing
        st.write(f"**מקרה דומיננטי:** {CASE_LABELS.get(routing.dominant_case)}")
        gates_df = pd.DataFrame(
            {"gate": list(vars(routing.gates).keys()), "value": list(vars(routing.gates).values())}
        )
        st.bar_chart(gates_df.set_index("gate"))
        st.write(
            f"**Capacity hints:** dim={routing.embed_dim}, heads={routing.num_heads}, "
            f"depth={routing.depth}, activation={routing.activation_hint}"
        )

# ---------------------------------------------------------------------------
# Tab 4: Training
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("אימון")
    if st.session_state.pipeline_bundle is None:
        st.warning("בנה Pipeline בטאב הראשון.")
    else:
        epochs = st.slider("Epochs", 1, 50, 5)
        window = st.slider("Window", 16, 128, 48)
        if st.button("התחל אימון"):
            st.session_state.pipeline_bundle.config.train_config = TrainConfig(
                window=window,
                epochs=epochs,
                batch_size=16,
                log_every=1000,
            )
            progress = st.progress(0.0, text="מתחיל אימון...")
            log = st.empty()

            def _cb(epoch, total, loss):
                progress.progress(
                    (epoch + 1) / total, text=f"Epoch {epoch + 1}/{total} — loss={loss:.5f}"
                )
                log.write(f"Epoch {epoch + 1}/{total}: train_loss={loss:.5f}")

            series = st.session_state.price_series
            split = int(len(series) * 0.8)
            with st.spinner("מאמן... (הכרטיסייה חסומה עד שהאימון מסתיים)"):
                state = run_training(
                    st.session_state.pipeline_bundle,
                    series[:split],
                    series[split:],
                    progress_callback=_cb,
                )
            st.success(f"אימון הסתיים. best_val_loss={state.best_val_loss:.5f}")

# ---------------------------------------------------------------------------
# Tab 5: Inference & Decision
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("אינפרנס והחלטה")
    if st.session_state.pipeline_bundle is None:
        st.warning("בנה Pipeline בטאב הראשון.")
    else:
        if st.button("קבל החלטה"):
            decision = run_inference(
                st.session_state.pipeline_bundle,
                st.session_state.price_series,
                nav_df=st.session_state.nav_df,
                price_dates=st.session_state.price_dates,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Action", decision.action.name)
            c2.metric("Position Scale", f"{decision.position_scale:+.3f}")
            c3.metric("Confidence", f"{decision.confidence:.1%}")

            st.write("**הסברה:**")
            for line in decision.explanation:
                st.write(f"- {line}")

            if decision.leverage_recommendation is not None:
                lr = decision.leverage_recommendation
                st.write("### פאנל מינוף")
                c1, c2, c3 = st.columns(3)
                c1.metric("Leverage", f"{lr.recommended_leverage_ratio:.2f}x")
                c2.metric("Margin-call distance", f"{lr.margin_call_distance_pct:.1%}")
                c3.metric("Financing cost/yr", f"{lr.financing_cost_annual_pct_of_equity:.2%}")
                for n in lr.notes:
                    st.write(f"- {n}")

            if st.session_state.nav_df is not None:
                st.write("### פאנל NAV Premium")
                from nav_premium import NAVPremiumCalculator

                calc = NAVPremiumCalculator()
                computed = calc.compute(st.session_state.nav_df)
                snap = calc.latest_snapshot(computed)
                c1, c2, c3 = st.columns(3)
                c1.metric("Premium", f"{snap.premium:+.1%}")
                c2.metric("mNAV ratio", f"{snap.mnav_ratio:.3f}")
                c3.metric("Regime", snap.regime_label)

# ---------------------------------------------------------------------------
# Tab 6: Backtest
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Backtest")
    if st.session_state.pipeline_bundle is None:
        st.warning("בנה Pipeline בטאב הראשון.")
    else:
        if st.button("הרץ Backtest"):
            progress = st.progress(0.0, text="מריץ backtest...")

            def _cb(t, n):
                progress.progress(min(t / n, 1.0), text=f"t={t}/{n}")

            with st.spinner("רץ walk-forward..."):
                result = run_backtest(
                    st.session_state.pipeline_bundle,
                    st.session_state.price_series,
                    nav_df=st.session_state.nav_df,
                    price_dates=st.session_state.price_dates,
                    progress_callback=_cb,
                )
            st.line_chart(pd.Series(result.equity_curve, name="equity"))
            m = result.metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Return", f"{m.total_return:+.2%}")
            c2.metric("Sharpe", f"{m.sharpe:.3f}")
            c3.metric("Max Drawdown", f"{m.max_drawdown:.2%}")
            c4.metric("Win Rate", f"{m.win_rate:.1%}")
            st.text("\n".join(result.report))
