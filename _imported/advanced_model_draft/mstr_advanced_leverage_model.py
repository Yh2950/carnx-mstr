"""
MSTR Advanced Predictive & Leverage Model
=========================================
מודל עצמאי לחלוטין – לא תלוי בקוד המקורי שלך.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ===============================
# 1. DATA LAYER – היסטוריה
# ===============================
def fetch_mstr_data(ticker="MSTR", period="2y"):
    """
    מביא נתונים היסטוריים.
    בפועל משתמשים ב-yfinance. כאן יש fallback סינתטי למקרה שאין אינטרנט.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        # סימולציה סבירה של MSTR (תנודתיות גבוהה)
        np.random.seed(42)
        dates = pd.date_range(end=datetime.today(), periods=500, freq="B")
        returns = np.random.normal(0.0015, 0.045, len(dates))  # MSTR-like
        price = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame({
            "Open": price * (1 + np.random.normal(0, 0.005, len(dates))),
            "High": price * (1 + np.abs(np.random.normal(0, 0.015, len(dates)))),
            "Low":  price * (1 - np.abs(np.random.normal(0, 0.015, len(dates)))),
            "Close": price,
            "Volume": np.random.randint(1_000_000, 8_000_000, len(dates))
        }, index=dates)
        return df

# ===============================
# 2. LEVERAGE & RISK ENGINE
# ===============================
def calculate_leverage_metrics(df, risk_free=0.04):
    """
    מחשב מדדי מינוף וסיכון קלאסיים ל-MSTR
    """
    returns = df["Close"].pct_change().dropna()

    # Beta מול BTC (אם יש) או מול עצמו
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252 - risk_free) / volatility
    max_dd = (df["Close"] / df["Close"].cummax() - 1).min()

    # VaR היסטורי ופרמטרי
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)

    # Effective leverage (MSTR נחשבת ~2-3x BTC בדרך כלל)
    # כאן אנחנו מודדים leverage "פנימי" דרך תנודתיות
    effective_leverage = volatility / 0.60   # 60% = תנודתיות BTC משוערת

    return {
        "annual_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "VaR_95": var_95,
        "VaR_99": var_99,
        "effective_btc_leverage": effective_leverage,
        "returns": returns
    }

def leverage_position_size(capital, price, leverage=2.0, stop_loss_pct=0.08):
    """
    מחשב גודל פוזיציה עם מינוף ו-stop loss
    """
    risk_per_trade = capital * 0.01          # 1% מההון
    stop_distance = price * stop_loss_pct
    shares = (risk_per_trade / stop_distance) * leverage
    notional = shares * price
    margin_required = notional / leverage
    return {
        "shares": shares,
        "notional_exposure": notional,
        "margin_required": margin_required,
        "liquidation_price_approx": price * (1 - 1/leverage * 0.85)
    }

# ===============================
# 3. PROBABILISTIC MODELS
# ===============================
def fit_distributions(returns):
    """
    מתאים התפלגויות שונות לתשואות
    """
    mu, sigma = stats.norm.fit(returns)
    df_t, loc_t, scale_t = stats.t.fit(returns)

    # Poisson על ימים חיוביים/שליליים (ספירה)
    up_days = (returns > 0).sum()
    down_days = (returns < 0).sum()

    return {
        "normal": {"mu": mu, "sigma": sigma},
        "student_t": {"df": df_t, "loc": loc_t, "scale": scale_t},
        "up_days": up_days,
        "down_days": down_days
    }

def monte_carlo_forecast(last_price, mu, sigma, days=30, n_sims=5000):
    """
    סימולציית Monte Carlo קלאסית
    """
    dt = 1/252
    paths = np.zeros((n_sims, days))
    paths[:, 0] = last_price

    for t in range(1, days):
        shock = np.random.normal(0, 1, n_sims)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shock)

    return paths

# ===============================
# 4. NEURAL NETWORK – LSTM PREDICTION
# ===============================
def create_lstm_model(lookback=30):
    """
    בונה מודל LSTM פשוט (TensorFlow / Keras)
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        return model
    except ImportError:
        print("TensorFlow לא מותקן – מדלגים על LSTM")
        return None

def prepare_lstm_data(series, lookback=30):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

# ===============================
# 5. NEWS / ANALYST SENTIMENT ENGINE
# ===============================
def analyze_news_sentiment(texts):
    """
    ניתוח סנטימנט בסיסי + מילות מפתח של אנליסטים.
    בגרסה מלאה מחליפים ב-transformers (FinBERT / RoBERTa).
    """
    positive_words = [
        "bullish", "buy", "outperform", "overweight", "accumulate",
        "strong", "upside", "catalyst", "bitcoin", "btc", "treasury",
        "leverage", "nav", "premium", "undervalued"
    ]
    negative_words = [
        "bearish", "sell", "underperform", "underweight", "reduce",
        "risk", "dilution", "debt", "volatility", "overvalued",
        "premium", "concern", "caution"
    ]

    results = []
    for text in texts:
        text_l = text.lower()
        pos = sum(1 for w in positive_words if w in text_l)
        neg = sum(1 for w in negative_words if w in text_l)
        score = (pos - neg) / max(pos + neg, 1)
        results.append({
            "text": text[:120] + "...",
            "score": score,
            "label": "Bullish" if score > 0.15 else "Bearish" if score < -0.15 else "Neutral"
        })
    return results

# דוגמה לכותרות אנליסטים (אפשר להחליף ב-API אמיתי)
EXAMPLE_NEWS = [
    "MicroStrategy continues aggressive Bitcoin accumulation strategy, analysts remain bullish on leveraged exposure",
    "MSTR premium to NAV widens as BTC rallies; some brokers flag dilution risk from equity raises",
    "JPMorgan maintains overweight on MSTR citing strong treasury and optionality",
    "Concerns grow over MSTR debt levels and high volatility relative to pure BTC exposure"
]

# ===============================
# 6. MAIN PIPELINE
# ===============================
def run_full_model():
    print("=" * 60)
    print("MSTR ADVANCED MODEL – Starting...")
    print("=" * 60)

    # 1. נתונים
    df = fetch_mstr_data()
    print(f"\nLoaded {len(df)} trading days | Last close: {df['Close'].iloc[-1]:.2f}")

    # 2. מינוף וסיכון
    risk = calculate_leverage_metrics(df)
    print("\n--- LEVERAGE & RISK ---")
    for k, v in risk.items():
        if k != "returns":
            print(f"{k:25}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # 3. התפלגויות
    dist = fit_distributions(risk["returns"])
    print("\n--- PROBABILISTIC FIT ---")
    print(f"Normal μ={dist['normal']['mu']:.5f}, σ={dist['normal']['sigma']:.5f}")
    print(f"Student-t df={dist['student_t']['df']:.2f}")

    # 4. Monte Carlo
    paths = monte_carlo_forecast(
        last_price=df["Close"].iloc[-1],
        mu=dist["normal"]["mu"],
        sigma=dist["normal"]["sigma"],
        days=30,
        n_sims=3000
    )
    median_path = np.median(paths, axis=0)
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    print(f"\nMonte Carlo 30-day median forecast: {median_path[-1]:.2f}")
    print(f"5%-95% range: {p5[-1]:.2f} – {p95[-1]:.2f}")

    # 5. LSTM (אם זמין)
    model = create_lstm_model()
    if model is not None:
        X, y, scaler = prepare_lstm_data(df["Close"], lookback=30)
        split = int(len(X) * 0.85)
        model.fit(X[:split], y[:split], epochs=8, batch_size=32, verbose=0)
        pred_scaled = model.predict(X[split:], verbose=0)
        pred = scaler.inverse_transform(pred_scaled)
        actual = scaler.inverse_transform(y[split:].reshape(-1, 1))
        rmse = np.sqrt(mean_squared_error(actual, pred))
        print(f"\nLSTM RMSE on test set: {rmse:.2f}")

    # 6. חדשות
    print("\n--- ANALYST / NEWS SENTIMENT ---")
    sentiments = analyze_news_sentiment(EXAMPLE_NEWS)
    for s in sentiments:
        print(f"[{s['label']:8}] {s['score']:+.2f} | {s['text']}")

    # 7. גרפים
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # מחיר היסטורי
    axes[0,0].plot(df.index, df["Close"], color="navy", lw=1.5)
    axes[0,0].set_title("MSTR Historical Close")
    axes[0,0].grid(True, alpha=0.3)

    # Monte Carlo paths
    for i in range(min(80, paths.shape[0])):
        axes[0,1].plot(paths[i], color="steelblue", alpha=0.08)
    axes[0,1].plot(median_path, color="red", lw=2, label="Median")
    axes[0,1].plot(p5, color="orange", ls="--", label="5%")
    axes[0,1].plot(p95, color="orange", ls="--", label="95%")
    axes[0,1].set_title("Monte Carlo 30-day Forecast")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # התפלגות תשואות
    axes[1,0].hist(risk["returns"], bins=60, density=True, alpha=0.7, color="teal")
    x = np.linspace(risk["returns"].min(), risk["returns"].max(), 200)
    axes[1,0].plot(x, stats.norm.pdf(x, *stats.norm.fit(risk["returns"])), "r-", lw=2)
    axes[1,0].set_title("Daily Returns Distribution")
    axes[1,0].grid(True, alpha=0.3)

    # Drawdown
    dd = df["Close"] / df["Close"].cummax() - 1
    axes[1,1].fill_between(dd.index, dd, 0, color="crimson", alpha=0.4)
    axes[1,1].set_title("Drawdown")
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mstr_advanced_analysis.png", dpi=140)
    print("\nגרף נשמר: mstr_advanced_analysis.png")
    plt.show()

    # דוגמת מינוף
    print("\n--- EXAMPLE LEVERAGE POSITION ---")
    pos = leverage_position_size(capital=100_000, price=df["Close"].iloc[-1], leverage=2.5)
    for k, v in pos.items():
        print(f"{k:25}: {v:,.2f}")

# ===============================
# הרצה
# ===============================
if __name__ == "__main__":
    run_full_model()
