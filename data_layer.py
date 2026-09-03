"""
CARN-X  --  Data Layer  (real market data)
=========================================
Historical data ingestion for the MSTR forecasting model. Pulls daily OHLCV
for MSTR and every context asset from Yahoo Finance (yfinance), aligns
everything to the MSTR trading calendar, forward-fills the 24/7 crypto series
onto trading days, and caches the assembled panel to parquet.

Everything downstream (features.py, targets.py, walk_forward.py) consumes the
single DataFrame returned by ``build_panel`` -- indexed by trading date, one
column block per asset.

No look-ahead is introduced here: every row holds only values observable at
that day's close. Forward returns / targets are built separately in targets.py.
"""

# ruff: noqa: I001  -- import order is load-bearing: `import envcheck` (the
#                    interpreter guard) must run before torch / pandas are imported.
from __future__ import annotations

import envcheck  # noqa: F401  (must precede heavy imports)

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf

    HAS_YF = True
except ImportError:  # pragma: no cover
    HAS_YF = False

try:
    from urllib.parse import quote
    from urllib.request import Request, urlopen

    HAS_URLLIB = True
except ImportError:  # pragma: no cover
    HAS_URLLIB = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# asset key -> Yahoo ticker.  The primary asset MUST be first.
DEFAULT_TICKERS: dict[str, str] = {
    "mstr": "MSTR",  # primary -- MicroStrategy / Strategy
    "btc": "BTC-USD",  # the balance-sheet driver
    "ibit": "IBIT",  # spot-BTC ETF (post 2024-01) -- clean BTC proxy
    "mstu": "MSTU",  # 2x long MSTR ETF (post 2024-08) -- leveraged sentiment
    "mstz": "MSTZ",  # -2x MSTR ETF -- short sentiment
    "gld": "GLD",  # gold -- alternative store of value
    "vix": "^VIX",  # equity implied vol
    "tnx": "^TNX",  # 10y UST yield (x10)
    "dxy": "DX-Y.NYB",  # US dollar index
    "spx": "^GSPC",  # S&P 500
    "ndx": "^IXIC",  # Nasdaq Composite
    "qqq": "QQQ",  # Nasdaq-100 ETF (tradable, dividend-adjusted)
}

# assets that trade 24/7 (crypto) -- forward-filled onto trading days
CRYPTO_KEYS = frozenset({"btc"})

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")

# Yahoo's public chart endpoint. The `yfinance` library's crumb/cookie dance
# is what trips over cloud IPs -- this raw endpoint with a browser UA is far
# more robust, so it is the fallback feed for a manual "refresh" on the
# deployed app.
_YF_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"


@dataclass
class DataConfig:
    tickers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TICKERS))
    start: str = "2019-06-01"  # a year of pre-BTC-treasury context
    end: str | None = None  # None -> today
    cache_dir: str = "./data_cache"
    max_staleness_hours: float = 6.0  # refetch if the cache is older than this
    min_history_rows: int = 250  # a ticker with fewer rows is dropped
    ffill_limit: int = 3  # max consecutive trading days to carry a stale price
    patch_with_intraday: bool = True  # backfill missing recent daily bars from 1h data
    retries: int = 4
    retry_backoff_sec: float = 2.0
    primary_key: str = "mstr"


# ---------------------------------------------------------------------------
# trading-session helpers
# ---------------------------------------------------------------------------

_NY = "America/New_York"


def _now_ny() -> pd.Timestamp:
    return pd.Timestamp.now(tz=_NY)


def expected_last_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Most recent *completed* US equity session (naive date).

    Holiday-agnostic: a holiday just means the patch step finds no data for that
    date and moves on. The regular close is 16:00 ET; we wait until 16:20.
    """
    now = now or _now_ny()
    d = now.normalize().tz_localize(None)
    session_done_today = now.weekday() < 5 and (now.hour, now.minute) >= (16, 20)
    if not session_done_today:
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:  # roll back over the weekend
        d -= pd.Timedelta(days=1)
    return d


def _intraday_to_daily(ticker: str, days: int = 12) -> pd.DataFrame:
    """1h bars -> daily OHLCV, used only to backfill the last day or two that
    Yahoo's daily endpoint publishes late (or as a NaN-price placeholder)."""
    try:
        h = yf.Ticker(ticker).history(period=f"{days}d", interval="1h", auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if h is None or len(h) == 0 or "Close" not in h.columns:
        return pd.DataFrame()
    h = h.rename(columns=str.lower)
    idx = pd.to_datetime(h.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(_NY).tz_localize(None)
    h.index = idx
    h = h[np.isfinite(h["close"]) & (h["close"] > 0)]
    if h.empty:
        return pd.DataFrame()
    g = h.groupby(h.index.normalize())
    daily = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(),
        }
    )
    return daily


# ---------------------------------------------------------------------------
# Low-level fetch
# ---------------------------------------------------------------------------


def _download_yahoo_direct(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Daily OHLCV straight from Yahoo's chart JSON endpoint (browser UA, no
    library) -- the fallback when `yfinance` fails, e.g. from a cloud host.
    Returns an empty frame on any failure. Close is split+dividend adjusted."""
    if not HAS_URLLIB:
        return pd.DataFrame()
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end or _now_ny().normalize().tz_localize(None)).timestamp()) + 86400
    url = (
        _YF_CHART.format(sym=quote(ticker))
        + f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urlopen(req, timeout=20) as resp:  # noqa: S310 -- fixed https host
            j = json.loads(resp.read().decode())
        res = j["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    except Exception:  # noqa: BLE001 -- any shape / network error -> no data
        return pd.DataFrame()
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(_NY).tz_localize(None).normalize()
    raw_close = pd.Series(q.get("close"), index=idx, dtype="float64")
    close = pd.Series(adj if adj is not None else q.get("close"), index=idx, dtype="float64")
    factor = (close / raw_close).where(lambda s: (s > 0) & np.isfinite(s), 1.0)
    df = pd.DataFrame(
        {
            "open": pd.Series(q["open"], index=idx, dtype="float64") * factor,
            "high": pd.Series(q["high"], index=idx, dtype="float64") * factor,
            "low": pd.Series(q["low"], index=idx, dtype="float64") * factor,
            "close": close,
            "volume": pd.Series(q["volume"], index=idx, dtype="float64"),
        }
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]
    last_session = expected_last_session()
    return df[df.index <= last_session]


def _download_one(
    ticker: str,
    start: str,
    end: str | None,
    retries: int,
    backoff: float,
    patch_intraday: bool = True,
) -> pd.DataFrame:
    """Single-ticker daily OHLCV from Yahoo, auto-adjusted, with retry, and --
    critically -- backfilled from 1h bars for the most recent session(s) that
    Yahoo's daily endpoint has not yet published or has left as a NaN row."""
    if not HAS_YF:
        raise RuntimeError("yfinance is not installed -- `pip install yfinance`")

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is not None and len(raw) > 0:
                break
            last_err = RuntimeError(f"empty frame for {ticker}")
        except Exception as e:  # noqa: BLE001 -- yfinance raises a zoo of errors
            last_err = e
        time.sleep(backoff * (attempt + 1))
    else:
        # Yahoo exhausted -- try the Stooq fallback before giving up
        alt = _download_yahoo_direct(ticker, start, end)
        if len(alt):
            print(f"[data_layer] {ticker}: yfinance failed, served {len(alt)} rows via chart API")
            return alt
        raise RuntimeError(f"failed to download {ticker} after {retries} tries: {last_err}")

    # yfinance may return a MultiIndex (field, ticker) column frame
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw = raw[[c for c in OHLCV_FIELDS if c in raw.columns]].copy()

    # robust tz handling: yfinance returns tz-aware on some versions, naive on others
    idx = pd.to_datetime(raw.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    raw.index = idx.normalize()
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    # keep only completed sessions; drop Yahoo's NaN-price placeholder rows
    last_session = expected_last_session()
    raw = raw[raw.index <= last_session]
    raw = raw[raw["close"].notna() & (raw["close"] > 0)]

    # backfill the tail from 1h data when the daily feed lags the last session
    if patch_intraday and (len(raw) == 0 or raw.index.max() < last_session):
        intra = _intraday_to_daily(ticker)
        if len(intra):
            have_max = raw.index.max() if len(raw) else pd.Timestamp.min
            add = intra[(intra.index > have_max) & (intra.index <= last_session)]
            add = add[add["close"].notna() & (add["close"] > 0)]
            if len(add):
                cols = [c for c in OHLCV_FIELDS if c in raw.columns] or list(OHLCV_FIELDS)
                raw = pd.concat([raw, add.reindex(columns=cols)]).sort_index()
                raw = raw[~raw.index.duplicated(keep="last")]
                print(
                    f"[data_layer] {ticker}: backfilled {len(add)} session(s) "
                    f"from 1h data (through {add.index.max().date()})"
                )
    return raw


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_paths(cache_dir: str) -> tuple[Path, Path]:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / "panel.parquet", d / "panel_meta.json"


def _config_fingerprint(cfg: DataConfig) -> str:
    import hashlib

    payload = json.dumps(
        {
            "start": cfg.start,
            "end": cfg.end,
            "tickers": cfg.tickers,
            "ffill_limit": cfg.ffill_limit,
            "primary_key": cfg.primary_key,
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _cache_is_fresh(meta_path: Path, cfg: DataConfig) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    if meta.get("fingerprint") != _config_fingerprint(cfg):
        return False
    # invalidate if a newer trading session has completed since the last fetch
    try:
        if pd.Timestamp(meta.get("date_max")) < expected_last_session():
            return False
    except Exception:
        return False
    age_h = (time.time() - meta.get("fetched_at", 0)) / 3600.0
    return age_h < cfg.max_staleness_hours


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------


def _assemble_panel(frames: dict[str, pd.DataFrame], cfg: DataConfig) -> pd.DataFrame:
    """Align every asset to the primary asset's trading calendar.

    Bounded forward-fill only: a short market-holiday / data-gap (<= ffill_limit
    trading days) is carried forward; a longer outage is left NaN so features.py
    raises a missingness flag instead of silently feeding stale prices. This is
    the key data-sync guard -- crypto included (BTC-USD has a bar every calendar
    day, so any gap on a trading day is a real Yahoo outage, not a weekend).
    """
    if cfg.primary_key not in frames:
        raise RuntimeError(f"primary asset '{cfg.primary_key}' missing from downloaded data")

    calendar = frames[cfg.primary_key].index
    out = pd.DataFrame(index=calendar)

    for key, df in frames.items():
        block = df.reindex(calendar)
        raw_close_na = (
            block["close"].isna() if "close" in block else pd.Series(False, index=calendar)
        )
        block = block.ffill(limit=cfg.ffill_limit)
        block.columns = [f"{key}_{c}" for c in block.columns]
        # explicit "this row is forward-filled / stale" indicator per asset
        out[f"{key}_ffilled"] = (raw_close_na & block[f"{key}_close"].notna()).astype(float)
        out = out.join(block)

    # index yields like ^TNX come in as "x10 percent"; keep raw, features.py handles it
    out.index.name = "date"
    # drop the leading rows where the primary asset itself has no close
    out = out[out[f"{cfg.primary_key}_close"].notna()]
    return out


def assert_panel_integrity(panel: pd.DataFrame, cfg: DataConfig) -> None:
    """Fail loudly on any date-alignment / sync defect."""
    idx = panel.index
    assert idx.is_monotonic_increasing, "panel index is not sorted ascending"
    assert idx.is_unique, "panel index has duplicate dates"
    last_session = expected_last_session()
    assert idx.max() <= last_session, (
        f"panel's last row {idx.max().date()} is ahead of the last completed "
        f"session {last_session.date()}"
    )
    pk = cfg.primary_key
    assert panel[f"{pk}_close"].notna().all(), "primary asset has NaN closes after assembly"
    assert (panel[f"{pk}_close"] > 0).all(), "primary asset has non-positive closes"
    # weekday-only calendar (the primary is an equity)
    assert (idx.dayofweek < 5).all(), "primary trading calendar contains weekend dates"
    # no absurd single-day jumps that would signal an unadjusted split
    r = np.log(panel[f"{pk}_close"]).diff().dropna()
    assert r.abs().max() < 1.5, (
        f"implausible {pk} 1-day move ({r.abs().max():.2f}) -- check split adjustment"
    )


def build_panel(cfg: DataConfig | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Return the assembled, cached multi-asset daily panel.

    Columns: ``{asset}_{open,high,low,close,volume}`` for every asset that had
    at least ``cfg.min_history_rows`` rows. Indexed by the primary asset's
    trading dates.
    """
    cfg = cfg or DataConfig()
    panel_path, meta_path = _cache_paths(cfg.cache_dir)

    if not force_refresh and _cache_is_fresh(meta_path, cfg) and panel_path.exists():
        cached = pd.read_parquet(panel_path)
        assert_panel_integrity(cached, cfg)  # re-validate on every load
        return cached

    # a non-primary asset that neither feed can serve is recovered from the
    # previous cache (frozen + ffilled) rather than dropped -- features.py
    # depends on every macro column being present.
    prev_cache = pd.read_parquet(panel_path) if panel_path.exists() else None

    def _from_prev_cache(key: str) -> pd.DataFrame | None:
        if prev_cache is None:
            return None
        cols = {f"{key}_{c}": c for c in OHLCV_FIELDS if f"{key}_{c}" in prev_cache.columns}
        if "close" not in cols.values():
            return None
        return prev_cache[list(cols)].rename(columns=cols).dropna(how="all")

    frames: dict[str, pd.DataFrame] = {}
    dropped: list[str] = []
    frozen: list[str] = []
    for key, ticker in cfg.tickers.items():
        try:
            df = _download_one(
                ticker,
                cfg.start,
                cfg.end,
                cfg.retries,
                cfg.retry_backoff_sec,
                patch_intraday=cfg.patch_with_intraday,
            )
        except Exception as e:  # noqa: BLE001
            if key == cfg.primary_key:
                raise
            df = _from_prev_cache(key)
            if df is None or len(df) < cfg.min_history_rows:
                dropped.append(f"{key} ({ticker}): {e}")
                continue
            frozen.append(key)
        if len(df) < cfg.min_history_rows:
            if key == cfg.primary_key:
                raise RuntimeError(f"primary asset {ticker} has only {len(df)} rows")
            fb = _from_prev_cache(key)
            if fb is not None and len(fb) >= cfg.min_history_rows:
                df = fb
                frozen.append(key)
            else:
                dropped.append(f"{key} ({ticker}): only {len(df)} rows")
                continue
        frames[key] = df
    if frozen:
        print(f"[data_layer] frozen from previous cache (no fresh feed): {frozen}")

    panel = _assemble_panel(frames, cfg)
    assert_panel_integrity(panel, cfg)

    # per-asset lag between its last real observation and the panel end
    cal_end = panel.index.max()
    staleness = {key: int((cal_end - frames[key].index.max()).days) for key in frames}
    stale_assets = {k: v for k, v in staleness.items() if v > 4}

    panel.to_parquet(panel_path)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": _config_fingerprint(cfg),
                "start": cfg.start,
                "end": cfg.end,
                "tickers": cfg.tickers,
                "ffill_limit": cfg.ffill_limit,
                "fetched_at": time.time(),
                "fetched_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "rows": int(len(panel)),
                "columns": list(panel.columns),
                "assets_kept": list(frames.keys()),
                "assets_dropped": dropped,
                "asset_staleness_days": staleness,
                "date_min": str(panel.index.min().date()),
                "date_max": str(panel.index.max().date()),
            },
            indent=2,
        )
    )

    if dropped:
        print(f"[data_layer] dropped {len(dropped)} asset(s):")
        for d in dropped:
            print(f"  - {d}")
    if stale_assets:
        print(f"[data_layer] WARNING stale assets (>4d behind panel end): {stale_assets}")
    return panel


# US equity market holidays (observed) -- for the live market-status pill only.
_MKT_HOLIDAYS = frozenset(
    np.array(
        [
            "2026-01-01",
            "2026-01-19",
            "2026-02-16",
            "2026-04-03",
            "2026-05-25",
            "2026-06-19",
            "2026-07-03",
            "2026-09-07",
            "2026-11-26",
            "2026-12-25",
            "2027-01-01",
            "2027-01-18",
            "2027-02-15",
            "2027-03-26",
            "2027-05-31",
            "2027-06-18",
            "2027-07-05",
            "2027-09-06",
            "2027-11-25",
            "2027-12-24",
            "2028-01-17",
            "2028-02-21",
            "2028-04-14",
            "2028-05-29",
            "2028-06-19",
            "2028-07-04",
            "2028-09-04",
            "2028-11-23",
            "2028-12-25",
        ],
        dtype="datetime64[D]",
    ).tolist()
)


def market_status(ticker: str = "MSTR", now: pd.Timestamp | None = None) -> str:
    """Live session state: ``regular`` | ``pre`` | ``post`` | ``closed`` | ``holiday``.
    Crypto (``*-USD``) trades 24/7 -> always ``regular``."""
    if ticker.upper().endswith("-USD"):
        return "regular"
    now = now or _now_ny()
    if now.weekday() >= 5:
        return "closed"
    if np.datetime64(now.date(), "D") in _MKT_HOLIDAYS:
        return "holiday"
    mins = now.hour * 60 + now.minute
    if 570 <= mins < 960:  # 09:30 - 16:00 ET
        return "regular"
    if 240 <= mins < 570:  # 04:00 - 09:30 ET
        return "pre"
    if 960 <= mins < 1200:  # 16:00 - 20:00 ET
        return "post"
    return "closed"


@dataclass
class LiveQuote:
    ticker: str
    price: float | None
    prev_close: float | None
    change_pct: float | None
    as_of: str
    source: str
    day_open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    year_high: float | None = None
    year_low: float | None = None
    market_state: str = "unknown"
    epoch: float = 0.0  # time.time() at fetch (freshness clock)


def live_quote(ticker: str = "MSTR") -> LiveQuote:
    """Best-effort *real-time* quote, separate from the daily panel. For display and
    the live ticker -- the model always runs on completed daily bars.
    """
    price = prev = None
    d_open = d_high = d_low = vol = y_high = y_low = None
    src = "unavailable"
    try:
        fi = yf.Ticker(ticker).fast_info
        price = fi.get("lastPrice") or fi.get("last_price")
        prev = (
            fi.get("previousClose")
            or fi.get("previous_close")
            or fi.get("regularMarketPreviousClose")
        )
        d_open = fi.get("open")
        d_high = fi.get("dayHigh") or fi.get("day_high")
        d_low = fi.get("dayLow") or fi.get("day_low")
        vol = fi.get("lastVolume") or fi.get("last_volume")
        y_high = fi.get("yearHigh") or fi.get("year_high")
        y_low = fi.get("yearLow") or fi.get("year_low")
        src = "yahoo fast_info"
    except Exception:
        pass
    if price is None:  # fall back to the last 1h bar
        intra = _intraday_to_daily(ticker, days=5)
        if len(intra):
            price = float(intra["close"].iloc[-1])
            prev = float(intra["close"].iloc[-2]) if len(intra) > 1 else None
            d_high = float(intra["high"].iloc[-1]) if "high" in intra else None
            d_low = float(intra["low"].iloc[-1]) if "low" in intra else None
            src = "yahoo 1h bar"

    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    price, prev = _f(price), _f(prev)
    chg = ((price / prev - 1.0) * 100.0) if (price and prev) else None
    return LiveQuote(
        ticker=ticker,
        price=price,
        prev_close=prev,
        change_pct=chg,
        as_of=_now_ny().strftime("%Y-%m-%d %H:%M %Z"),
        source=src,
        day_open=_f(d_open),
        day_high=_f(d_high),
        day_low=_f(d_low),
        volume=_f(vol),
        year_high=_f(y_high),
        year_low=_f(y_low),
        market_state=market_status(ticker),
        epoch=time.time(),
    )


def live_quotes(tickers=("MSTR", "BTC-USD")) -> dict[str, LiveQuote]:
    """Batch live quotes, one Yahoo call per ticker. Cache this in the app layer."""
    return {t: live_quote(t) for t in tickers}


def panel_freshness(panel: pd.DataFrame) -> dict[str, object]:
    """How current is the panel vs the last completed session."""
    last_row = panel.index.max()
    last_session = expected_last_session()
    lag_sessions = int(np.busday_count(last_row.date(), last_session.date()))
    return {
        "panel_last_date": str(last_row.date()),
        "expected_last_session": str(last_session.date()),
        "sessions_behind": lag_sessions,
        "is_current": lag_sessions <= 0,
    }


def primary_close(panel: pd.DataFrame, cfg: DataConfig | None = None) -> pd.Series:
    cfg = cfg or DataConfig()
    return panel[f"{cfg.primary_key}_close"].dropna()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("CARN-X Data Layer -- Self Test")
    print("=" * 70)

    cfg = DataConfig()
    panel = build_panel(cfg)

    print(f"\npanel shape : {panel.shape}")
    print(f"date range  : {panel.index.min().date()} -> {panel.index.max().date()}")
    print(f"assets      : {sorted({c.split('_')[0] for c in panel.columns})}")

    close = primary_close(panel, cfg)
    rets = np.log(close).diff().dropna()
    print(
        f"\nMSTR daily log-return  mean={rets.mean():+.5f}  std={rets.std():.5f}  "
        f"ann.vol={rets.std() * np.sqrt(252):.1%}"
    )
    print(f"MSTR worst day         {rets.min():+.2%}   best day {rets.max():+.2%}")

    na_frac = panel.isna().mean().sort_values(ascending=False)
    print("\ncolumns with most missing values:")
    print(na_frac.head(8).to_string())

    print("\n" + "=" * 70)
    print("Data Layer ready.")
    print("=" * 70)
