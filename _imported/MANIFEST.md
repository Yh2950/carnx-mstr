# _imported — MSTR / stock-analysis code gathered from the whole machine

Every file here is a **copy** (`cp -p`, mtime preserved). The originals were
**not moved or deleted** — the paths below still exist untouched. This folder is
a consolidation point only; nothing in it is wired into the CARN-X pipeline.

Scan date: 2026-08-29. Scanned: entire `/home/yedidyahkim` (excluding
`.venv`, `site-packages`, `node_modules`, IDE caches, `miniconda3`, `Trash`).

---

## order_book_calculator/ — the "share price from buyer/seller quotes" family

One idea in four iterations: enter each side's `(percent, price)` quotes, take a
percent-weighted average per side, multiply by a deterministic `market_factor`
sine wave, and call the midpoint the "close price". Plus a percent-change
calculator and hand-rolled probability functions.

| file | origin | date | sha256 (first 12) | notes |
|------|--------|------|-------------------|-------|
| `MSTR1.py` | `~/PycharmProjects/PythonProject/MSTR1.py` | 2025-12-21 | `252d2f4b133e` | **oldest** — OOP version (`StockMathModel` class) |
| `MSTR.py` | `~/PycharmProjects/PythonProject/MSTR.py` | 2026-02-13 | `295023ba8664` | procedural version + interactive CLI + direction calc |
| `mstr_model.py` | `~/PycharmProjects/PythonProject/mstr_model.py` | 2026-08-29 | `b838c5618cd1` | **merged + bug-fixed** (Claude): class + validated CLI + stable stats. Run: `python3 mstr_model.py --test` |
| `MSTR.sql` | `~/Coding/MSTR.sql` | 2025-11-27 | `666e5b5264a2` | the DB side — `stock_daily(trade_date, avg_ask, avg_sell, close_price)` schema + one INSERT |

A VS Code auto-history snapshot of `MSTR.py` also exists at
`~/.config/Code/User/History/-72e3bbc9/Z2oz.py` (byte-identical to an older
`MSTR.py`) — not copied, it is just an editor backup.

## advanced_model_draft/ — an earlier full predictive-model attempt

| file | origin | date | sha256 (first 12) | notes |
|------|--------|------|-------------------|-------|
| `mstr_advanced_leverage_model.py` | `~/Desktop/פרויקטים/MSTR/Mstr1.py` | 2026-07-31 | `feeb3bf43ff7` | "MSTR Advanced Predictive & Leverage Model": yfinance data (synthetic fallback), leverage/VaR/Sharpe metrics, Monte-Carlo forecast, optional Keras LSTM, keyword news-sentiment, matplotlib dashboard. Predates CARN-X; superseded by it. Renamed from `Mstr1.py` to avoid the name clash with the order-book `MSTR1.py`. An identical copy also sits in `~/.local/share/Trash/`. |

---

## Deliberately EXCLUDED (looked related, is not — nuance)

| path | why excluded |
|------|--------------|
| `~/.local/share/Trash/files/promptmstr.txt` | filename says "mstr" but the content is a physics / F-35 fighter-jet prompt (belongs to `~/Desktop/פרויקטים/דשבורד מטוס קרב F35`) |
| `~/Desktop/File/SQL_Projects/temp.sql` | an e-commerce shop schema — "stock" here means **warehouse inventory**, not shares |
| `~/IdeaProjects/mstr/` | project named "mstr" but contains only "Hello, World!" Java scaffolding (`App.java`, `MSTR1`) |
| `~/PycharmProjects/PythonProject/sqlite.py`, `Matala5.py` | generic sqlite / string-exercise tutorials |
| `~/Desktop/File/*.java` | Java data-structures coursework (DHeap, HashTable, sorting) |
| IDE type stubs (`stock_chart.pyi`, `matplotlib/ticker.pyi`), vendored JS, `.claude/` files | not user code |

## Relationship to CARN-X (the parent folder)

`../` (the `תחזיות מתמטיות` folder) already **is** the real, current MSTR
forecasting system. Nothing here needs to be integrated into it:

- `order_book_calculator/` — a standalone quote-midpoint calculator + math toy.
  Keep as a utility; it is not a forecasting model.
- `advanced_model_draft/` — every idea in it (historical data, Student-t fit,
  Monte-Carlo, LSTM, leverage/VaR, sentiment) is already implemented more
  rigorously in `data_layer.py` / `targets.py` / `forecast_net.py` /
  `walk_forward.py` / `leverage_factorization.py`.
