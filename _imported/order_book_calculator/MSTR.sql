-- 1. יצירת טבלה
CREATE TABLE IF NOT EXISTS stock_daily (
    trade_date DATE PRIMARY KEY,
    avg_ask REAL,
    avg_sell REAL,
    close_price REAL
);

-- 2. הכנסה של נתון חדש (ליום מסחר מסוים)
INSERT INTO stock_daily (trade_date, avg_ask, avg_sell, close_price)
VALUES ('2025-11-27', 10.5, 11.2, 10.85);
