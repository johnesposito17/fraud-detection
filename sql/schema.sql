-- ============================================================
-- Core schema for the fraud detection transaction warehouse
-- Auto-executed by Postgres on first container start
-- ============================================================

CREATE TABLE IF NOT EXISTS transactions (
    id              SERIAL PRIMARY KEY,
    time_seconds    FLOAT       NOT NULL,   -- seconds since first transaction in dataset
    hour_of_day     SMALLINT    NOT NULL,   -- derived: time_seconds % 86400 // 3600
    day             SMALLINT    NOT NULL,   -- derived: time_seconds // 86400 (0 or 1, ~48hr window)
    amount          NUMERIC(12, 2) NOT NULL,
    class           SMALLINT    NOT NULL CHECK (class IN (0, 1)),  -- 0=legit, 1=fraud

    -- PCA-anonymized features (V1–V28)
    v1  FLOAT, v2  FLOAT, v3  FLOAT, v4  FLOAT,
    v5  FLOAT, v6  FLOAT, v7  FLOAT, v8  FLOAT,
    v9  FLOAT, v10 FLOAT, v11 FLOAT, v12 FLOAT,
    v13 FLOAT, v14 FLOAT, v15 FLOAT, v16 FLOAT,
    v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT,
    v21 FLOAT, v22 FLOAT, v23 FLOAT, v24 FLOAT,
    v25 FLOAT, v26 FLOAT, v27 FLOAT, v28 FLOAT
);

-- Indexes that support the SQL query library and dashboard queries
CREATE INDEX IF NOT EXISTS idx_transactions_class        ON transactions (class);
CREATE INDEX IF NOT EXISTS idx_transactions_hour         ON transactions (hour_of_day);
CREATE INDEX IF NOT EXISTS idx_transactions_amount       ON transactions (amount);
CREATE INDEX IF NOT EXISTS idx_transactions_time         ON transactions (time_seconds);
