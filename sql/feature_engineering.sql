-- ============================================================
-- Feature Engineering Reference SQL
-- ============================================================
-- This file documents the feature definitions as SQL expressions.
-- Actual computation runs in src/features/build_features.py using
-- pandas rolling(), which handles 284k rows in seconds.
--
-- Rolling velocity features (txn_count_last_1hr, etc.) require
-- partitioning by card_id for production use. The Kaggle dataset
-- has no card identifier (V1-V28 are PCA-anonymized), so these
-- are computed over the global transaction stream as a proxy.
-- In production: add PARTITION BY card_id to each window function.
-- ============================================================

-- Amount features
-- amount_log        = LN(amount + 1)
-- amount_zscore     = (amount - global_mean) / global_std
-- amount_percentile = PERCENT_RANK() OVER (ORDER BY amount)
-- amount_bucket     = CASE WHEN amount < 10   THEN 'micro'
--                          WHEN amount < 100  THEN 'small'
--                          WHEN amount < 1000 THEN 'medium'
--                          ELSE 'large' END

-- Time features
-- is_night          = CASE WHEN hour_of_day BETWEEN 0 AND 5  THEN 1 ELSE 0 END
-- is_business_hours = CASE WHEN hour_of_day BETWEEN 9 AND 17 THEN 1 ELSE 0 END
-- seconds_since_last_txn = time_seconds - LAG(time_seconds) OVER (ORDER BY time_seconds)

-- Velocity features (global stream; production: add PARTITION BY card_id)
-- txn_count_last_1hr  = COUNT(*) OVER (ORDER BY time_seconds RANGE BETWEEN 3600  PRECEDING AND CURRENT ROW) - 1
-- txn_count_last_24hr = COUNT(*) OVER (ORDER BY time_seconds RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW) - 1
-- amount_sum_last_1hr = SUM(amount) OVER (ORDER BY time_seconds RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW) - amount
-- avg_amount_last_1hr = AVG(amount) OVER (ORDER BY time_seconds RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW)
-- amount_vs_1hr_avg   = amount / NULLIF(avg_amount_last_1hr, 0)

-- View the materialized results:
SELECT
    id,
    time_seconds,
    hour_of_day,
    amount,
    amount_log,
    amount_zscore,
    amount_bucket,
    is_night,
    seconds_since_last_txn,
    txn_count_last_1hr,
    txn_count_last_24hr,
    amount_sum_last_1hr,
    avg_amount_last_1hr,
    amount_vs_1hr_avg,
    class
FROM transaction_features
LIMIT 10;
