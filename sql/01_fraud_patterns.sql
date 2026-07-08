-- ============================================================
-- 01_fraud_patterns.sql
-- Fraud pattern analysis against transaction_features
-- All queries run against the full dataset (no train/test split)
-- Note: Postgres requires ::numeric cast for ROUND(float, n)
-- ============================================================


-- ── Q1: Overall fraud summary ────────────────────────────────
-- Baseline numbers every fraud analyst keeps on hand.
SELECT
    COUNT(*)                                                        AS total_transactions,
    SUM(class)                                                      AS total_fraud,
    SUM(CASE WHEN class = 0 THEN 1 END)                             AS total_legit,
    ROUND(AVG(class)::numeric * 100, 4)                             AS fraud_rate_pct,
    ROUND(AVG(CASE WHEN class = 1 THEN amount END)::numeric, 2)     AS avg_fraud_amount,
    ROUND(AVG(CASE WHEN class = 0 THEN amount END)::numeric, 2)     AS avg_legit_amount,
    ROUND(SUM(CASE WHEN class = 1 THEN amount END)::numeric, 2)     AS total_fraud_dollars
FROM transaction_features;


-- ── Q2: Fraud rate by hour of day ───────────────────────────
-- Identifies peak fraud windows for dynamic threshold scheduling.
-- A fraud ops team would use this to tighten review queues at night.
SELECT
    hour_of_day,
    COUNT(*)                                                        AS total_txns,
    SUM(class)                                                      AS fraud_count,
    ROUND(AVG(class)::numeric * 100, 3)                             AS fraud_rate_pct,
    ROUND(AVG(amount)::numeric, 2)                                  AS avg_amount,
    -- 3-hour rolling average fraud rate to smooth noise
    ROUND(AVG(AVG(class)::numeric * 100) OVER (
        ORDER BY hour_of_day
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ), 3)                                                           AS fraud_rate_3hr_rolling_avg
FROM transaction_features
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- ── Q3: Fraud rate by amount bucket ─────────────────────────
-- Micro transactions (<$10) show disproportionate fraud — consistent
-- with test-charge behavior before larger fraudulent withdrawals.
SELECT
    amount_bucket,
    COUNT(*)                                                        AS total_txns,
    SUM(class)                                                      AS fraud_count,
    ROUND(AVG(class)::numeric * 100, 3)                             AS fraud_rate_pct,
    ROUND(AVG(amount)::numeric, 2)                                  AS avg_amount,
    ROUND(MIN(amount)::numeric, 2)                                  AS min_amount,
    ROUND(MAX(amount)::numeric, 2)                                  AS max_amount,
    ROUND(SUM(CASE WHEN class=1 THEN amount ELSE 0 END)::numeric, 2) AS total_fraud_dollars
FROM transaction_features
GROUP BY amount_bucket
ORDER BY
    CASE amount_bucket
        WHEN 'micro'  THEN 1
        WHEN 'small'  THEN 2
        WHEN 'medium' THEN 3
        WHEN 'large'  THEN 4
    END;


-- ── Q4: Night vs. day fraud breakdown ───────────────────────
-- EDA showed fraud rate 10× higher at night. This query validates
-- it at the segment level and quantifies the dollar impact.
SELECT
    CASE WHEN is_night = 1 THEN 'Night (00:00-05:59)'
         ELSE                   'Day   (06:00-23:59)' END          AS time_segment,
    COUNT(*)                                                        AS total_txns,
    SUM(class)                                                      AS fraud_count,
    ROUND(AVG(class)::numeric * 100, 3)                             AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN class=1 THEN amount ELSE 0 END)::numeric, 2) AS fraud_dollars,
    ROUND(AVG(CASE WHEN class=1 THEN amount END)::numeric, 2)       AS avg_fraud_amount
FROM transaction_features
GROUP BY is_night
ORDER BY is_night DESC;


-- ── Q5: Fraud by amount decile ───────────────────────────────
-- Decile breakdown reveals where fraud concentrates in the amount
-- distribution. Analysts use this to build rule-based tiers.
SELECT
    amount_decile,
    COUNT(*)                                                        AS total_txns,
    SUM(class)                                                      AS fraud_count,
    ROUND(AVG(class)::numeric * 100, 3)                             AS fraud_rate_pct,
    ROUND(MIN(amount)::numeric, 2)                                  AS decile_min,
    ROUND(MAX(amount)::numeric, 2)                                  AS decile_max,
    ROUND(AVG(amount)::numeric, 2)                                  AS avg_amount
FROM (
    SELECT
        *,
        NTILE(10) OVER (ORDER BY amount) AS amount_decile
    FROM transaction_features
) bucketed
GROUP BY amount_decile
ORDER BY amount_decile;


-- ── Q6: Rolling hourly fraud count over dataset timeline ─────
-- Shows how fraud bursts cluster in time. A spike in rolling count
-- is an early-warning signal for coordinated attacks.
SELECT
    ROUND(time_seconds / 3600)::int                                 AS hour_number,
    COUNT(*)                                                        AS total_txns,
    SUM(class)                                                      AS fraud_count,
    ROUND(AVG(class)::numeric * 100, 3)                             AS fraud_rate_pct,
    -- Rolling 2-hour fraud count
    SUM(SUM(class)) OVER (
        ORDER BY ROUND(time_seconds / 3600)
        ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
    )                                                               AS rolling_2hr_fraud_count
FROM transaction_features
GROUP BY ROUND(time_seconds / 3600)
ORDER BY hour_number;


-- ── Q7: Percentile anomaly flags ────────────────────────────
-- Flags transactions in the top 5% of amount AND at night.
-- A simple rule-based anomaly detector — no model needed.
-- Used to benchmark model performance against naive rules.
SELECT
    COUNT(*)                                                        AS flagged_transactions,
    SUM(class)                                                      AS fraud_caught,
    ROUND(AVG(class)::numeric * 100, 2)                             AS precision_pct,
    ROUND(
        SUM(class)::numeric /
        NULLIF((SELECT SUM(class) FROM transaction_features), 0) * 100,
        2
    )                                                               AS recall_pct
FROM (
    SELECT *, PERCENT_RANK() OVER (ORDER BY amount) AS amt_rank
    FROM transaction_features
) ranked
WHERE amt_rank >= 0.95
  AND is_night = 1;


-- ── Q8: Fraud velocity — transactions clustered in time ──────
-- Identifies minutes where fraud density is unusually high.
-- Groups transactions into 10-minute bins and ranks by fraud rate.
SELECT
    FLOOR(time_seconds / 600)::int                                  AS bin_10min,
    COUNT(*)                                                        AS total_txns,
    SUM(class)                                                      AS fraud_count,
    ROUND(AVG(class)::numeric * 100, 2)                             AS fraud_rate_pct,
    ROUND(AVG(amount)::numeric, 2)                                  AS avg_amount
FROM transaction_features
GROUP BY FLOOR(time_seconds / 600)
HAVING SUM(class) >= 2
ORDER BY fraud_rate_pct DESC
LIMIT 20;
