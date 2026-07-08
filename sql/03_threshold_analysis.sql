-- ============================================================
-- 03_threshold_analysis.sql
-- Threshold tradeoff analysis — the core operational question:
-- "Where do we set the score cutoff to balance fraud caught
--  against false positives and review cost?"
-- All queries use the test split only.
-- Note: Postgres requires ::numeric cast for ROUND(float, n)
-- ============================================================


-- ── Q16: Manual threshold sweep (10 representative thresholds) ─
-- A quick precision/recall/cost table across the score range.
-- The Python dashboard generates the full 200-point sweep;
-- this query gives an analyst a fast view in any SQL client.
WITH thresholds AS (
    SELECT unnest(ARRAY[0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                        0.60, 0.70, 0.80, 0.90]) AS t
),
test_preds AS (
    SELECT class, fraud_prob FROM model_predictions WHERE split = 'test'
),
metrics AS (
    SELECT
        t,
        SUM(CASE WHEN class=1 AND fraud_prob >= t THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN class=0 AND fraud_prob >= t THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN class=1 AND fraud_prob <  t THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN class=0 AND fraud_prob <  t THEN 1 ELSE 0 END) AS tn
    FROM thresholds
    CROSS JOIN test_preds
    GROUP BY t
)
SELECT
    ROUND(t::numeric, 2)                                        AS threshold,
    tp, fp, fn, tn,
    ROUND(tp::numeric / NULLIF(tp + fp, 0) * 100, 1)           AS precision_pct,
    ROUND(tp::numeric / NULLIF(tp + fn, 0) * 100, 1)           AS recall_pct,
    (tp + fp)                                                   AS review_queue_size,
    fn * 500 + fp * 5                                           AS total_cost_dollars
FROM metrics
ORDER BY t;


-- ── Q17: Break-even threshold analysis ───────────────────────
-- At what threshold does tightening the cutoff stop saving money?
-- Computes cumulative TP/FP/FN as threshold rises — the minimum-cost
-- row is the operational sweet spot.
WITH score_buckets AS (
    SELECT
        ROUND(fraud_prob::numeric, 2) AS threshold
    FROM model_predictions
    WHERE split = 'test'
    GROUP BY ROUND(fraud_prob::numeric, 2)
),
metrics AS (
    SELECT
        sb.threshold,
        SUM(CASE WHEN mp.class=1 AND mp.fraud_prob >= sb.threshold THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN mp.class=0 AND mp.fraud_prob >= sb.threshold THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN mp.class=1 AND mp.fraud_prob <  sb.threshold THEN 1 ELSE 0 END) AS fn
    FROM score_buckets sb
    CROSS JOIN (SELECT class, fraud_prob FROM model_predictions WHERE split='test') mp
    GROUP BY sb.threshold
)
SELECT
    threshold,
    tp, fp, fn,
    fn * 500 + fp * 5                                           AS total_cost,
    ROUND(tp::numeric / NULLIF(tp + fn, 0) * 100, 1)           AS recall_pct,
    ROUND(tp::numeric / NULLIF(tp + fp, 0) * 100, 1)           AS precision_pct
FROM metrics
ORDER BY total_cost
LIMIT 10;


-- ── Q18: False positive rate by time of day ──────────────────
-- Are false alarms concentrated at specific hours? If so, a
-- time-conditioned threshold (tighter at night) could reduce FP
-- volume without hurting recall.
SELECT
    hour_of_day,
    COUNT(*)                                                    AS legit_flagged,
    ROUND(AVG(fraud_prob)::numeric, 4)                          AS avg_score,
    ROUND(
        COUNT(*)::numeric /
        NULLIF(SUM(COUNT(*)) OVER (), 0) * 100, 2
    )                                                           AS pct_of_all_fps
FROM model_predictions
WHERE split   = 'test'
  AND class   = 0
  AND pred_05 = 1
GROUP BY hour_of_day
ORDER BY legit_flagged DESC;


-- ── Q19: Review queue composition at optimal threshold ────────
-- If we send all flagged transactions to a human review queue,
-- what does that queue look like? Helps staff the right analysts.
SELECT
    amount_bucket,
    CASE WHEN class = 1 THEN 'Fraud' ELSE 'Legit' END          AS actual,
    COUNT(*)                                                    AS count,
    ROUND(AVG(fraud_prob)::numeric, 3)                          AS avg_score,
    ROUND(AVG(amount)::numeric, 2)                              AS avg_amount,
    ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100, 1)  AS pct_of_queue
FROM model_predictions
WHERE split        = 'test'
  AND pred_optimal = 1
GROUP BY amount_bucket, class
ORDER BY amount_bucket, class DESC;


-- ── Q20: Score percentile table ───────────────────────────────
-- What fraud_prob score cutoff corresponds to each percentile?
-- Useful for setting SLA-based review tiers:
-- "auto-block top 1%, expedited review 1-5%, normal queue rest."
WITH percentiled AS (
    SELECT
        class,
        fraud_prob,
        NTILE(100) OVER (ORDER BY fraud_prob DESC) AS percentile
    FROM model_predictions
    WHERE split = 'test'
),
per_bucket AS (
    SELECT
        percentile,
        MIN(fraud_prob)                             AS min_score,
        SUM(CASE WHEN class=1 THEN 1 ELSE 0 END)   AS fraud_in_bucket,
        SUM(CASE WHEN class=0 THEN 1 ELSE 0 END)   AS legit_in_bucket
    FROM percentiled
    GROUP BY percentile
)
SELECT
    percentile,
    ROUND(min_score::numeric, 4)                                AS min_score_at_percentile,
    fraud_in_bucket,
    legit_in_bucket,
    SUM(fraud_in_bucket) OVER (ORDER BY percentile)             AS cum_fraud_caught,
    SUM(legit_in_bucket) OVER (ORDER BY percentile)             AS cum_legit_flagged
FROM per_bucket
ORDER BY percentile
LIMIT 20;
