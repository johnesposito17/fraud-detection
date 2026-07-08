-- ============================================================
-- 02_model_evaluation.sql
-- Model performance queries against model_predictions table
-- All evaluation queries filter to split = 'test' to avoid
-- assessing the model on data it was trained on.
-- Note: Postgres requires ::numeric cast for ROUND(float, n)
-- ============================================================


-- ── Q9: Confusion matrix at default threshold (0.50) ────────
-- The four cells every fraud analyst needs to interpret a model:
-- TP (caught fraud), FP (false alarms), FN (missed fraud), TN (correct clears)
SELECT
    SUM(CASE WHEN class=1 AND pred_05=1 THEN 1 ELSE 0 END)  AS true_positives,
    SUM(CASE WHEN class=0 AND pred_05=1 THEN 1 ELSE 0 END)  AS false_positives,
    SUM(CASE WHEN class=1 AND pred_05=0 THEN 1 ELSE 0 END)  AS false_negatives,
    SUM(CASE WHEN class=0 AND pred_05=0 THEN 1 ELSE 0 END)  AS true_negatives,
    ROUND(
        SUM(CASE WHEN class=1 AND pred_05=1 THEN 1 ELSE 0 END)::numeric /
        NULLIF(SUM(CASE WHEN pred_05=1 THEN 1 ELSE 0 END), 0) * 100, 2
    )                                                        AS precision_pct,
    ROUND(
        SUM(CASE WHEN class=1 AND pred_05=1 THEN 1 ELSE 0 END)::numeric /
        NULLIF(SUM(class), 0) * 100, 2
    )                                                        AS recall_pct,
    ROUND(
        SUM(CASE WHEN class=0 AND pred_05=1 THEN 1 ELSE 0 END)::numeric /
        NULLIF(SUM(CASE WHEN class=0 THEN 1 ELSE 0 END), 0) * 100, 4
    )                                                        AS false_positive_rate_pct
FROM model_predictions
WHERE split = 'test';


-- ── Q10: Confusion matrix at optimal threshold ───────────────
-- Same as Q9 but at the F1-optimal threshold (0.458 for RF).
-- Shows how threshold choice shifts the precision/recall balance.
SELECT
    SUM(CASE WHEN class=1 AND pred_optimal=1 THEN 1 ELSE 0 END) AS true_positives,
    SUM(CASE WHEN class=0 AND pred_optimal=1 THEN 1 ELSE 0 END) AS false_positives,
    SUM(CASE WHEN class=1 AND pred_optimal=0 THEN 1 ELSE 0 END) AS false_negatives,
    SUM(CASE WHEN class=0 AND pred_optimal=0 THEN 1 ELSE 0 END) AS true_negatives,
    ROUND(
        SUM(CASE WHEN class=1 AND pred_optimal=1 THEN 1 ELSE 0 END)::numeric /
        NULLIF(SUM(CASE WHEN pred_optimal=1 THEN 1 ELSE 0 END), 0) * 100, 2
    )                                                            AS precision_pct,
    ROUND(
        SUM(CASE WHEN class=1 AND pred_optimal=1 THEN 1 ELSE 0 END)::numeric /
        NULLIF(SUM(class), 0) * 100, 2
    )                                                            AS recall_pct
FROM model_predictions
WHERE split = 'test';


-- ── Q11: False positive breakdown by amount bucket ───────────
-- Which legitimate transaction types generate the most false alarms?
-- Helps fraud ops teams know where to apply manual review vs. auto-clear.
SELECT
    amount_bucket,
    COUNT(*)                                                   AS legit_flagged,
    ROUND(AVG(fraud_prob)::numeric, 4)                         AS avg_model_score,
    ROUND(MIN(fraud_prob)::numeric, 4)                         AS min_score,
    ROUND(MAX(fraud_prob)::numeric, 4)                         AS max_score,
    ROUND(AVG(amount)::numeric, 2)                             AS avg_amount
FROM model_predictions
WHERE split   = 'test'
  AND class   = 0
  AND pred_05 = 1
GROUP BY amount_bucket
ORDER BY legit_flagged DESC;


-- ── Q12: Missed fraud analysis (false negatives) ─────────────
-- What does the model miss? Understanding FN patterns informs
-- feature engineering and threshold tuning for the next model version.
SELECT
    amount_bucket,
    COUNT(*)                                                   AS missed_fraud,
    ROUND(AVG(fraud_prob)::numeric, 4)                         AS avg_model_score,
    ROUND(AVG(amount)::numeric, 2)                             AS avg_amount,
    ROUND(MAX(amount)::numeric, 2)                             AS max_amount,
    ROUND(AVG(CASE WHEN is_night=1 THEN 1.0 ELSE 0 END)::numeric * 100, 1)
                                                               AS pct_at_night
FROM model_predictions
WHERE split   = 'test'
  AND class   = 1
  AND pred_05 = 0
GROUP BY amount_bucket
ORDER BY missed_fraud DESC;


-- ── Q13: Model score distribution — fraud vs. legitimate ─────
-- Visualizes how well the model separates the two classes.
-- A good model shows fraud scores clustered near 1 and legit near 0.
SELECT
    score_bucket,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END)                AS fraud_count,
    SUM(CASE WHEN class = 0 THEN 1 ELSE 0 END)                AS legit_count,
    COUNT(*)                                                   AS total,
    ROUND(AVG(CASE WHEN class=1 THEN 1.0 ELSE 0 END)::numeric * 100, 1)
                                                               AS pct_fraud_in_bucket
FROM (
    SELECT class, ROUND(fraud_prob::numeric, 1) AS score_bucket
    FROM model_predictions
    WHERE split = 'test'
) scored
GROUP BY score_bucket
ORDER BY score_bucket;


-- ── Q14: Cost analysis at default vs. optimal threshold ──────
-- Translates model performance into dollars using:
--   False Negative cost: $500 (fraud loss + chargeback ops)
--   False Positive cost: $5  (analyst review + customer friction)
SELECT
    threshold,
    tp, fp, fn, tn,
    (tp + fp)                                                  AS flagged_for_review,
    ROUND((tp + fp)::numeric / NULLIF(tp + fp + fn + tn, 0) * 100, 3)
                                                               AS pct_transactions_reviewed,
    fn * 500 + fp * 5                                          AS total_cost_dollars,
    ROUND(tp::numeric / NULLIF(tp + fn, 0) * 100, 1)          AS fraud_recall_pct
FROM (
    SELECT
        '0.50 (default)'                                       AS threshold,
        SUM(CASE WHEN class=1 AND pred_05=1    THEN 1 ELSE 0 END) AS tp,
        SUM(CASE WHEN class=0 AND pred_05=1    THEN 1 ELSE 0 END) AS fp,
        SUM(CASE WHEN class=1 AND pred_05=0    THEN 1 ELSE 0 END) AS fn,
        SUM(CASE WHEN class=0 AND pred_05=0    THEN 1 ELSE 0 END) AS tn
    FROM model_predictions WHERE split = 'test'
    UNION ALL
    SELECT
        '0.46 (F1-optimal)',
        SUM(CASE WHEN class=1 AND pred_optimal=1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN class=0 AND pred_optimal=1 THEN 1 ELSE 0 END),
        SUM(CASE WHEN class=1 AND pred_optimal=0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN class=0 AND pred_optimal=0 THEN 1 ELSE 0 END)
    FROM model_predictions WHERE split = 'test'
) costs;


-- ── Q15: High-confidence predictions ─────────────────────────
-- Transactions where the model is most certain (score > 0.9).
-- These warrant automated blocking rather than human review.
SELECT
    CASE WHEN class = 1 THEN 'Fraud' ELSE 'Legit' END         AS actual_class,
    COUNT(*)                                                   AS count,
    ROUND(AVG(fraud_prob)::numeric, 4)                         AS avg_score,
    ROUND(AVG(amount)::numeric, 2)                             AS avg_amount,
    ROUND(AVG(is_night::numeric) * 100, 1)                     AS pct_at_night
FROM model_predictions
WHERE split      = 'test'
  AND fraud_prob >= 0.9
GROUP BY class
ORDER BY class DESC;
