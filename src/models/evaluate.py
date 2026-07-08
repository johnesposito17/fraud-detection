"""
Shared evaluation utilities used by train.py and the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)

# Cost assumptions (documented in docs/NOTES.md)
# FN: missed fraud — cardholder absorbs loss, bank pays chargeback/ops cost
# FP: false alert — analyst review time + customer friction (declined legitimate txn)
FN_COST = 500   # dollars: conservative fraud loss estimate
FP_COST = 5     # dollars: review time + customer friction


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return average_precision_score(y_true, y_prob)


def f1_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    return f1_score(y_true, y_pred, zero_division=0)


def cost_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn * fn_cost + fp * fp_cost


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> tuple[float, float]:
    """Return (best_threshold, best_score). metric: 'f1' or 'cost'."""
    thresholds = np.linspace(0.01, 0.99, 200)
    if metric == "f1":
        scores = [f1_at_threshold(y_true, y_prob, t) for t in thresholds]
        best_idx = int(np.argmax(scores))
    else:
        scores = [cost_at_threshold(y_true, y_prob, t, fn_cost, fp_cost) for t in thresholds]
        best_idx = int(np.argmin(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
    n_points: int = 200,
) -> pd.DataFrame:
    """Full threshold sweep — used by the dashboard slider."""
    thresholds = np.linspace(0.01, 0.99, n_points)
    rows = []
    fraud_total = y_true.sum()
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        total_cost = fn * fn_cost + fp * fp_cost
        fraud_caught_pct = tp / fraud_total * 100 if fraud_total > 0 else 0.0
        rows.append({
            "threshold":        round(float(t), 4),
            "precision":        round(precision, 4),
            "recall":           round(recall, 4),
            "f1":               round(f1, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "total_cost":       round(total_cost, 2),
            "fraud_caught_pct": round(fraud_caught_pct, 2),
        })
    return pd.DataFrame(rows)


def full_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "",
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> dict:
    """Return a dict of all metrics for experiment logging."""
    auc = pr_auc(y_true, y_prob)
    f1_default = f1_at_threshold(y_true, y_prob, 0.5)
    best_t, best_f1 = optimal_threshold(y_true, y_prob, metric="f1")
    cost_t, min_cost = optimal_threshold(y_true, y_prob, metric="cost", fn_cost=fn_cost, fp_cost=fp_cost)

    y_pred_default = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_default, labels=[0, 1]).ravel()
    cost_default = fn * fn_cost + fp * fp_cost

    y_pred_best = (y_prob >= best_t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_true, y_pred_best, labels=[0, 1]).ravel()
    cost_best_t = fn2 * fn_cost + fp2 * fp_cost

    return {
        "label":             label,
        "pr_auc":            round(auc, 4),
        "f1_at_0.5":         round(f1_default, 4),
        "f1_optimal":        round(best_f1, 4),
        "optimal_threshold": round(best_t, 4),
        "cost_at_0.5":       round(cost_default, 2),
        "cost_at_optimal_t": round(cost_best_t, 2),
        "min_cost":          round(min_cost, 2),
        "cost_opt_threshold":round(cost_t, 4),
        "tp_default": int(tp), "fp_default": int(fp),
        "fn_default": int(fn), "tn_default": int(tn),
        "tp_optimal": int(tp2), "fp_optimal": int(fp2),
        "fn_optimal": int(fn2), "tn_optimal": int(tn2),
        "fn_cost": fn_cost, "fp_cost": fp_cost,
    }
