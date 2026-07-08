"""
Train and evaluate fraud classifiers with cost-weighted metrics.

Experiments run:
  Imbalance comparison (Logistic Regression):
    1. No handling (shows why you need it)
    2. Class weighting
    3. SMOTE oversampling
    4. Random undersampling

  Model comparison (best imbalance method — class weighting):
    5. Random Forest
    6. XGBoost

Split: time-based 80/20 (first 80% of transactions by time_seconds).
       Random split would leak future transactions into training — wrong
       for time-series fraud data.

Results saved to results/experiment_log.csv and results/threshold_sweeps/.
Best model (by PR-AUC) saved to results/best_model.pkl.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.models.evaluate import full_report, threshold_sweep, FN_COST, FP_COST

FEATURES_PATH = Path("data/processed/features.parquet")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "threshold_sweeps").mkdir(exist_ok=True)

FEATURE_COLS = (
    [f"v{i}" for i in range(1, 29)]
    + [
        "amount_log", "amount_zscore", "amount_percentile",
        "is_night", "is_business_hours",
        "txn_count_last_1hr", "txn_count_last_24hr",
        "amount_sum_last_1hr", "avg_amount_last_1hr", "amount_vs_1hr_avg",
        "seconds_since_last_txn",
    ]
)
TARGET = "class"


def load_and_split(path: Path):
    df = pd.read_parquet(path).sort_values("time_seconds")
    df["seconds_since_last_txn"] = df["seconds_since_last_txn"].fillna(0)

    split_idx = int(len(df) * 0.80)
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values

    print(f"Train: {len(train):,} rows | {y_train.sum()} fraud ({y_train.mean()*100:.3f}%)")
    print(f"Test:  {len(test):,}  rows | {y_test.sum()}  fraud ({y_test.mean()*100:.3f}%)")
    return X_train, y_train, X_test, y_test


def scale(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def run_experiment(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    imbalance: str = "none",
) -> dict:
    print(f"\n[{name}] Training ...")
    t0 = time.time()

    if imbalance == "smote":
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr, y_tr = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE: {y_train.sum()} → {y_tr.sum()} fraud samples")
    elif imbalance == "undersample":
        rus = RandomUnderSampler(random_state=42)
        X_tr, y_tr = rus.fit_resample(X_train, y_train)
        print(f"  Undersample: {len(y_train):,} → {len(y_tr):,} rows")
    else:
        X_tr, y_tr = X_train, y_train

    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = full_report(y_test, y_prob, label=name, fn_cost=FN_COST, fp_cost=FP_COST)
    metrics["train_time_s"] = round(elapsed, 2)
    metrics["imbalance"]    = imbalance

    sweep = threshold_sweep(y_test, y_prob)
    sweep.to_csv(RESULTS_DIR / "threshold_sweeps" / f"{name.replace(' ', '_')}.csv", index=False)

    print(
        f"  PR-AUC={metrics['pr_auc']:.4f}  "
        f"F1@0.5={metrics['f1_at_0.5']:.4f}  "
        f"F1@opt={metrics['f1_optimal']:.4f}  "
        f"cost@opt_t=${metrics['cost_at_optimal_t']:,.0f}  "
        f"({elapsed:.1f}s)"
    )
    return metrics


def main():
    print("=" * 60)
    print("Fraud Detection — Model Training")
    print(f"FN cost: ${FN_COST}  |  FP cost: ${FP_COST}")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_and_split(FEATURES_PATH)
    X_train_sc, X_test_sc, scaler = scale(X_train, X_test)

    experiments = [
        # ── Imbalance comparison (Logistic Regression) ───────────
        (
            "LR_no_handling",
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_sc, y_train, X_test_sc, y_test, "none",
        ),
        (
            "LR_class_weight",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            X_train_sc, y_train, X_test_sc, y_test, "class_weight",
        ),
        (
            "LR_SMOTE",
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_sc, y_train, X_test_sc, y_test, "smote",
        ),
        (
            "LR_undersample",
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_sc, y_train, X_test_sc, y_test, "undersample",
        ),
        # ── Model comparison (class weighting) ───────────────────
        (
            "RF_class_weight",
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced",
                n_jobs=-1, random_state=42,
            ),
            X_train, y_train, X_test, y_test, "class_weight",
        ),
        (
            "XGB_class_weight",
            XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                eval_metric="aucpr", random_state=42, n_jobs=-1,
            ),
            X_train, y_train, X_test, y_test, "class_weight",
        ),
    ]

    all_results = []
    best_model  = None
    best_auc    = -1
    best_name   = ""

    for name, model, Xtr, ytr, Xte, yte, imb in experiments:
        metrics = run_experiment(name, model, Xtr, ytr, Xte, yte, imb)
        all_results.append(metrics)
        if metrics["pr_auc"] > best_auc:
            best_auc  = metrics["pr_auc"]
            best_model = model
            best_name  = name

    # Save experiment log
    log = pd.DataFrame(all_results)
    log_path = RESULTS_DIR / "experiment_log.csv"
    log.to_csv(log_path, index=False)
    print(f"\nExperiment log saved to {log_path}")

    # Save best model
    model_path = RESULTS_DIR / "best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_model, "scaler": scaler, "features": FEATURE_COLS}, f)
    print(f"Best model ({best_name}, PR-AUC={best_auc:.4f}) saved to {model_path}")

    # Save best model metadata as JSON (tracked in git; pkl is gitignored)
    meta = {
        "best_model": best_name,
        "pr_auc": best_auc,
        "feature_cols": FEATURE_COLS,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
    }
    with open(RESULTS_DIR / "best_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    summary = log[["label", "imbalance", "pr_auc", "f1_at_0.5",
                   "f1_optimal", "optimal_threshold", "cost_at_optimal_t"]].copy()
    summary.columns = ["Model", "Imbalance", "PR-AUC", "F1@0.5",
                       "F1@Opt", "Opt Thresh", "Cost@Opt"]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
