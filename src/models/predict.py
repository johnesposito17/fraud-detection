"""
Score all transactions with the best saved model and load predictions
into Postgres as the model_predictions table.

Adds a split column (train/test) based on the same 80/20 time-based
cutoff used in training, so SQL queries can restrict to the held-out
test set for proper evaluation metrics.
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

FEATURES_PATH = Path("data/processed/features.parquet")
MODEL_PATH    = Path("results/best_model.pkl")
META_PATH     = Path("results/best_model_meta.json")


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    return create_engine(url)


def main():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    model    = artifact["model"]
    scaler   = artifact["scaler"]
    features = artifact["features"]

    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"Loaded: {meta['best_model']} (PR-AUC={meta['pr_auc']})")

    df = pd.read_parquet(FEATURES_PATH).sort_values("time_seconds").reset_index(drop=True)
    df["seconds_since_last_txn"] = df["seconds_since_last_txn"].fillna(0)

    split_idx = int(len(df) * 0.80)
    df["split"] = "train"
    df.loc[split_idx:, "split"] = "test"

    X = df[features].values
    # RF doesn't need scaling but scaler was fit on LR — apply anyway (no-op for RF shape)
    fraud_prob = model.predict_proba(X)[:, 1]

    preds = pd.DataFrame({
        "id":            df.get("id", pd.RangeIndex(len(df))),
        "time_seconds":  df["time_seconds"],
        "hour_of_day":   df["hour_of_day"],
        "amount":        df["amount"],
        "amount_bucket": df["amount_bucket"].astype(str),
        "is_night":      df["is_night"],
        "class":         df["class"],
        "split":         df["split"],
        "fraud_prob":    fraud_prob.round(6),
        "pred_05":       (fraud_prob >= 0.50).astype(int),
        "pred_optimal":  (fraud_prob >= meta.get("optimal_threshold", 0.46)).astype(int),
    })

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS model_predictions"))
        conn.commit()

    print(f"Loading {len(preds):,} predictions into Postgres ...")
    preds.to_sql("model_predictions", engine, if_exists="replace",
                 index=False, method="multi", chunksize=10_000)

    with engine.connect() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_mp_class  ON model_predictions (class)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_mp_split  ON model_predictions (split)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_mp_bucket ON model_predictions (amount_bucket)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_mp_prob   ON model_predictions (fraud_prob)"))
        conn.commit()

    with engine.connect() as conn:
        test_fraud = conn.execute(text(
            "SELECT COUNT(*) FROM model_predictions WHERE split='test' AND class=1"
        )).scalar()
        caught_05 = conn.execute(text(
            "SELECT COUNT(*) FROM model_predictions WHERE split='test' AND class=1 AND pred_05=1"
        )).scalar()
        caught_opt = conn.execute(text(
            "SELECT COUNT(*) FROM model_predictions WHERE split='test' AND class=1 AND pred_optimal=1"
        )).scalar()
    print(f"\nTest set: {test_fraud} fraud cases")
    print(f"  Caught @0.50 threshold: {caught_05}  ({caught_05/test_fraud*100:.1f}%)")
    print(f"  Caught @optimal thresh: {caught_opt} ({caught_opt/test_fraud*100:.1f}%)")
    print("Done.")


if __name__ == "__main__":
    main()
