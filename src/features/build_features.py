"""
Feature engineering pipeline.

Rolling/velocity features are computed in pandas (fast) then loaded
into Postgres. Pure SQL RANGE BETWEEN windows over 284k rows with
multiple passes are prohibitively slow without partitioning by card_id
(which this dataset lacks). Pandas rolling() on a sorted numeric index
handles this in seconds.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

PARQUET_IN  = Path("data/processed/transactions.parquet")
PARQUET_OUT = Path("data/processed/features.parquet")


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    return create_engine(url)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Time").reset_index(drop=True)

    # ── Amount features ──────────────────────────────────────────
    df["amount_log"]         = np.log1p(df["Amount"])
    global_mean              = df["Amount"].mean()
    global_std               = df["Amount"].std()
    df["amount_zscore"]      = (df["Amount"] - global_mean) / global_std
    df["amount_percentile"]  = df["Amount"].rank(pct=True)
    df["amount_bucket"]      = pd.cut(
        df["Amount"],
        bins=[-0.01, 10, 100, 1000, float("inf")],
        labels=["micro", "small", "medium", "large"],
    )

    # ── Time features ────────────────────────────────────────────
    df["is_night"]           = df["hour_of_day"].between(0, 5).astype(int)
    df["is_business_hours"]  = df["hour_of_day"].between(9, 17).astype(int)
    df["seconds_since_last_txn"] = df["Time"].diff().round(2)

    # ── Velocity features (rolling on time index) ────────────────
    # Set Time as index so pandas .rolling() uses a time-based window.
    # We convert seconds to a DatetimeIndex offset from epoch so the
    # window sizes (1h = 3600s, 24h = 86400s) work as timedeltas.
    t0 = time.time()
    print("Computing rolling velocity features ...")

    # Use a monotonic integer index in seconds for rolling
    df = df.set_index("Time")

    # 1-hour window (3600s)
    roll_1hr = df["Amount"].rolling(window=3600, min_periods=1)
    df["txn_count_last_1hr"]   = roll_1hr.count().astype(int) - 1
    df["amount_sum_last_1hr"]  = (roll_1hr.sum() - df["Amount"]).round(2)
    df["avg_amount_last_1hr"]  = roll_1hr.mean().round(2)

    # 24-hour window (86400s)
    roll_24hr = df["Amount"].rolling(window=86400, min_periods=1)
    df["txn_count_last_24hr"]  = roll_24hr.count().astype(int) - 1

    # Ratio of current amount to 1-hour rolling average
    df["amount_vs_1hr_avg"] = (
        df["Amount"] / df["avg_amount_last_1hr"].replace(0, np.nan)
    ).round(4)

    df = df.reset_index().rename(columns={"Time": "time_seconds"})
    print(f"Rolling features done in {time.time() - t0:.1f}s")

    # Rename remaining raw columns to snake_case for Postgres
    df = df.rename(columns={
        "Amount": "amount",
        "Class":  "class",
        **{f"V{i}": f"v{i}" for i in range(1, 29)},
    })

    return df


def save_parquet(df: pd.DataFrame) -> None:
    PARQUET_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_OUT, index=False)
    print(f"Saved {len(df):,} rows to {PARQUET_OUT}")


def load_to_postgres(df: pd.DataFrame, engine) -> None:
    pg_df = df.copy()
    pg_df["amount_bucket"] = pg_df["amount_bucket"].astype(str)

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS transaction_features"))
        conn.commit()

    print("Loading transaction_features into Postgres ...")
    t0 = time.time()
    pg_df.to_sql(
        "transaction_features", engine,
        if_exists="replace", index=False,
        method="multi", chunksize=10_000,
    )
    with engine.connect() as conn:
        for col, idx in [
            ("class",        "idx_tf_class"),
            ("hour_of_day",  "idx_tf_hour"),
            ("amount_bucket","idx_tf_bucket"),
            ("time_seconds", "idx_tf_time"),
            ("is_night",     "idx_tf_night"),
        ]:
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS {idx} ON transaction_features ({col})"
            ))
        conn.commit()
    print(f"Postgres load done in {time.time() - t0:.1f}s")


def print_summary(df: pd.DataFrame) -> None:
    print("\n--- Feature comparison: Fraud vs. Legit ---")
    compare = [
        "amount_log", "amount_zscore", "is_night",
        "txn_count_last_1hr", "amount_vs_1hr_avg", "seconds_since_last_txn",
    ]
    print(
        df.groupby("class")[compare]
        .mean().round(4)
        .rename(index={0: "Legit", 1: "Fraud"})
        .to_string()
    )


if __name__ == "__main__":
    print(f"Reading {PARQUET_IN} ...")
    df_raw = pd.read_parquet(PARQUET_IN)
    df     = build_features(df_raw)
    save_parquet(df)
    engine = get_engine()
    load_to_postgres(df, engine)
    print_summary(df)
    print("\nDone.")
