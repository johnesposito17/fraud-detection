"""
Load and validate the raw Kaggle credit card fraud CSV.
Outputs a cleaned parquet to data/processed/transactions.parquet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/creditcard.csv")
PROCESSED_PATH = Path("data/processed/transactions.parquet")

EXPECTED_COLUMNS = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}


def load_and_validate(path: Path = RAW_PATH) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        print(f"ERROR: missing expected columns: {missing}", file=sys.stderr)
        sys.exit(1)

    null_counts = df.isnull().sum()
    if null_counts.any():
        print("WARNING: null values detected:")
        print(null_counts[null_counts > 0])

    assert df["Class"].isin([0, 1]).all(), "Class column contains values other than 0/1"
    assert (df["Amount"] >= 0).all(), "Negative amounts detected"
    assert (df["Time"] >= 0).all(), "Negative time values detected"

    print(f"Rows: {len(df):,}")
    print(f"Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"Legit: {(df['Class'] == 0).sum():,}")
    print(f"Amount range: ${df['Amount'].min():.2f} – ${df['Amount'].max():.2f}")

    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time in the raw data is seconds elapsed since the first transaction,
    spanning ~48 hours. We derive hour-of-day by taking modulo 86400.
    This is an approximation since we don't know the actual start time,
    but it captures intraday patterns for EDA.
    """
    df = df.copy()
    df["hour_of_day"] = (df["Time"] % 86400) // 3600
    df["day"] = (df["Time"] // 86400).astype(int)
    return df


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved processed data to {path}")


if __name__ == "__main__":
    df = load_and_validate()
    df = engineer_time_features(df)
    save_processed(df)
    print("Done.")
