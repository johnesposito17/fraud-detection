"""
Load processed transaction data into PostgreSQL.
Reads from data/processed/transactions.parquet and bulk-inserts via COPY.

Usage:
    python src/ingestion/load_to_postgres.py
"""

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

PARQUET_PATH = Path("data/processed/transactions.parquet")

COLUMN_MAP = {
    "Time":       "time_seconds",
    "hour_of_day": "hour_of_day",
    "day":        "day",
    "Amount":     "amount",
    "Class":      "class",
    **{f"V{i}": f"v{i}" for i in range(1, 29)},
}


def get_engine():
    url = (
        f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )
    return create_engine(url)


def load(engine) -> None:
    print(f"Reading {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]

    with engine.connect() as conn:
        existing = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
        if existing > 0:
            print(f"Table already has {existing:,} rows. Truncating before reload ...")
            conn.execute(text("TRUNCATE TABLE transactions RESTART IDENTITY"))
            conn.commit()

    print(f"Inserting {len(df):,} rows ...")
    t0 = time.time()
    df.to_sql(
        "transactions",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=10_000,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")


def verify(engine) -> None:
    with engine.connect() as conn:
        total   = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
        fraud   = conn.execute(text("SELECT COUNT(*) FROM transactions WHERE class = 1")).scalar()
        legit   = conn.execute(text("SELECT COUNT(*) FROM transactions WHERE class = 0")).scalar()
        avg_amt = conn.execute(text("SELECT ROUND(AVG(amount)::numeric, 2) FROM transactions")).scalar()
    print(f"\n--- Verification ---")
    print(f"Total rows : {total:,}")
    print(f"Fraud      : {fraud:,} ({fraud/total*100:.3f}%)")
    print(f"Legit      : {legit:,}")
    print(f"Avg amount : ${avg_amt}")


if __name__ == "__main__":
    engine = get_engine()
    load(engine)
    verify(engine)
