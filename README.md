# Real-Time Credit Card Fraud Detection System

A full-stack fraud analytics pipeline demonstrating the tooling a card-fraud risk team would actually use: behavioral feature engineering, imbalanced-data ML with cost-weighted evaluation, a 20-query SQL library, and an analyst-facing Streamlit dashboard with a live threshold slider.

**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European cardholder transactions, 0.173% fraud rate  
**Stack:** Python · PostgreSQL · SQL · Streamlit · scikit-learn · XGBoost

---

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                        # Raw Kaggle CSV (gitignored)
│   └── processed/                  # Parquet outputs (gitignored)
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory analysis with 6 charts
├── src/
│   ├── ingestion/
│   │   ├── load_data.py            # Validate CSV → parquet
│   │   └── load_to_postgres.py     # Load parquet → PostgreSQL
│   ├── features/
│   │   └── build_features.py       # Engineer 12 features → parquet + Postgres
│   └── models/
│       ├── evaluate.py             # Shared metrics (PR-AUC, cost, threshold sweep)
│       ├── train.py                # Train 6 experiments, log results
│       └── predict.py              # Score all transactions → model_predictions table
├── sql/
│   ├── schema.sql                  # PostgreSQL table definitions + indexes
│   ├── feature_engineering.sql     # Feature definitions as SQL reference
│   ├── 01_fraud_patterns.sql       # Q1–Q8: fraud rate by time, amount, velocity
│   ├── 02_model_evaluation.sql     # Q9–Q15: confusion matrix, FP/FN analysis, cost
│   └── 03_threshold_analysis.sql   # Q16–Q20: threshold sweep, break-even, queue
├── dashboard/
│   └── app.py                      # Streamlit analyst dashboard
├── results/
│   ├── experiment_log.csv          # All model runs with metrics
│   ├── best_model_meta.json        # Best model metadata
│   └── threshold_sweeps/           # Per-model precision/recall/cost at 200 thresholds
├── docs/
│   ├── eda_summary.md              # Key EDA findings
│   ├── feature_rationale.md        # Per-feature fraud-detection rationale
│   ├── NOTES.md                    # Design tradeoffs and decisions
│   └── figures/                    # EDA charts (PNG)
├── docker-compose.yml              # PostgreSQL 16 via Docker
├── .env.example                    # Database credential template
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- Docker Desktop (for PostgreSQL)
- Kaggle account (to download dataset)

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
brew install libomp             # macOS only — required by XGBoost
```

### 2. Configure environment
```bash
cp .env.example .env
# Default credentials match docker-compose.yml — no changes needed
```

### 3. Download the dataset
Go to [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), click **Download**, and move the CSV:
```bash
mv ~/Downloads/creditcard.csv data/raw/creditcard.csv
```

### 4. Start PostgreSQL
```bash
docker compose up -d
```

---

## Quickstart

Run the full pipeline end-to-end:

```bash
# 1. Validate and convert raw data to parquet
python src/ingestion/load_data.py

# 2. Load into PostgreSQL
python src/ingestion/load_to_postgres.py

# 3. Engineer features (runs in pandas, loads result to Postgres)
python src/features/build_features.py

# 4. Train all models and log results
python -m src.models.train

# 5. Generate predictions and load to Postgres
python -m src.models.predict

# 6. Launch dashboard
streamlit run dashboard/app.py
```

---

## Methodology

### Data
The Kaggle dataset contains 284,807 transactions from European cardholders over a 48-hour window in September 2013. Features V1–V28 are the result of PCA applied to the original (confidential) transaction attributes. `Time` is seconds elapsed since the first transaction; `Amount` is the transaction value; `Class` is the fraud label (1 = fraud).

### Feature Engineering
12 features engineered on top of the raw PCA components (see [`docs/feature_rationale.md`](docs/feature_rationale.md)):

| Category | Features |
|----------|----------|
| Amount | `amount_log`, `amount_zscore`, `amount_percentile`, `amount_bucket` |
| Time | `is_night`, `is_business_hours`, `seconds_since_last_txn` |
| Velocity (global stream) | `txn_count_last_1hr`, `txn_count_last_24hr`, `amount_sum_last_1hr`, `avg_amount_last_1hr`, `amount_vs_1hr_avg` |

Velocity features are computed over the global transaction stream (no card identifier in this dataset). In production they would be partitioned by `card_id`.

### Modeling
Six experiments across imbalance strategies and model families, evaluated on a **time-based 80/20 split** (first 38 hours train, last 10 hours test):

| Model | Imbalance | PR-AUC | F1 @ Opt | Cost @ Opt |
|-------|-----------|--------|----------|------------|
| Logistic Regression | None | 0.736 | 0.723 | $12,075 |
| Logistic Regression | Class weight | 0.756 | 0.598 | $8,805 |
| Logistic Regression | SMOTE | 0.764 | 0.600 | $7,825 |
| Logistic Regression | Undersample | 0.724 | 0.559 | $8,385 |
| **Random Forest** | **Class weight** | **0.820** | **0.857** | **$9,005** |
| XGBoost | Class weight | 0.794 | 0.809 | $10,030 |

**Primary metric:** Precision-Recall AUC. Plain accuracy is meaningless at 0.173% fraud rate — a model predicting "legit" for everything achieves 99.83% accuracy while catching zero fraud.

**Cost metric:** `total_cost = false_negatives × $500 + false_positives × $5`  
Assumptions: $500 per missed fraud (loss + chargeback ops); $5 per false positive (analyst review + customer friction). Adjustable in the dashboard sidebar.

### SQL Query Library
20 production-style queries in three files covering fraud patterns, model evaluation, and threshold analysis. All queries verified against live PostgreSQL data.

---

## Key Findings

1. **578:1 class imbalance** — requires explicit handling; plain accuracy and ROC-AUC are misleading metrics at this ratio. PR-AUC is the right primary metric.

2. **Fraud peaks at 2 AM** — fraud rate at 02:00 is 1.71% vs. 0.17% baseline (10× elevated). Consistent with attackers exploiting off-peak hours when monitoring thresholds may be looser and cardholders are asleep.

3. **Micro transactions skew fraudulent** — transactions under $10 have the highest fraud concentration (0.25%), consistent with test-charge behavior before larger cash-out attempts. Median fraud amount ($9.25) is less than half the median legitimate amount ($22.00).

4. **V3, V14, V17 are the most discriminative features** — largest mean separation between fraud and legitimate distributions among the 28 PCA components. These dominate feature importance in tree-based models.

5. **Random Forest outperforms all variants** — PR-AUC 0.820, F1 0.857 at optimal threshold. Tree-based models handle the PCA feature structure better than logistic regression. XGBoost is competitive (PR-AUC 0.794) and trains 10× faster, making it preferable in production for real-time scoring.

---

## Business Impact

At the Random Forest model's optimal threshold (0.458) on the held-out test set:

- **76% of fraud caught** — 57 of 75 fraud cases detected
- **Review queue:** 58 transactions flagged (57 real fraud + 1 false alarm)
- **99.8% precision** — nearly every flagged transaction is genuine fraud
- **Total cost: $9,005** — vs. $37,500 baseline (missing all fraud with no model)
- **$28,495 saved** per test-set equivalent period (~10 hours)

Extrapolated to the full 48-hour window: approximately **$136,000 in projected savings** vs. operating without a model, while sending fewer than 300 transactions to the review queue out of 284,807 total — a **review rate of 0.10%**.

These figures use conservative cost assumptions ($500/FN, $5/FP) that can be tuned interactively in the dashboard.

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. The dashboard runs entirely from local parquet/CSV files — no live database connection required after the pipeline has been run once.

**Sidebar controls:**
- **Model** — switch between all 6 trained models
- **Decision Threshold** — drag to see precision/recall/cost update in real time
- **Cost Parameters** — adjust FN and FP costs; all cost charts update live

---

## Tradeoffs & Design Decisions

See [`docs/NOTES.md`](docs/NOTES.md) for full rationale on:
- Why Random Forest over XGBoost for the best model
- Why class weighting over SMOTE as the primary imbalance strategy
- Why PR-AUC over ROC-AUC and plain F1
- Why a time-based train/test split (not random)
- Why velocity features were computed in pandas rather than SQL
- Cost model design and assumptions
