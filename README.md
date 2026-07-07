# Real-Time Credit Card Fraud Detection System

A full-stack fraud analytics pipeline built to demonstrate the kind of tooling a card-fraud risk team would actually use: behavioral feature engineering in SQL, imbalanced-data ML with cost-weighted evaluation, and an analyst-facing dashboard with a live threshold slider.

**Built for:** Capital One Data Analyst internship portfolio  
**Dataset:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European cardholder transactions, 0.172% fraud rate  
**Stack:** Python · PostgreSQL · SQL · Streamlit

---

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/            # Raw Kaggle CSV (gitignored)
│   └── processed/      # Cleaned + feature-engineered data (gitignored)
├── notebooks/          # EDA and exploratory analysis
├── src/
│   ├── ingestion/      # Data loading and Postgres ingestion scripts
│   ├── features/       # Feature engineering pipeline
│   └── models/         # Model training, evaluation, experiment logging
├── sql/                # Analyst SQL query library (fraud patterns, model eval)
├── dashboard/          # Streamlit analyst dashboard
└── docs/               # EDA summary, methodology notes, NOTES.md
```

---

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ (local or Docker)
- Kaggle account (to download dataset)

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure database
```bash
cp .env.example .env
# Edit .env with your Postgres credentials
```

### Download data
```bash
# Requires Kaggle CLI configured with your API key
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

---

## Quickstart

```bash
# 1. Load data into PostgreSQL
python src/ingestion/load_to_postgres.py

# 2. Run feature engineering (SQL window functions)
python src/features/build_features.py

# 3. Train and evaluate models
python src/models/train.py

# 4. Launch dashboard
streamlit run dashboard/app.py
```

---

## Methodology

*(To be filled in — Phase 7)*

---

## Key Findings

*(To be filled in after EDA and modeling)*

---

## Business Impact

*(To be filled in — Phase 7)*

---

## Tradeoffs & Design Decisions

See [`docs/NOTES.md`](docs/NOTES.md) for documented tradeoffs including:
- Model selection rationale
- Imbalanced-data handling approach
- Threshold choice and cost-weighted metric design
