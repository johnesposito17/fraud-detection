# EDA Summary — Credit Card Fraud Detection

**Dataset:** 284,807 transactions | 492 fraud (0.173%) | 284,315 legitimate  
**Features:** Time (seconds elapsed), V1–V28 (PCA-anonymized), Amount, Class

---

## Key Findings

### 1. Class Imbalance — 578:1 ratio
The dataset has a 578:1 legitimate-to-fraud ratio. This makes plain accuracy meaningless
(a model predicting "legit" for everything gets 99.83% accuracy while catching zero fraud).
All modeling uses precision-recall AUC, F1, and cost-weighted metrics.

### 2. Transaction Amount Patterns
| Metric | Legitimate | Fraud |
|--------|-----------|-------|
| Median | $22.00 | $9.25 |
| Mean   | $88.29 | $122.21 |
| Max    | $25,691.16 | $2,125.87 |

Fraud transactions have a **lower median** ($9.25 vs $22.00), consistent with small
"test charges" attackers use to verify a stolen card before larger withdrawals.
However, the **mean fraud amount ($122) exceeds legitimate ($88)** due to a tail of
large fraudulent cash-out attempts, indicating fraudsters operate in two modes:
low-value probing and high-value extraction.

### 3. Time-of-Day Signal
- **Fraud rate peaks at 02:00** (1.713%) — nearly 10x the baseline rate
- Transaction volume is lowest at night, yet fraud rate is highest
- This is consistent with attackers targeting off-peak windows when real-time
  monitoring thresholds may be lower and cardholders are unlikely to notice alerts
- **Business implication:** a time-of-day feature or dynamic threshold that tightens
  scoring between midnight and 4am could catch high-rate fraud with minimal false
  positive impact on daytime legitimate traffic

### 4. Most Discriminative PCA Features
The following features show the largest mean separation between fraud and legitimate
distributions (higher = more discriminative):

| Feature | Mean Separation | Notes |
|---------|----------------|-------|
| V3  | 7.05 | Strongest signal |
| V14 | 6.98 | Strong negative skew in fraud |
| V17 | 6.68 | |
| V12 | 6.27 | |
| V10 | 5.69 | |
| V7  | 5.58 | |
| V1  | 4.78 | |
| V4  | 4.55 | |

V3, V14, and V17 will likely be the top feature importances in tree-based models.
Since features are PCA-derived, their original meaning is anonymized — but their
discriminative power is empirically clear from the distribution plots.

### 5. Feature Correlation Structure
PCA features are orthogonal by construction — the correlation heatmap for fraud
transactions shows near-zero correlations among V1–V28. Amount has modest
correlations with a few V features. This means:
- No multicollinearity concerns for logistic regression
- Tree-based models won't gain much from interaction features between PCA components
- Amount remains a useful standalone feature (different scale than V features — will need standardization for logistic regression)

---

## Modeling Implications

1. **Do not use accuracy** — use precision-recall AUC and cost-weighted F-beta
2. **Address imbalance** — test SMOTE, class weighting, and undersampling; compare tradeoffs
3. **Prioritize V3, V14, V17, V12** in feature importance analysis
4. **Scale Amount** before logistic regression (V features are already standardized post-PCA)
5. **Consider hour_of_day as a feature** — 02:00 peak is a real signal
