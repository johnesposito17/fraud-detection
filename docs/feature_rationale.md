# Feature Engineering Rationale

Each feature is documented with its fraud-detection motivation. A reviewer
should be able to ask "why does this feature signal fraud?" and get a crisp
business answer for any column in the model.

---

## Amount Features

### `amount_log` — Log-transformed transaction amount
**Why it signals fraud:** Raw `Amount` spans $0–$25,691 with a severe right skew.
Log-transforming compresses outliers and brings the distribution closer to normal,
which stabilizes gradient-based models and regularized regression. Without this,
a handful of large legitimate transactions can dominate the loss function.

**Business framing:** Models trained on raw dollar amounts are implicitly more
sensitive to $10,000 transactions than $10 ones. Log-scaling treats a 10× change
consistently regardless of magnitude.

---

### `amount_zscore` — Standardized amount (z-score vs. global mean/std)
**Why it signals fraud:** Fraud occurs at both extremes — very small amounts
($1–$5 "test charges" to verify a stolen card is live) and very large amounts
(cash-out attempts). A z-score above +2 or below −1 flags statistical outliers.

**Business framing:** A z-score feature lets a risk analyst write a simple rule:
"flag any transaction more than 3 standard deviations from the mean" without
needing to know the actual dollar threshold, which changes as spending patterns shift.

---

### `amount_percentile` — Rank-based percentile (0–1)
**Why it signals fraud:** A rank-based transform is robust to outliers in a way
z-score is not. The 99th-percentile transaction is always unusual regardless of
whether the dataset contains extreme values.

**Business framing:** Useful for segment-level comparisons ("this transaction is
in the 95th percentile for its time-of-day bucket") without requiring per-segment
mean/std calculations.

---

### `amount_bucket` — Categorical size tier (micro/small/medium/large)
**Why it signals fraud:** EDA showed fraud skews toward micro transactions ($0–$10),
consistent with test-charge behavior. Bucketing makes this pattern exploitable by
tree-based models as a direct split and by SQL rules in the analyst query library.

**Thresholds:**
- `micro`: < $10 — high fraud concentration; test charge range
- `small`: $10–$100 — typical everyday spending; baseline fraud rate
- `medium`: $100–$1,000 — elevated fraud for cash-out attempts
- `large`: > $1,000 — rare but high-value fraud target

---

## Time Features

### `is_night` — Binary flag: hour 0–5 AM
**Why it signals fraud:** EDA showed fraud rate at 02:00 AM is 1.713% vs. the
0.173% baseline — nearly 10× elevated. Attackers exploit off-peak hours when
cardholders are asleep (less likely to receive and respond to alerts) and when
some monitoring systems apply looser thresholds due to low transaction volume.

**Business framing:** A simple binary flag captures this signal without overfitting
to a specific hour. In production, this would also interact with the cardholder's
timezone and typical behavior pattern.

---

### `is_business_hours` — Binary flag: hour 9 AM–5 PM
**Why it signals fraud:** The inverse of `is_night`. Legitimate consumer spending
clusters heavily during business hours; fraud is disproportionately absent during
peak legitimate volume times (harder to hide in noise, more monitoring staff on call).

---

### `seconds_since_last_txn` — Time gap from the preceding transaction
**Why it signals fraud:** Automated fraud (card-testing bots, account-takeover
scripts) generates transactions with unnaturally short inter-arrival times — often
sub-second. A human spending $5 at a coffee shop and then $500 at an electronics
store within 2 seconds is physically impossible; a bot is not.

**Limitation:** Without a card ID, this measures the gap between any two consecutive
transactions in the dataset, not between two transactions on the same card. In
production this would be `LAG(time) OVER (PARTITION BY card_id ORDER BY time)`.

---

## Velocity Features (Global Stream)

> **Important caveat:** The Kaggle dataset contains no card identifier — V1–V28
> are PCA-anonymized. All velocity features below are computed over the *global*
> transaction stream. In a production system, they would be partitioned by
> `card_id` using `PARTITION BY card_id` in each window. The SQL pattern is
> identical; only the partition key differs. This limitation is worth stating
> explicitly in interviews — it shows you understand what the feature *should*
> measure even when the data constrains what you can compute.

---

### `txn_count_last_1hr` — Transaction count in preceding 60 minutes
**Why it signals fraud:** Card-testing attacks typically involve rapid-fire small
transactions to verify which stolen card numbers are active. A spike in system-wide
transaction rate — especially during off-hours — is a real-time fraud indicator
used by fraud operations teams.

**In production:** `COUNT(*) OVER (PARTITION BY card_id ORDER BY txn_time RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW)`

---

### `txn_count_last_24hr` — Transaction count in preceding 24 hours
**Why it signals fraud:** Establishes a daily velocity baseline. A cardholder who
normally makes 3 transactions/day suddenly generating 40 in 24 hours is a strong
account-takeover signal. The 24-hour window also captures fraud that is deliberately
spread across more than 1 hour to evade short-window rules.

---

### `amount_sum_last_1hr` — Total dollar volume in preceding 60 minutes
**Why it signals fraud:** Even if individual transaction amounts look normal,
the aggregate dollar value processed in a short window can be anomalous.
A fraudster making ten $200 transactions in an hour looks unremarkable
per-transaction but alarming in aggregate.

---

### `avg_amount_last_1hr` — Mean transaction amount in preceding 60 minutes
**Why it signals fraud:** Provides context for the current transaction's amount.
A $500 transaction is unusual if the preceding hour's average was $20.

---

### `amount_vs_1hr_avg` — Ratio of current amount to 1-hour rolling average
**Why it signals fraud:** A ratio >> 1 means the current transaction is
disproportionately large relative to the recent period. This is a normalized
anomaly score that is more interpretable than raw z-score for time-series
patterns. Values above 5× or 10× are strong fraud signals for cash-out attempts.

---

## PCA Features (V1–V28)

Passed through from the raw dataset without modification. These are the result
of PCA applied to the original (confidential) transaction features by the dataset
authors. Their interpretation is anonymized, but their discriminative power is
empirically clear from EDA:

| Feature | Mean Separation (fraud vs. legit) |
|---------|----------------------------------|
| V3      | 7.05 (strongest) |
| V14     | 6.98 |
| V17     | 6.68 |
| V12     | 6.27 |
| V10     | 5.69 |
| V7      | 5.58 |
| V1      | 4.78 |
| V4      | 4.55 |

These will dominate model feature importance and are the primary reason
tree-based models substantially outperform logistic regression on this dataset.
