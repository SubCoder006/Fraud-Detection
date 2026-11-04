# AI-Based Financial Transaction Fraud Detection — Finalized Process

Dataset: Fraud Detection Handbook simulated data — 1,754,155 rows, 9 columns, ~0.84% fraud rate, ~100 MB CSV.

---

## Phase 1: Data Prep & Validation

- Combine the 183 `.pkl` files → single CSV (done)
- Validate: nulls, dtypes, duplicate `TRANSACTION_ID`s, negative/zero `TX_AMOUNT`, timestamp continuity
- Set proper dtypes early (`category` for IDs, `datetime64` for `TX_DATETIME`) to control memory — don't let pandas infer object dtypes across 1.7M rows
- Sort the entire dataset by `TX_DATETIME` once, and keep it sorted through every downstream step — this is what makes rolling features and temporal splitting work correctly

## Phase 2: EDA

- Fraud rate, amount distributions (legit vs fraud), hour/day-of-week patterns
- Customer/terminal transaction volume distributions
- `TX_FRAUD_SCENARIO` breakdown (analysis only — never a feature)
- Check whether `TX_TIME_SECONDS` / `TX_TIME_DAYS` are redundant with `TX_DATETIME` before dropping

## Phase 3: Feature Engineering (the critical phase)

Build features **strictly using vectorized time-window operations**, not row-wise `.apply()`. A naive per-row lambda that filters transaction history is effectively O(n²) and will run for hours on 1.7M rows.

```python
df = df.set_index('TX_DATETIME').sort_index()

df['cust_txn_count_1h'] = (
    df.groupby('CUSTOMER_ID')['TX_AMOUNT']
      .rolling('1h').count().reset_index(level=0, drop=True)
)
```

Same pattern for `24h`, `7d` windows, per customer and per terminal.

**Feature set:**

- **Transaction-level:** `log_amount`, `hour`, `day_of_week`, `is_weekend`
- **Customer behavioral:** rolling counts/averages (1h / 24h / 7d), `time_since_last_txn`, `amount / customer_avg_amount`
- **Terminal behavioral:** rolling counts, terminal average amount, terminal historical fraud rate (computed leak-safely — only past data)
- Drop `TX_TIME_SECONDS` / `TX_TIME_DAYS` if confirmed redundant, drop `TRANSACTION_ID` and `TX_FRAUD_SCENARIO` from the feature matrix

**Leakage rule, made concrete:** every rolling/aggregate feature must be computed using `closed='left'` (or shifted by one) so the current row's own fraud outcome/amount never leaks into its own features.

## Phase 4: Temporal Split with a Delay Gap

```
[ TRAIN ] --- gap (e.g. 7 days) --- [ VALIDATION ] --- gap --- [ TEST ]
```

The gap simulates realistic fraud-label confirmation delay (this matches the actual Fraud Detection Handbook methodology) — worth reading that chapter before locking feature windows.

## Phase 5: Imbalance Handling

- **Logistic Regression / Random Forest:** `class_weight='balanced'`
- **XGBoost:** `scale_pos_weight = (n_legit / n_fraud)` ≈ 118 for this dataset — no SMOTE at this scale/dimensionality (synthetic oversampling on high-dimensional behavioral features tends to generate unrealistic transactions and gets slow/memory-heavy)
- Undersampling only as a lightweight baseline experiment, not for the main models

## Phase 6: Model Progression

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline, interpretable |
| Random Forest | Non-linear baseline |
| **XGBoost** (`hist`, `scale_pos_weight`) | Primary model |
| Neural Network | Comparison |
| Autoencoder (optional) | Unsupervised anomaly angle |

Tune XGBoost via randomized search over ~20–30 configs with **temporal CV** (expanding-window folds, not k-fold).

Starting config:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    tree_method="hist",
    scale_pos_weight=118
)
```

## Phase 7: Evaluation

- Precision, Recall, F1, ROC-AUC
- **PR-AUC as primary metric** (most informative given ~0.84% fraud rate)
- Confusion matrix, with explicit analysis of false negatives and false positives
- Simple **cost-based framing**: assumed $ loss per missed fraud vs. friction cost per false positive, to justify threshold choice in business terms, not just statistically

## Phase 8: Threshold Selection

Pick threshold from the PR curve using cost assumptions, not a flat 0.5.

## Phase 9: Explainability

- `TreeExplainer` (SHAP) on XGBoost
- Global importance: sample 10k–50k rows, not all 1.7M
- Per-transaction explanation: single-row SHAP, fast enough for live app use

## Phase 10: Application

Streamlit dashboard: input/select a transaction → fraud probability → top SHAP contributing factors.

---

## Technology Stack

- **Data Science:** Python, Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Deep Learning:** PyTorch or TensorFlow/Keras (one is sufficient)
- **Explainability:** SHAP
- **Application:** Streamlit (initial); FastAPI + Next.js/React (advanced, optional)
- **Deployment:** Docker (optional)

---

## Timing Estimates (Ryzen 5 5600H, 16GB RAM, RTX 3050)

| Task | Estimated time |
|---|---|
| Feature engineering (vectorized, `groupby().rolling()`) | ~5–15 min |
| Feature engineering (naive `.apply()`) | 30+ min, possibly hours — avoid this |
| Single XGBoost fit (~20–50 features, 300 trees, `hist`) | ~1–5 min |
| Full hyperparameter search (20–30 configs, temporal CV) | ~30 min–2 hrs |
| SHAP (sampled, 10k–50k rows) | a few minutes |

**Bottom line:** feature engineering correctness — not XGBoost training speed — is where this project lives or dies. GPU (`gpu_hist`) is not necessary at this data scale.

---

## Key Decisions Locked In

- **Target:** `TX_FRAUD`
- **Never a feature:** `TX_FRAUD_SCENARIO` (target leakage), `TRANSACTION_ID`
- **Split:** temporal, with a delay gap between train/validation/test
- **Primary metric:** PR-AUC (not accuracy)
- **Imbalance method:** class weighting (`scale_pos_weight` / `class_weight='balanced'`), not SMOTE
- **Final objective:** build a system that generalizes to future transactions, maximizes fraud detection, controls false positives, and produces interpretable risk scores — not simply the highest accuracy