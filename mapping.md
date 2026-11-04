# Fraud Detection: What Maps to What (final)

Math renders in VS Code's Markdown preview (KaTeX). Open with `Ctrl+Shift+V`.

**Final model:** XGBoost, 20 features, threshold 0.7. Test PR-AUC 0.6822, precision 0.679, recall 0.674.

---

## 1. Fraud scenario → feature → SHAP → recall

| Scenario | Fraud mechanism (simulator) | Features that catch it | SHAP evidence | Recall before terminal-risk (val) | Recall, final model (test, 0.7) |
|---|---|---|---|---|---|
| 1 | Amount above a fixed threshold (about 220) | `TX_AMOUNT`, `log_amount` | `TX_AMOUNT` rank 3 | 99.3% | 100% |
| 2 | Terminal compromised for about 4 weeks | `term_risk_1d/7d/28d`, trend features | `term_risk_7d` rank 1 | 3.3% | 66.0% |
| 3 | Customer amounts multiplied about 5x | `amount_vs_cust_avg`, `cust_avg_amount_7d` | `amount_vs_cust_avg` rank 2 | 80.8% | 63.7% |

Scenario details come from the Fraud Detection Handbook; verify exact numbers in that chapter.
`TX_FRAUD_SCENARIO` is used for analysis only, never as an input feature.
The "before" column is an earlier model at threshold 0.5, so it is not directly comparable to the final column.

---

## 2. Feature → formula → what it captures (20 features)

| Feature | Formula | Captures |
|---|---|---|
| `log_amount` | $\log(1 + a_t)$ | Compresses skewed amounts |
| `cust_txn_count_w` | $\sum_{s \in [t-w,\, t)} 1$ per customer | Customer activity burst |
| `cust_avg_amount_7d` | $\bar{a}_{7d} = \frac{1}{n}\sum_{s \in [t-7d,\, t)} a_s$ | Customer's normal spend |
| `amount_vs_cust_avg` | $\dfrac{a_t}{\bar{a}_{7d}}$ | Is this amount unusual for this person |
| `time_since_cust_prev_txn` | $t_i - t_{i-1}$ (seconds, same customer) | Rapid-fire activity |
| `term_txn_count_w` | same as customer count, grouped by terminal | Terminal traffic spike |
| `term_avg_amount_7d` | same as customer average, grouped by terminal | Terminal's normal amount |
| `term_risk_w` | see Section 3 | Terminal's confirmed fraud rate |
| `term_risk_trend_1_7` | $\text{risk}_{1d} - \text{risk}_{7d}$ | Terminal risk rising or falling |
| `term_risk_trend_7_28` | $\text{risk}_{7d} - \text{risk}_{28d}$ | Same, one step older |

Other inputs: `TX_AMOUNT`, `hour`, `day_of_week`, `is_weekend`.
The window $[t-w,\, t)$ is closed on the left and open on the right (`closed="left"`), so the current transaction is never counted in its own features.

---

## 3. Delayed terminal risk (leak-safe)

For a transaction at time $t$ on terminal $k$, with delay $\delta = 7$ days and window $w$:

$$
\text{risk}_{k}(t, w) =
\frac{\text{fraud count in } [\,t-\delta-w,\; t-\delta\,)}
     {\text{transaction count in } [\,t-\delta-w,\; t-\delta\,)}
$$

If the denominator is $0$, the feature is set to $0$.

| Term | Meaning |
|---|---|
| $\delta$ | Label confirmation delay: labels newer than $t-\delta$ are unknown in real life |
| $w \in \{1, 7, 28\}$ | Look-back length beyond the delay |
| Effect | First week of a compromise stays hard to catch, so Scenario 2 recall cannot reach 100% |

---

## 4. Handling missing history (cold start)

| Feature group | Why NaN | Fill |
|---|---|---|
| Count features | No prior transactions | $0$ (literally true) |
| `time_since_cust_prev_txn` | No previous transaction | Column maximum (a very long gap) |
| Average amounts | No history to average | The transaction's own amount, so ratio is $1.0$ |

---

## 5. Split strategy

| Set | Dates | Rows | Fraud rate |
|---|---|---|---|
| Train | before 2018-08-01 | 1,169,694 | 0.809% |
| Gap | 7 days | none | none |
| Validation | 2018-08-08 to 2018-09-01 | 229,581 | 0.890% |
| Gap | 7 days | none | none |
| Test | from 2018-09-08 | 220,879 | 0.882% |

A random or stratified split is not used: features look backward in time, and fraud clusters on compromised terminals, so shuffling would leak future patterns into training and inflate scores. The temporal split already keeps the fraud rate stable across sets.

---

## 6. Model progression

| Step | Model / change | Val PR-AUC | Val ROC-AUC | Note |
|---|---|---|---|---|
| 1 | Logistic Regression | 0.2228 | 0.6597 | Linear baseline |
| 2 | Random Forest | 0.2931 | 0.6504 | Catches the Scenario 1 rule |
| 3 | XGBoost, 15 features | 0.2806 | 0.6480 | Same ceiling as RF: features were the limit |
| 4 | XGBoost + `term_risk_*` (18 features) | 0.6104 | 0.8726 | Best iteration 0: overfitting |
| 5 | Regularized XGBoost (depth 4, soft class weight) | 0.6683 | 0.8818 | Best iteration 318 |
| 6 | + trend features (20 features) | 0.6836 | 0.8798 | 15% fewer false alarms on test |
| 7 | Neural network (128-64-32) | 0.6979 | 0.8768 | Leads XGBoost on PR-AUC |
| 8 | Autoencoder (anomaly detection) | 0.1569 | 0.7764 | Weak alone; see Section 9 |

Step 3 used 15 features (before the three `term_risk_*` columns were added); step 4 used 18; steps 6 to 8 use 20.

---

## 7. Imbalance handling

$$
\text{spw} = \frac{n_{\text{legit}}}{n_{\text{fraud}}} \approx 122.6
$$

| Setting | Value | Result |
|---|---|---|
| Full weight | $\text{spw} \approx 122.6$ | PR-AUC 0.6082 (depth 3) |
| Soft weight | $\sqrt{\text{spw}} \approx 11$ | PR-AUC 0.6648 (depth 3), 0.6683 (depth 4) |

The neural network uses the same soft weight (about 11.1).

---

## 8. Metrics

$$
\text{Precision} = \frac{TP}{TP + FP}
\qquad
\text{Recall} = \frac{TP}{TP + FN}
\qquad
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

| Metric | Use here |
|---|---|
| PR-AUC | Primary metric (fraud is about 0.84% of rows) |
| ROC-AUC | Ranking quality |
| Recall | Missed fraud is costly |
| Precision | Controls false alarms |
| Accuracy | Not used (always-legit scores about 99.2%) |

---

## 9. Neural network and autoencoder

**Neural network vs XGBoost (validation PR-AUC, 4 seeds each):**

| Model | Mean | Std | Range |
|---|---|---|---|
| Neural network | 0.6982 | 0.0003 | 0.6979 to 0.6986 |
| XGBoost | 0.6843 | 0.0008 | 0.6834 to 0.6853 |

The gap is about 0.014 and every network run beat every XGBoost run, so it is not seed luck. Four seeds is a small sample.

**Precision at fixed recall (validation):**

| Model | PR-AUC | prec at recall 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|
| XGBoost | 0.6836 | 0.920 | 0.884 | 0.490 |
| Neural network | 0.6979 | 0.989 | 0.933 | 0.530 |
| Average of both | 0.6968 | 0.968 | 0.918 | 0.520 |

Averaging the two did not beat the network alone.

**Autoencoder** (trained on legitimate rows only; score is reconstruction error). ROC-AUC per fraud type, each type versus legit:

| Scenario | Autoencoder | Neural net | XGBoost |
|---|---|---|---|
| 1 (large amount) | 0.533 | 1.000 | 0.999 |
| 2 (compromised terminal) | 0.829 | 0.845 | 0.844 |
| 3 (customer spike) | 0.719 | 0.917 | 0.928 |

The autoencoder is near chance on large-amount fraud. Its Scenario 2 result is strong, but its inputs include `term_risk_*`, which are built from past fraud labels, so it is unsupervised in training and not label-free in practice. It had not stopped improving after 30 epochs.

---

## 10. Threshold and cost

$$
\text{Cost}(\tau) = \sum_{i \in \text{missed fraud}} a_i \;+\; c_{FP} \cdot FP(\tau)
$$

Threshold sensitivity, validation, earlier 18-feature XGBoost:

| False-alarm cost $c_{FP}$ | Best threshold $\tau$ | Cost reduction vs flag-nothing |
|---|---|---|
| 1 | 0.07 | 87% |
| 5 | 0.60 | 83% |
| 20 | 0.74 | 79% |
| 50 | 0.78 | 71% |
| 100 | 0.93 | 63% |

Locked operating threshold for the deployed model: $\tau = 0.7$. The cost curve is flat between about 0.1 and 0.8.
The amounts in this simulated data have no real currency, so costs are for comparison only.

---

## 11. Final test results

**Deployed model (XGBoost, 20 features) at $\tau = 0.7$:**

| | Predicted legit | Predicted fraud |
|---|---|---|
| Actual legit | 218,310 (TN) | 621 (FP) |
| Actual fraud | 636 (FN) | 1,312 (TP) |

| PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.6822 | 0.8775 | 0.679 | 0.674 | 0.676 |

**Head-to-head on test** (each model's threshold chosen on validation by cost):

| | XGBoost | Neural network |
|---|---|---|
| PR-AUC | 0.6822 | 0.6963 |
| ROC-AUC | 0.8775 | 0.8756 |
| Cost, false alarm = 5 | 44,766 | 44,435 |
| Cost, false alarm = 20 | 54,495 | 55,809 |
| No detection | 248,292 | 248,292 |

**Decision:** XGBoost stays deployed. The network has the higher PR-AUC, but cost is a tie (within about 1% to 2%, winning at different prices), and the app and SHAP explanations are built on XGBoost's fast tree explainer.
The test set was used several times (original model, trend-feature model, final head-to-head), so no further model decisions should be made from it.

---

## 12. SHAP

$$
f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j
$$

Each $\phi_j$ is the push feature $j$ gives to the prediction for one transaction. Positive means toward fraud, negative toward legit (log-odds units).

Ranking below is from the earlier 18-feature model and was not recomputed for the final 20-feature model:

| Rank | Feature | Mean absolute SHAP | Group |
|---|---|---|---|
| 1 | `term_risk_7d` | 0.4166 | Terminal risk |
| 2 | `amount_vs_cust_avg` | 0.2947 | Amount deviation |
| 3 | `TX_AMOUNT` | 0.2553 | Amount |
| 4 | `cust_avg_amount_7d` | 0.1594 | Customer baseline |
| 5 | `term_risk_28d` | 0.1183 | Terminal risk |
| 6 | `cust_txn_count_7d` | 0.0934 | Customer activity |
| 7 | `term_avg_amount_7d` | 0.0821 | Terminal baseline |
| 8 | `term_risk_1d` | 0.0594 | Terminal risk |
| 9 to 18 | remaining features | below 0.04 each | `is_weekend` (0.0013) and `term_txn_count_1h` (0.0007) are near zero |

Notes, marked as hypotheses (not verified):

- `cust_avg_amount_7d` ranking high may partly reflect inflated fraud amounts feeding the customer's own average.
- `cust_txn_count_7d` low values pushing strongly toward legit may be the cold-start rows (count filled with 0), which are rarely fraud.
- Correlated features share credit (the three `term_risk_*` columns, and the amount columns), so rank each one as part of its group.
- SHAP shows what the model relies on, not what causes fraud.

**Single-transaction examples (waterfall plots):**

| | Caught fraud | False alarm |
|---|---|---|
| Final risk score | 98.6% | 88.2% |
| `term_risk_7d` / `28d` / `1d` | 0.556 / 0.194 / 1.000 | 0.500 / 0.118 / 0.000 |
| Driver | Terminal risk across all three windows | Terminal risk in the 7-day window only |

Why that false alarm happened is unconfirmed: a check on validation found similar shares of `term_risk_1d == 0` among false alarms (0.618) and true catches (0.587).

---

## 13. Main findings

| Finding | Evidence |
|---|---|
| Features mattered far more than model choice | Delayed terminal risk moved PR-AUC by about 0.33; swapping XGBoost for a neural net moved it by 0.014 |
| Each top feature serves one fraud scenario | Terminal risk, amount ratio and raw amount are SHAP ranks 1 to 3 |
| Soft class weight beat the full ratio | $\sqrt{\text{spw}}$ gave +0.056 PR-AUC at depth 3 |
| Temporal split is required | Features look backward and fraud clusters by terminal |

---

## 14. Remaining work

| Step | Status |
|---|---|
| Streamlit app (Guide and Live testing pages) | Built; tested on synthetic artifacts, not yet on real ones |
| `artifacts/test_metrics.json` for the app's Performance tab | Run the notebook block; until then the app shows typed-in values |
| Recompute SHAP ranking for the 20-feature model | Pending |
| Longer customer baselines (to lift Scenario 3 recall) | Optional improvement |
| Cross-validated threshold and confidence intervals | Optional |