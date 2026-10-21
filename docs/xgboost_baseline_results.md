# XGBoost Baseline Results (Task #19)

Owner: Mohit · Run on October 7, 2026 on the real IEEE-CIS training data · Split `chrono-v1-c3eab22dc2`

Both runs are recorded in `results/registry.jsonl`. All metrics are on the validation split: 85,044 transactions, days 122 to 151, 3.39% fraud. The blind test split was not touched.

## Setup

| Setting | Value |
|---|---|
| Training rows | 420,066 (days 0 to 121, 3.52% fraud) |
| Imbalance handling | `scale_pos_weight` = 27.4 (legitimate ÷ fraud in train) |
| Trees | `hist`, depth 8, learning rate 0.1, subsample 0.8, colsample 0.5 |
| Stopping | Early stopping on validation AUPRC, patience 100, cap 3,000 rounds |
| Categoricals | Native XGBoost categorical splits, categories learned on train only |
| Hardware | 2 CPU cores, 8 GB RAM. Peak memory about 5.3 GB |

Learning rate 0.1 rather than the 0.05 default to fit the compute budget. On a faster machine, 0.05 with more rounds is worth trying.

## Results

| | `raw_all_v1` | `fe_v2` | Change |
|---|---|---|---|
| Features | 432 raw columns | 445 (raw + engineered) | |
| **AUPRC** | 0.620 | **0.664** | **+0.044** |
| ROC-AUC | 0.917 | 0.913 | −0.003 |
| KS | 0.703 | 0.710 | +0.007 |
| Fraud caught, top 1% reviewed | 27.8% | 28.2% | +0.4 pts |
| Fraud caught, top 5% reviewed | 65.8% | 69.8% | +4.0 pts |
| Legit customers flagged at 1% review | 0.06% | 0.05% | |
| Precision / recall at 0.5 | 0.836 / 0.427 | 0.887 / 0.455 | |
| Best round | 3,000 (hit the cap) | 1,752 | |
| Training time | 17.5 min | 10.4 min | |

For comparison, LightGBM on `raw_all_v1` (Task #18 test run) reached AUPRC 0.640. Random guessing scores 0.034.

### Fraud dollar capture (`fe_v2`)

| Riskiest transactions reviewed | Transactions | Fraud cases found | Share of fraud cases | Share of fraud dollars | Precision |
|---|---|---|---|---|---|
| Top 1% | 851 | 814 | 28.2% | 22.6% | 95.7% |
| Top 5% | 4,253 | 2,014 | 69.8% | 67.2% | 47.4% |
| Top 10% | 8,505 | 2,266 | 78.6% | 76.1% | 26.6% |
| Top 20% | 17,009 | 2,459 | 85.3% | 84.1% | 14.5% |

At the 1% review rate in our business framework (about 28 reviews a day), 96% of reviewed transactions are fraud. The dollar share trails the case share in every bucket, so the model catches small frauds a little more easily than large ones.

## What `fe_v2` adds

All statistics come from the train split only, and none use the label.

1. **Frequency encodings** for card1, card2, card3, card5, addr1, addr2, both email domains, DeviceInfo and id_31. Values not seen in training get 0. In the raw run, card1 was treated as a plain number, so splits like "card1 > 9500" meant nothing.
2. **Normalized D columns.** D1, D2, D4, D10, D11 and D15 count days since a past event, so they grow with calendar time. They are replaced by `day − D`, a fixed point in time (literature review, section 1.4).
3. **Time and amount features:** hour of day, the cents part of the amount, and log amount.
4. **`max_cat_threshold = 32`**, which limits how many categories one split can group, so the model can't memorize rare DeviceInfo or email values.

The raw run had nearly perfect training AUPRC by round 1,000 while validation crept up slowly, a sign of overfitting and drift. `fe_v2` reached a higher validation score in fewer rounds.

## Top features (`fe_v2`, share of total gain)

V258 (6.8%), P_emaildomain, R_emaildomain, DeviceInfo, D1_norm, id_31, card1_freq, D15_norm, C13, card1, D2_norm, TransactionAmt, C14, V294, addr1_freq.

Three of the new features (D1_norm, card1_freq, D15_norm) are in the top 8. id_31 (browser version) is still high even though it is the most drifting feature in the data, so it is a candidate to drop or group by browser family in October.

## Caveats

- **One seed per run.** Our framework (business objective, section 5.4) asks for 3 seeds before calling a winner. The +0.044 gain is well above the 0.01 bar, but it should be confirmed.
- **All four `fe_v2` changes went in together.** We don't yet know which one drives the gain. That ablation is cheap to run in October.
- **No customer-level features yet.** The customer-ID check (literature review, section 4) says these are feasible and probably the biggest remaining gain. They must use only earlier transactions.
- **Results are optimistic** for the reasons in the business objective, section 2: chargeback labels arrive with a delay in real life, and labels spread across a customer's transactions.

## Reproduce

```
python -m fraud.models.xgboost_baseline \
    --learning-rate 0.1 --feature-set fe_v2 --report-dir reports/xgboost
```

## Note on the top 1% column

The top 1% of validation is 851 transactions and validation has 2,884 frauds, so no model can catch more than 29.5% of fraud at that review rate. Both runs are close to that ceiling (27.8% and 28.2%), which is why the 5% column and AUPRC separate them better.

The full write-up with EDA charts is in `docs/reports/milestone1_eda_and_baselines.pdf`.
