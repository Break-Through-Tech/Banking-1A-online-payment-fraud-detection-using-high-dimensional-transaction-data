# Business Objective, Assumptions, and Evaluation Framework

Task #13 · Milestone 1 (September) · Owner: Mohit

This document fixes what "a good model" means for Banking-1A before we compare any models. Every baseline in September and every model in October and November is judged against the framework below, using the shared evaluation module (Task #7) and the frozen chronological split (Task #6).

## 1. Business objective

A card-not-present (online) payment provider wants to stop fraudulent transactions before they settle, without blocking or delaying so many legitimate customers that it damages revenue and trust.

The model's job is to rank transactions by fraud risk so that a fixed-capacity fraud operations team, or an automated decline rule, can act on the riskiest ones first. The objective is therefore:

> Maximize the fraud caught, by count and by dollar amount, within the review volume a fraud team can realistically absorb, while keeping the false positive rate low enough that legitimate customers are rarely affected.

This is a ranking problem first and a classification problem second. A threshold is only chosen at the end, based on review capacity and cost.

## 2. Target definition

| Item | Definition |
|---|---|
| Target | `isFraud` from `train_transaction.csv` (1 = fraud, 0 = legitimate) |
| Unit of prediction | One transaction (`TransactionID`) |
| Prediction time | At authorization, using only information available at or before `TransactionDT` |
| Labeled rows | 590,540 transactions, about 3.5% fraud (20,663) |
| Out of scope | Kaggle `test_*.csv` files, which have no labels |

How the label was produced matters. Vesta, the data provider, explained on the competition forum that a transaction is labeled fraud when a chargeback is reported on the card, and that later transactions linked to the same user account, email address, or billing address are labeled fraud as well. A transaction is treated as legitimate if no chargeback is reported within about 120 days.

Three consequences for us:

1. **The label is about the account, not only the single transaction.** Once a client is flagged, their later transactions tend to be flagged too. Models can score well by recognizing a client they have already seen labeled as fraud. That is a real signal in production (the card is already compromised), but it inflates offline scores when the same client appears on both sides of a split.
2. **Labels arrive late.** In a real deployment, chargebacks show up weeks after the transaction. Our chronological split ignores this delay, so our results will be somewhat optimistic. We record this as a known limitation instead of trying to simulate it in September.
3. **Some "legitimate" labels are really unreported fraud.** Label noise is concentrated in the negative class. Precision estimates are a lower bound on true precision.

## 3. Cost asymmetry

The two errors do not cost the same.

| Outcome | What happens | Rough cost driver |
|---|---|---|
| Missed fraud (false negative) | Chargeback, lost goods, network fees, possible scheme penalties | Roughly the full transaction amount plus a fixed chargeback fee |
| False alarm (false positive) | Manual review cost, or a declined good customer | Analyst time per review, plus lost margin and churn risk if declined |
| Caught fraud (true positive) | Loss avoided | Saves the transaction amount, costs one review |
| Correct approval (true negative) | Normal sale | No cost |

Working assumptions for analysis (to confirm with our Challenge Advisor):

- **Cost of a missed fraud** = `TransactionAmt` + a fixed fee `F_cb`. We use `F_cb = $20` as a placeholder.
- **Cost of a false positive sent to review** = `C_review`. We use `C_review = $5` as a placeholder for analyst time.
- **Cost of a false positive that is auto-declined** = `C_decline`, set higher than `C_review` because it loses the sale and can lose the customer. Placeholder: `max($15, 0.10 × TransactionAmt)`.

With these placeholders a missed $100 fraud costs about $120, while a false review costs about $5. That is roughly a 24 to 1 ratio, which is why we value recall highly but still cap review volume. We will run sensitivity checks with the ratio at 10:1, 25:1, and 50:1 rather than trusting one number.

Because the cost of missed fraud scales with the amount, we report **fraud dollars captured** next to fraud counts. A model that catches many small frauds but misses large ones can look good on recall and still lose money.

## 4. Review capacity assumption

A fraud operations team cannot review every flagged transaction. We assume:

- **Primary operating point:** the team reviews the riskiest **1%** of transactions.
- **Secondary operating point:** **5%**, representing a larger team or a step-up check (for example, 3-D Secure or an OTP) instead of a manual review.
- **Customer friction limit:** the false positive rate at the chosen threshold should stay at or below **1%** of legitimate transactions.

At our validation volume (about 30 days and 90,000 to 100,000 transactions), a 1% review rate means about 30 reviews a day. That is a plausible load for a small team and keeps the numbers concrete. These rates are assumptions to be confirmed with the Challenge Advisor.

## 5. Evaluation framework

### 5.1 Data split

| Split | Days (from first `TransactionDT`) | Use |
|---|---|---|
| Train | 0 to 121 | Fitting models and any encoders or scalers |
| Validation | 122 to 151 | Early stopping, model choice, threshold choice |
| Blind test | 152 to 182 | Sealed until November. Scored once per finalist model |

Rules:

- The split is chronological, never random, so we measure performance on future transactions.
- Encoders, imputers, scalers, and feature selection are fitted on train only.
- Feature engineering that aggregates over time (counts, means, time since last transaction) may only use rows earlier than the current transaction.
- Nobody tunes on the blind test. If a test result is used to change the model, that test set is no longer blind and we report that.

### 5.2 Metrics

All metrics come from the shared evaluation module (Task #7), computed on the validation split.

| Priority | Metric | Why |
|---|---|---|
| Primary | **AUPRC** (average precision) | Ranks models on the minority class and is not inflated by the 96.5% legitimate majority |
| Primary | **Recall at 1% review rate** | Fraud caught at our primary operating point |
| Primary | **Fraud dollar share in top 1% and 5%** | Links ranking quality to money saved |
| Secondary | Recall at 5% review rate | Second operating point |
| Secondary | Precision, recall, F1, FPR at the chosen threshold | Operational view of one decision rule |
| Secondary | KS statistic | Score separation, standard in credit and fraud risk |
| Secondary | ROC-AUC | Comparable with Kaggle results and published work, but reported second because it looks strong on imbalanced data |
| Diagnostic | Confusion matrix, PR curve, score distribution | Error analysis and sanity checks |

### 5.3 Threshold selection

Ranking metrics are compared first. A decision threshold is picked only for the model we carry forward:

1. Find the threshold at the 1% review rate on validation.
2. Check that FPR at that threshold is at most 1%.
3. Report expected cost on validation using the cost assumptions in section 3, at the 10:1, 25:1, and 50:1 ratios.

### 5.4 What counts as an improvement

A new model replaces the current best only if, on the same validation rows:

- AUPRC improves by at least **0.01** absolute, **and**
- Recall at 1% review does not get worse, **and**
- The improvement holds across at least 3 random seeds, or the change is larger than the seed-to-seed spread.

Small differences inside seed noise are reported as ties.

### 5.5 Reproducibility

Each run is logged to the results registry (Task #9) with model, feature set version, hyperparameters, split identifier, seed, git commit, and the full metric set. A result that is not in the registry is not used in a comparison.

## 6. Assumptions and open questions for the Challenge Advisor

1. Are the 1% and 5% review rates and the 1% FPR limit realistic for a payment network or issuer, or should we use different numbers?
2. Is an auto-decline scenario in scope, or should we only model "send to review"?
3. What fixed chargeback cost and review cost should we assume?
4. Should fairness checks cover segments we can see (ProductCD, DeviceType, card type, email domain), given there are no demographic fields?
5. How should we treat repeat transactions from a client already labeled fraud: count them as wins or report them separately?

## 7. Known limitations

- Most features are anonymized (V, C, D, M, id_ groups), so business explanations of model drivers will be limited.
- No label delay is simulated. Real-world performance will be lower than offline results.
- The dataset comes from one merchant network over about six months, so seasonality and longer drift are not visible.
- `TransactionAmt` has no currency field. We treat all amounts as US dollars when reporting dollar capture.
