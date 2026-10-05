# Literature and Prior Approach Review

Task #14 · Milestone 1 (September) · Owners: Nicholas, Mohit

Summary of published IEEE-CIS approaches, the leakage traps reported in them, and what we take from each. Section 5 lists open questions for the biweekly Challenge Advisor check-in.

> Nicholas: sections 1 to 4 are a first pass. Please add anything from your reading, especially academic papers on MLPs or sequence models for October.

## 1. What the top Kaggle solutions did

The IEEE-CIS competition (Vesta, 2019) was scored on ROC-AUC against a test set from a later time period. The best teams mostly differed in how they handled time and client identity, not in model choice.

### 1.1 First place: client identification ("magic UID")

Chris Deotte and Konstantin Yakovlev won (private leaderboard AUC 0.9459). Their main finding was that the data contains no client ID, but one can be rebuilt:

```python
day = TransactionDT / 86400
UID = card1 + "_" + addr1 + "_" + floor(day - D1)
```

`D1` behaves like "days since the card was first used", so `day - D1` is roughly constant for one card. Combined with `card1` (card number group) and `addr1` (billing region), it approximates one client.

What they did with it:

- **Did not train on UID directly.** It does not generalize, because test clients are new.
- **Built aggregates over UID**, such as mean and std of `TransactionAmt`, counts, and `nunique` of email domain, device, and address. These let a tree model learn "this client's transactions all look alike", which matters because fraud labels propagate across a client (see section 2.1).
- **Frequency encoding** (how common a value is) and **aggregation encoding** (for example, mean `TransactionAmt` per `card1`).
- **Post-processing:** averaging predictions across all transactions with the same UID.
- **Validation:** train on earlier months, validate on later ones (GroupKFold by month), never random folds.
- **Models:** an ensemble of XGBoost, LightGBM, and CatBoost.

Local validation AUC rose to about 0.947 after UID features.

### 1.2 Feature selection by time consistency

The winning team also checked each feature for **time consistency**: train a model on the first month using only that feature, then score it on the last month. Features that score well in-sample but drop to chance out-of-time are dropped, because they describe something that changes over time instead of fraud. We should do the same before trusting any feature in our model.

### 1.3 V column reduction

The 339 V columns come in blocks that share the same missing-value pattern. Common practice was to group V columns by missingness pattern, then keep one column from each highly correlated subgroup (for example, correlation above 0.75). This cuts the V set to roughly a third with little or no AUC loss. This links directly to Task #8 (missing value analysis).

### 1.4 D column normalization

Many D columns are "days since some event", so they grow with time. A raw `D15` value means something different in month 1 and month 6. The standard fix is `D_n = day - D_n`, which turns a time delta into a fixed point in time. Without this, the model learns calendar position instead of behavior, and performance drops on future data.

### 1.5 A top 5% public write-up

A public top 5% XGBoost solution reported this progression of validation AUC: plain XGBoost about 0.923, after normalizing D columns about 0.934, after encoding features and UID aggregates about 0.948 to 0.951. It also used month-grouped folds. The order of gains is useful as a roadmap: time handling first, then encodings, then client aggregates.

## 2. Leakage traps reported in prior work

These are the ones most likely to make our offline numbers look better than they would be in production.

| # | Trap | How it leaks | What we do |
|---|---|---|---|
| 1 | **Random train/test split** | Fraud labels spread across a client's transactions, so random splits put the same client in both sets. Public benchmarks note that random KFold inflates AUC. | Chronological split only (Task #6) |
| 2 | **Transductive aggregates** | Kaggle solutions computed UID counts and means over train and test together, so a row's feature uses future rows. A local benchmark that rebuilt the magic UID found that switching to strictly causal (past-only) UID statistics lowered AUC from about 0.946 to about 0.918. | All aggregates use only earlier rows. Report both versions if we try UID features |
| 3 | **Encoders fitted on all data** | Frequency or target encoding computed on the full dataset sees validation and test distributions. | Fit on train only. `CategoricalEncoder` (PR #3) already does this |
| 4 | **Unnormalized D columns** | The model learns time position. Out-of-time performance drops. | Normalize D columns in October feature work |
| 5 | **Target encoding without out-of-fold** | The row's own label leaks into its feature. | Use out-of-fold or time-ordered target encoding, or avoid it |
| 6 | **TransactionID and TransactionDT as features** | Both are monotonic in time, so they act as a proxy for date. | Excluded from features in the XGBoost baseline |
| 7 | **Tuning on the blind test set** | Repeated scoring makes the "blind" set part of training. | Test sealed until November (Task #6) |
| 8 | **Label propagation** | Once a client is flagged, later transactions from the same account, email, or address are also labeled fraud. A model can "recognize" a client instead of detecting fraud. | Report results for first-seen clients separately once UID is feasible (Task #15) |

## 3. Takeaways for our plan

1. **Our scores will be lower than Kaggle scores, and that is expected.** Kaggle numbers use transductive features and ROC-AUC. We use causal features and AUPRC first. We should say this clearly in the final report so the Challenge Advisor compares like with like.
2. **Client grouping is the biggest single lever** (about +0.01 AUC in public benchmarks). Task #15 decides whether we can build a reliable UID. If we can, October behavioral features and any sequence model (GRU, LSTM, Transformer) depend on it.
3. **Gradient boosted trees are the bar to beat.** Every strong public solution is tree-based. The October MLP should be judged against the best September tree baseline under the same features and split.
4. **Run the time-consistency check** on every feature group before final feature selection.
5. **Reduce the V columns** by missingness block and correlation. It speeds training and is needed anyway for the MLP.

## 4. Sources

- Deotte, C. and Yakovlev, K. *1st Place Solution, Part 2*. Kaggle IEEE-CIS Fraud Detection writeups. https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2
- NVIDIA Technical Blog. *Leveraging Machine Learning to Detect Fraud: Tips to Developing a Winning Kaggle Solution* (UID formula, encodings, month-based validation, scores). https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
- Vesta. *Data Description (Details and Discussion)*, Kaggle forum thread including the labeling logic. https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203
- Arun M. *IEEE-CIS Fraud Detection: Top 5% Solution*. Towards Data Science. https://towardsdatascience.com/ieee-cis-fraud-detection-top-5-solution-5488fc66e95f/
- Local benchmark of top IEEE-CIS solutions, with a magic UID ablation and causal vs transductive comparison. https://github.com/Edgar100800/ieee-cis-fraud-detection-local-benchmark
- Mishra, P. *A realistic approach to Kaggle's IEEE-CIS Fraud Detection Challenge* (adversarial validation, avoiding test leakage). https://medium.com/@mr.priyankmishra/a-realistic-approach-to-ieee-cis-fraud-detection-25faea54137

## 5. Open questions for the Challenge Advisor check-in

1. Is a reconstructed client ID (card1 + addr1 + D1) acceptable for features if we only use past transactions, or does Mastercard consider any client reconstruction out of scope?
2. Should transactions from clients already labeled fraud be scored, excluded, or reported separately?
3. Is ROC-AUC worth reporting next to AUPRC so results can be compared with Kaggle benchmarks?
4. For the October sequence models, what minimum history length per client would make a sequence model worth trying?
5. Is there a preferred way to handle label delay (chargebacks arriving weeks later) in the evaluation, or is it acceptable to note it as a limitation?
