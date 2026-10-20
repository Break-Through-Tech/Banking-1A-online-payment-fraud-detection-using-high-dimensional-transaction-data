# October Milestone Review and November Handoff: DRAFT

Task #23 · Draft by Mohit for the whole team · Status as of October 20, 2026

> **Draft.** Sections marked _owner to fill_ need results from the task owner before the October 31 review. Numbers here come from `results/` on `main`. Everything is on the frozen validation split (`chrono-v1-c3eab22dc2`); the blind test is still sealed.

## 1. Where the models stand

| Model | Task | Owner | Validation AUPRC | Fraud caught, top 5% | Status |
|---|---|---|---|---|---|
| **XGBoost, tuned** | #8 | Mohit | **0.682 ± 0.001** (3 seeds) | **71.2%** | Done |
| LightGBM, tuned | #10 | Orator | 0.668 | 69.8% | Done (PR #38) |
| XGBoost, fe_v2 (Sept #19) | | Mohit | 0.664 | 69.8% | Done |
| LightGBM, native categorical | #4 | Orator | 0.662 | 69.3% | Done (PR #36) |
| CatBoost | #3 / #11 | Faizah | _owner to fill_ | | To-do |
| Logistic regression / random forest | #5 / #12 | Refilwe | _owner to fill_ | | In progress |
| **MLP (focal loss)** | #14 | Mohit | **0.536 ± 0.008** (3 seeds) | 60.1% | Done |
| FT-Transformer (or TabNet) | #15 | Orator | _owner to fill_ | | To-do |
| Ensemble / stacking | #17 | Leo | _owner to fill_ | | To-do |

Random guessing scores 0.034. At the 1% review rate every tree model catches about 28% of fraud, which is close to the 29.5% ceiling (851 reviews, 2,884 frauds).

**Proposed strongest configuration (to confirm at the review):** tuned XGBoost on fe_v2 features, ranked by expected loss at a 1% daily review budget. It catches 48% of fraud dollars with 28 reviews a day and 65% precision, and saves an estimated $226k over 30 days under our placeholder costs. Revisit if the ensemble (#17) or CatBoost (#11) beats 0.682 by at least 0.01 AUPRC across 3 seeds.

## 2. What we learned in October

1. **Tuning helped XGBoost by +0.018 AUPRC**, with deeper trees, much stronger L2/L1 regularization, fewer categories per split and about half the full class weight (`docs/xgboost_tuning_results.md`).
2. **Deep learning trails the trees**, as the literature predicted (`docs/dl_literature_review.md`). The MLP is 0.146 AUPRC behind and adds only 0.002 in a blend. Focal loss beat weighted cross-entropy.
3. **How we rank matters as much as the model.** At the same 28 reviews a day, ranking by expected loss instead of probability more than doubles fraud dollars caught, from 21% to 48% (`docs/threshold_calibration.md`).
4. **Alerts are driven by velocity.** Counts linking cards, addresses and accounts (the C columns) are 60% of the top reasons for flagged transactions. Anonymized V columns carry 24% of total model influence, the main limit on explainability (`docs/shap_attribution.md`).
5. **Missed fraud is mostly product W with no identity data** (27% have identity vs 59% of caught fraud). Customer-history features are the obvious next step.
6. **Performance drifts within one month.** Fraud dollars caught at 5% fall from 83% to 71% between the first and second half of validation.

## 3. Task status

| # | Task | Owner | Status | Notes |
|---|---|---|---|---|
| 1, 2 | Sept carry-overs: encoding findings, distribution shift | Faizah | _owner to fill_ | |
| 3, 11, 21 | CatBoost baseline, tuning, rare-category pass | Faizah | _owner to fill_ | |
| 4, 10 | LightGBM baseline and tuning | Orator | Done | Merged in PRs #36 and #38; tuned 0.668 |
| 5, 12 | Logistic regression / random forest | Refilwe | _owner to fill_ | Refilwe's evaluation PR is also open |
| 6 | October issues on the board | Leo | _owner to fill_ | |
| 7 | Refresh canonical Parquet | Leo | _owner to fill_ | |
| **8** | **XGBoost tuning** | **Mohit** | **Done** | 60 trials, 3-seed refit |
| **9** | **DL literature refresh** | **Mohit** | **Done** | MLP first, FT-Transformer second, no SAINT |
| 13 | Baseline error analysis | Emily | _owner to fill_ | Starting points in the SHAP write-up, section 3 |
| **14** | **MLP baseline** | **Mohit** | **Done** | Below the bar; GPU numerical embeddings are next |
| 15 | Second DL model | Orator | _owner to fill_ | Use `TabularPreprocessor` for comparability |
| 16 | Customer temporal features | Emily | _owner to fill_ | Sept #15 said GO |
| 17 | Ensemble / stacking | Leo | _owner to fill_ | Validation scores for XGBoost and MLP are saved by both scripts (`--report-dir`) |
| **18** | **SHAP attribution** | **Mohit** | **Done** | |
| **19** | **Threshold calibration** | **Mohit** | **Done** | |
| 20 | Drift re-check | Refilwe | _owner to fill_ | Use the half-month stability numbers as a start |
| 22 | Comparison dashboard figure | Emily | _owner to fill_ | |
| 23 | This review | All | Draft | |

Dependency built along the way: the frozen split files (`splits/`, Sept #6) weren't on main, so this branch adds them with `fraud.split`. They use the same day rule as the derived split everyone has been using. Leonel and Nicholas, please review.

## 4. Proposed November scope

1. **Final model selection on the blind test**, scored once: tuned XGBoost, the best LightGBM/CatBoost, and the ensemble if it qualifies.
2. **Robustness:** performance by week and by segment (ProductCD, card type, has_identity), building on #20 and the within-month drop above.
3. **Customer-history features** (#16) in the final model if they add at least 0.01 AUPRC, using past transactions only.
4. **Explainability for the prototype:** reason codes from SHAP families ("many linked cards/accounts", "new or dormant card", "unusual device").
5. **Prototype:** a small API or dashboard that returns the fraud score, expected loss and top three reasons for a transaction.
6. **Optional:** a sequence model over each customer's earlier transactions, only if #16 shows history helps.
7. **Advisor sign-off** on the cost and capacity assumptions (1% / 5% review rates, $20 chargeback, $5 review).

## 5. Open questions for the Challenge Advisor

1. Should the operating rule maximize fraud dollars (expected-loss ranking) or fraud cases (probability ranking)?
2. Is 28 reviews a day (1%) the right capacity for the scenario, or should we plan for 5%?
3. Is it acceptable that 24% of model influence comes from anonymized V features, or should the prototype prefer a model built on describable features?
