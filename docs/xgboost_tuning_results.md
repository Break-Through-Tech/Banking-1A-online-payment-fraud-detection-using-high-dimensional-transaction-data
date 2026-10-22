# XGBoost Hyperparameter Tuning (October Task #8)

Owner: Mohit · Split `chrono-v1-c3eab22dc2` · Feature set `fe_v2` · Validation split only (blind test untouched)

**Result:** tuned XGBoost reaches **validation AUPRC 0.682 ± 0.0005** over 3 seeds, up from 0.664 for the September #19 baseline. It is the strongest model the team has logged so far.

## How the sweep worked

| Step | Setting |
|---|---|
| Search | Optuna TPE sampler (multivariate, 12 random start-up trials) with a median pruner |
| Trials | **60** (32 completed, 28 pruned early), 131 minutes on 2 CPU cores |
| Tuning data | 150,000 random training rows (the same budget as Orator's LightGBM sweep, Task #10). Scored on all 85,044 validation rows |
| Per trial | Learning rate 0.1, up to 2,000 rounds, early stopping on validation AUPRC (patience 50) |
| Trial 0 | The September baseline settings, so the sweep had to beat them |
| Refit | Best settings retrained on all 420,066 training rows, seeds 42/43/44, early stopping patience 100 |
| Code | `python -m fraud.models.xgboost_tuning --trials 60` (SQLite storage, resumes if interrupted) |

Search space: `max_depth` 4–12, `min_child_weight` 0.5–100, `gamma` 0.001–5, `subsample` 0.5–1, `colsample_bytree` 0.2–1, `colsample_bynode` 0.4–1, `reg_lambda` 0.01–100, `reg_alpha` 0.001–10, `max_cat_threshold` 8–128, and class weighting `scale_pos_weight = 27.4^p` with p from 0 to 1.

## Best configuration (trial 32)

| Parameter | Sept #19 baseline | Tuned |
|---|---|---|
| max_depth | 8 | **10** |
| min_child_weight | 1 | 0.92 |
| gamma | 0 | 0.013 |
| subsample | 0.8 | 0.96 |
| colsample_bytree | 0.5 | 0.43 |
| colsample_bynode | 1.0 | 0.84 |
| reg_lambda (L2) | 1 | **18.6** |
| reg_alpha (L1) | 0 | **2.77** |
| max_cat_threshold | 32 | **17** |
| scale_pos_weight | 27.4 (full ratio) | **15.7** |

What the sweep learned (from the 5 best trials and fANOVA importance):

- **Deeper trees with much stronger regularization.** All five top trials used depth 9–10 with L2 between 5 and 19.
- **Fewer categories per split** (max_cat_threshold 9–17). This keeps the model from memorizing rare DeviceInfo and email values, which we flagged in September.
- **Less class weighting than the full ratio.** Weights between 12 and 23 beat 27.4.
- Column sampling per node had the largest effect on the score (fANOVA importance 0.72). Low values hurt a lot.

## Results on the validation split

| Model | AUPRC | ROC-AUC | Fraud caught, top 1% | Fraud caught, top 5% | KS |
|---|---|---|---|---|---|
| **XGBoost tuned (3 seeds)** | **0.682 ± 0.001** | **0.935** | 28.2% | **71.2%** | 0.739 |
| XGBoost fe_v2, Sept #19 (1 seed) | 0.664 | 0.913 | 28.2% | 69.8% | 0.710 |
| LightGBM tuned, Oct #10 (Orator) | 0.668 | | 28.1% | 69.8% | 0.728 |
| XGBoost raw features, Sept #19 | 0.620 | 0.917 | 27.8% | 65.8% | 0.703 |

- **The gain is real, not noise.** +0.018 AUPRC is 36 times the seed-to-seed spread (std 0.0005) and well above our 0.01 bar (business objective, section 5.4).
- **ROC-AUC improved the most** (+0.022). That means better ranking across the whole score range, not only at the top.
- **The top 1% is at its ceiling.** 851 reviews can catch at most 29.5% of the 2,884 validation frauds, and every model is at 28%. The 5% column and AUPRC are where models differ.
- **Dollar capture (seed 42):** top 1% catches 22% of fraud dollars, top 5% 71%, top 10% 82%, top 20% 92%. Task #19 shows how to raise the 1% figure by ranking on expected loss.
- Training takes 15 minutes per seed on 2 CPU cores and stops on its own at rounds 1,600 to 1,840.

LightGBM numbers are from `results/lightgbm_optuna_task10.json` on main (PR #38). That run used the day-based derived split, which selects the same rows as the frozen files, so the comparison is fair.

## Caveats

- **Optimistic by design.** Hyperparameters and early stopping both used validation, so the validation score of the winner is slightly inflated. The November blind-test run is the unbiased check.
- **Tuned on a 150k-row sample at learning rate 0.1.** A slower learning rate (0.05 or 0.03) on a GPU may add a little more.
- **Memory:** depth-10 trees need `max_cached_hist_node` capped (now set to 256 in the baseline defaults). Without it, the full-data refit ran out of memory on an 8 GB machine. The cap doesn't change the model.

## Files

- `results/xgboost_optuna_task8.json`: summary, best parameters, 3-seed refit metrics
- `results/xgboost_optuna_task8_trials.csv`: every trial with parameters, score, best round and runtime
- `results/registry.jsonl`: the three refit runs (`task: "Oct #8 tuned"`)
