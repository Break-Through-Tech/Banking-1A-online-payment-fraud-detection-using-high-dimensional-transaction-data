# Repo structure and module contracts

Task #2 · Owners: Leonel, Nicholas

Agreed layout so the data, feature, evaluation, and model tracks can be built in parallel. If you need to change a signature below, open an issue first.

```
data/
  raw/            Kaggle CSVs (git-ignored, from scripts/setup_data.py)
  processed/      merged.parquet (git-ignored, also in the shared Drive)
  splits/         train_ids.csv, valid_ids.csv, test_ids.csv (committed)
docs/             write-ups and decisions
notebooks/        exploration only; reusable code moves to src/
reports/          generated tables and plots (eda/, per-model plots)
results/          registry.jsonl (committed, append-only)
scripts/          command-line entry points
src/
  data.py         load_raw, merge_tables, optimize_memory, build_canonical, load_data
  split.py        make_split, save_split, load_split, split_id
  evaluation.py   evaluate, dollar_capture_table, plots
  registry.py     log_run, load_registry, leaderboard
  eda.py          missingness, categorical profile, drift
  entity.py       UID candidates and feasibility report
  CategoricalEncoder.py   (Orator, PR #3) one-hot / frequency encoding for linear and MLP models
  models/
    common.py     load_train_valid, categorize, finish_run
    linear_rf_baseline.py, lightgbm_baseline.py, xgboost_baseline.py, catboost_baseline.py
tests/            pytest, synthetic data only (no Kaggle download needed)
```

## Contracts

| Function | Input | Output | Notes |
|---|---|---|---|
| `src.data.load_data(path=None, split=None)` | Parquet path, `"train"`/`"valid"`/`"test"` | DataFrame | Warns when loading `"test"` |
| `src.split.load_split(name)` | split name | Series of `TransactionID` | Raises if files not frozen yet |
| `src.split.split_id()` | none | `"chrono-v1-<hash>"` | Logged with every run |
| `src.evaluation.evaluate(y_true, y_score, amount=None, threshold=0.5)` | arrays | dict | The only place metrics are computed |
| `src.registry.log_run(record)` | dict with `model, feature_set, params, split_id, seed, metrics` | path | Append-only |
| Feature functions | `fit(train_df)` then `transform(df)` | DataFrame | Fit on train only. Aggregates may use earlier rows only |

## Conventions

- Column names stay as in Kaggle (`TransactionID`, `isFraud`, `id_01`).
- Any `fit` sees the train split only.
- `TransactionID` and `TransactionDT` are never model features.
- Random seeds default to 42. Report at least 3 seeds before calling a winner.
- Branches: `<name>/<task-number>-<short-desc>`, one PR per task, reference the issue (`Closes #N`).
