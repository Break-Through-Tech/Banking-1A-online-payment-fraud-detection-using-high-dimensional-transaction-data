# Frozen chronological split

Created by `python scripts/make_split.py` (Task #6). Do not edit these files by hand.

| File | Days from first `TransactionDT` | Purpose |
|---|---|---|
| `train_ids.csv` | 0 to 121 | Fit models, encoders, scalers, feature selection |
| `valid_ids.csv` | 122 to 151 | Early stopping, model choice, threshold choice |
| `test_ids.csv` | 152 to 182 | **Sealed until November.** Score finalists once |

Day index: `floor((TransactionDT - min(TransactionDT)) / 86400)`.

Each file holds one column, `TransactionID`, sorted ascending. `src.split.split_id()` hashes the three files. That hash is logged with every run in the results registry, so we can tell if anyone ran on a different split.

Rules:

1. Load splits through `src.split.load_split` or `src.data.load_data(split=...)`. Don't re-derive them in notebooks.
2. Nothing is fitted on validation or test rows.
3. If the test split is scored before November, write it in the run's `notes` field.
