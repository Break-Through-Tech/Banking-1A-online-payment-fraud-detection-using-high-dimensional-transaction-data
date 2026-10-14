# Data

## Join and null convention
- The transaction table (590,540 rows) is left-joined with the identity table (144,233 rows) on TransactionID. Every transaction is kept.
- has_identity is 1 if the transaction had an identity record and 0 otherwise.
- Missing values stay as NaN. No -999 fills, no "unknown" strings, no encoding in this file.
- When has_identity is 0, every identity column is NaN by construction, meaning "not collected." That is different from a NaN inside a row that does have identity data.
- Filling, encoding, and missing-value indicators happen downstream in each model's code, fitted on the training split only.


## where the data lives
- Raw Kaggle files: data/raw/ on your machine, from python scripts/setup_data.py.
- Canonical dataset: train_merged_v1.parquet in the team Google Drive folder https://drive.google.com/drive/folders/1fZAos9AeBxeuluYzd4gFHD41wT1p1-qW?usp=sharing. Put your local copy in this repo's data/ folder. Git ignores it
- Built by python scripts/build_dataset.py. Build numbers are in docs/dataset_v1_summary.json.

##  shape
- 590,540 rows x 435 columns: 394 transaction columns, 40 identity columns, and has_identity.
- 20,663 fraud rows (3.5%). 144,233 rows with identity data (24.4%).
- About 0.97 GB in memory once loaded.

## column types
- TransactionID, TransactionDT: int32.
- TransactionAmt: float64
- All other decimal columns: float32
- Whole-number columns: the smallest integer type that fits.
- Text columns: pandas "category". This changes storage only

## How to load
- from fraud.config import PARQUET_PATH, then df = pd.read_parquet(PARQUET_PATH).
- If your copy is not in data/, set FRAUD_DATA_DIR to its folder first.
- After Task 6, use load_data("train") and load_data("val") instead.

- v1, built 10/14: first build