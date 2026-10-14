# Data

## Join and null convention
- The transaction table (590,540 rows) is left-joined with the identity table (144,233 rows) on TransactionID. Every transaction is kept.
- has_identity is 1 if the transaction had an identity record and 0 otherwise.
- Missing values stay as NaN. No -999 fills, no "unknown" strings, no encoding in this file.
- When has_identity is 0, every identity column is NaN by construction, meaning "not collected." That is different from a NaN inside a row that does have identity data.
- Filling, encoding, and missing-value indicators happen downstream in each model's code, fitted on the training split only.