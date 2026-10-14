"""Build the canonical Banking 1A dataset from the raw Kaggle files.

Usage:
    python scripts/build_dataset.py
"""
import sys

import pandas as pd

from fraud.config import DATA_DIR

RAW_DIR = DATA_DIR / "raw"
EXPECTED_ROWS = 590_540
EXPECTED_FRAUD = 20_663


def fail(message):
    """Print an error and stop with exit code 1, so a bad build can't go unnoticed."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_raw():
    """Read the two raw CSVs. Stops with a clear error if either is missing."""
    paths = {name: RAW_DIR / f"train_{name}.csv" for name in ("transaction", "identity")}
    for path in paths.values():
        if not path.exists():
            fail(f"Missing {path}. Run python scripts/setup_data.py first.")
    print(f"Reading raw CSVs from {RAW_DIR} ...")
    tx = pd.read_csv(paths["transaction"])
    idn = pd.read_csv(paths["identity"])
    print(f"  train_transaction.csv: {len(tx):,} rows x {tx.shape[1]} columns")
    print(f"  train_identity.csv:    {len(idn):,} rows x {idn.shape[1]} columns")
    return tx, idn


def merge_tables(tx, idn):
    """Left-join identity onto transactions and flag which rows had identity data."""
    df = tx.merge(idn, on="TransactionID", how="left", validate="one_to_one", indicator=True)
    df["has_identity"] = (df.pop("_merge") == "both").astype("int8")
    return df


def check_merge(df, tx, idn):
    """Fail loudly if the merge lost, duplicated, or mismatched rows."""
    if len(df) != len(tx) or len(df) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS:,} rows after the merge, got {len(df):,}.")
    if df["has_identity"].sum() != len(idn):
        fail("has_identity count does not match the number of identity rows.")
    if df["isFraud"].sum() != EXPECTED_FRAUD:
        fail(f"Expected {EXPECTED_FRAUD:,} fraud rows, got {df['isFraud'].sum():,}.")


def main():
    tx, idn = load_raw()
    df = merge_tables(tx, idn)
    check_merge(df, tx, idn)
    print(f"Merged: {len(df):,} rows x {df.shape[1]} columns")
    print(f"  fraud rows:          {int(df['isFraud'].sum()):,}")
    print(f"  rows with identity:  {int(df['has_identity'].sum()):,} ({df['has_identity'].mean():.1%})")
    print("All merge checks passed.")


if __name__ == "__main__":
    main()