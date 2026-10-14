"""Build the canonical Banking 1A dataset from the raw Kaggle files.

Reads data/raw/train_transaction.csv and data/raw/train_identity.csv,
shrinks each column to a smaller type, merges the two tables, and writes
data/train_merged_v1.parquet plus a summary in docs/dataset_v1_summary.json.

Usage:
    python scripts/build_dataset.py
"""
import json
import subprocess
import sys
import time

import pandas as pd

from fraud.config import DATA_DIR, PARQUET_PATH, REPO_ROOT

RAW_DIR = DATA_DIR / "raw"
SUMMARY_PATH = REPO_ROOT / "docs" / "dataset_v1_summary.json"
EXPECTED_ROWS = 590_540
EXPECTED_FRAUD = 20_663
INT32_COLUMNS = {"TransactionID", "TransactionDT"}
KEEP_FLOAT64 = {"TransactionAmt"}  # exact cents


def fail(message):
    """Print an error and stop with exit code 1."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def gigabytes(*frames):
    """Total memory used by one or more DataFrames in gigabytes."""
    return sum(f.memory_usage(deep=True).sum() for f in frames) / 1e9


def load_raw():
    """Read the two raw CSVs. Stops with error if one is missing."""
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


def optimize(df):
    """Store each column in a smaller type
    """
    for col in df.columns:
        s = df[col]
        if col in INT32_COLUMNS:
            df[col] = s.astype("int32")
        elif col in KEEP_FLOAT64:
            continue
        elif s.dtype == "object" or pd.api.types.is_string_dtype(s):
            df[col] = s.astype("category")
        elif pd.api.types.is_float_dtype(s):
            df[col] = s.astype("float32")
        elif pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
    return df.copy()


def merge_tables(tx, idn):
    """Left-join identity onto transactions. flag which rows had identity data."""
    df = tx.merge(idn, on="TransactionID", how="left", validate="one_to_one", indicator=True)
    df["has_identity"] = (df.pop("_merge") == "both").astype("int8")
    return df


def check_merge(df, tx, idn):
    """Fail  if the merge lost, duplicated, or mismatched rows."""
    if len(df) != len(tx) or len(df) != EXPECTED_ROWS:
        fail(f"Expected {EXPECTED_ROWS:,} rows after the merge, got {len(df):,}.")
    if df["has_identity"].sum() != len(idn):
        fail("has_identity count does not match the number of identity rows.")
    if df["isFraud"].sum() != EXPECTED_FRAUD:
        fail(f"Expected {EXPECTED_FRAUD:,} fraud rows, got {df['isFraud'].sum():,}.")


def write_and_verify(df):
    """write the Parquet file, read it back, and confirm nothing changed."""
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False, engine="pyarrow")
    start = time.perf_counter()
    back = pd.read_parquet(PARQUET_PATH)
    seconds = time.perf_counter() - start
    if back.shape != df.shape or list(back.columns) != list(df.columns):
        fail("The Parquet file read back with a different shape or columns.")
    if (back.dtypes.astype(str) != df.dtypes.astype(str)).any():
        fail("The Parquet file read back with different column types.")
    return seconds


def git_commit():
    """Short ID of the Git commit"""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    start = time.perf_counter()
    tx, idn = load_raw()
    csv_seconds = time.perf_counter() - start

    before = gigabytes(tx, idn)
    tx, idn = optimize(tx), optimize(idn)
    after = gigabytes(tx, idn)
    print(f"Memory before optimizing: {before:.2f} GB")
    print(f"Memory after optimizing:  {after:.2f} GB ({1 - after / before:.0%} smaller)")

    df = merge_tables(tx, idn)
    check_merge(df, tx, idn)
    del tx, idn  # only the merged table is needed from here on
    merged = gigabytes(df)
    print(f"Merged: {len(df):,} rows x {df.shape[1]} columns")
    print(f"  fraud rows:          {int(df['isFraud'].sum()):,}")
    print(f"  rows with identity:  {int(df['has_identity'].sum()):,} ({df['has_identity'].mean():.1%})")
    print(f"  in memory:           {merged:.2f} GB")

    parquet_seconds = write_and_verify(df)
    size_mb = PARQUET_PATH.stat().st_size / 1e6
    print(f"Wrote {PARQUET_PATH} ({size_mb:,.0f} MB)")
    print(f"Read back in {parquet_seconds:.1f} s (the CSVs took {csv_seconds:.1f} s); shape and types match")

    summary = {
        "file": PARQUET_PATH.name,
        "version": "v1",
        "rows": len(df),
        "columns": df.shape[1],
        "fraud_rows": int(df["isFraud"].sum()),
        "rows_with_identity": int(df["has_identity"].sum()),
        "memory_gb_before": round(before, 2),
        "memory_gb_after": round(after, 2),
        "memory_gb_merged": round(merged, 2),
        "file_size_mb": round(size_mb, 1),
        "csv_load_seconds": round(csv_seconds, 1),
        "parquet_load_seconds": round(parquet_seconds, 1),
        "built_from_commit": git_commit(),
        "pandas_version": pd.__version__,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()