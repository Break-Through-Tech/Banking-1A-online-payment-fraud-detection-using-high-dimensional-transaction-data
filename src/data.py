"""
Data loading, merging, and memory optimization (Tasks #2, #4, #5).

Contract (agreed in docs/module_contracts.md):

    load_raw(raw_dir)                -> (transactions, identity)
    merge_tables(trans, ident)       -> merged DataFrame with has_identity
    optimize_memory(df)              -> same data, smaller dtypes
    build_canonical(raw_dir, out)    -> writes data/processed/merged.parquet
    load_data(path=None, split=None) -> merged DataFrame, optionally one split

Null convention (Task #4)
-------------------------
- The merge is a LEFT join from transactions to identity, so every labeled
  transaction is kept (590,540 rows).
- `has_identity` is 1 when an identity row exists, else 0.
- Identity columns are NaN when `has_identity == 0`. That NaN means "no
  identity record", not "value unknown". Do not impute identity columns
  without keeping `has_identity` next to them.
- The raw identity file names its columns `id-01` in the test file and
  `id_01` in the train file. We normalize to `id_01`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "merged.parquet"

ID_COL = "TransactionID"
TARGET = "isFraud"
TIME_COL = "TransactionDT"

EXPECTED_ROWS = {"train_transaction.csv": 590_540, "train_identity.csv": 144_233}

# Columns that are categorical even though some look numeric.
CATEGORICAL_COLS = (
    ["ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
    + [f"card{i}" for i in (4, 6)]
    + [f"M{i}" for i in range(1, 10)]
    + [f"id_{i}" for i in range(12, 39)]
)
# Numeric codes that should be modeled as categories, kept numeric on disk.
NUMERIC_CODE_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]


def load_raw(raw_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir or RAW_DIR)
    trans = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    ident.columns = [c.replace("id-", "id_") for c in ident.columns]
    return trans, ident


def merge_tables(trans: pd.DataFrame, ident: pd.DataFrame) -> pd.DataFrame:
    """Left join transactions to identity and add has_identity."""
    if trans[ID_COL].duplicated().any():
        raise ValueError("Duplicate TransactionID in transaction table")
    if ident[ID_COL].duplicated().any():
        raise ValueError("Duplicate TransactionID in identity table")

    ident = ident.copy()
    ident["has_identity"] = np.int8(1)
    df = trans.merge(ident, on=ID_COL, how="left", validate="one_to_one")
    df["has_identity"] = df["has_identity"].fillna(0).astype("int8")

    if len(df) != len(trans):
        raise AssertionError("Merge changed the number of transactions")
    return df


def optimize_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numerics and convert string columns to category."""
    before = df.memory_usage(deep=True).sum()
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            # float32 keeps ~7 significant digits, enough for every column
            # except TransactionAmt cents and TransactionDT, which we keep.
            if col in ("TransactionAmt", TIME_COL):
                continue
            out[col] = s.astype("float32")
        elif not pd.api.types.is_numeric_dtype(s):
            out[col] = s.astype("category")
    after = out.memory_usage(deep=True).sum()
    if verbose:
        print(f"Memory: {before / 1e6:,.0f} MB -> {after / 1e6:,.0f} MB")
    return out


def build_canonical(raw_dir: Path | str | None = None, out_path: Path | str | None = None) -> Path:
    """Merge, optimize, and write the single canonical Parquet file."""
    trans, ident = load_raw(raw_dir)
    df = optimize_memory(merge_tables(trans, ident))
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    out_path = Path(out_path or PROCESSED_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols to {out_path}")
    return out_path


def load_data(path: Path | str | None = None, split: str | None = None, split_dir=None) -> pd.DataFrame:
    """Load the canonical dataset, optionally restricted to one split.

    split: None, "train", "valid", or "test". Loading "test" prints a warning
    because the blind test set is sealed until November.
    """
    df = pd.read_parquet(path or PROCESSED_PATH)
    if split is None:
        return df
    from src.split import load_split  # local import avoids a cycle

    if split == "test":
        print("WARNING: loading the sealed blind test split.")
    ids = load_split(split, split_dir)
    return df[df[ID_COL].isin(ids)].reset_index(drop=True)


if __name__ == "__main__":
    build_canonical()
