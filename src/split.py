"""
Chronological train / validation / blind test split (Task #6).

    day = floor((TransactionDT - min(TransactionDT)) / 86400)
    train  days   0 to 121
    valid  days 122 to 151
    test   days 152 to 182   (sealed until November)

The split is frozen as three CSV files of TransactionID values in
data/splits/. Every model loads those files, so all results are on identical
rows. Re-running make_split on the same data reproduces the files exactly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "splits"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
SECONDS_PER_DAY = 86_400

SPLIT_DAYS = {"train": (0, 121), "valid": (122, 151), "test": (152, 182)}


def day_index(df: pd.DataFrame) -> pd.Series:
    return ((df[TIME_COL] - df[TIME_COL].min()) // SECONDS_PER_DAY).astype(int)


def make_split(df: pd.DataFrame) -> pd.Series:
    """Return a Series of 'train' / 'valid' / 'test' aligned to df."""
    day = day_index(df)
    split = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, (lo, hi) in SPLIT_DAYS.items():
        split[(day >= lo) & (day <= hi)] = name
    if split.isna().any():
        raise ValueError(f"{split.isna().sum()} rows fall outside the split day ranges")
    return split


def save_split(df: pd.DataFrame, split_dir: Path | str | None = None) -> dict:
    """Write {train,valid,test}_ids.csv and return a summary."""
    split_dir = Path(split_dir or SPLIT_DIR)
    split_dir.mkdir(parents=True, exist_ok=True)
    split = make_split(df)
    summary = {}
    for name in SPLIT_DAYS:
        ids = df.loc[split == name, ID_COL].sort_values()
        path = split_dir / f"{name}_ids.csv"
        ids.to_frame().to_csv(path, index=False)
        sub = df.loc[split == name]
        summary[name] = {
            "rows": int(len(sub)),
            "fraud_rate": float(sub["isFraud"].mean()) if "isFraud" in sub else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        }
    return summary


def load_split(name: str, split_dir: Path | str | None = None) -> pd.Series:
    if name not in SPLIT_DAYS:
        raise ValueError(f"split must be one of {list(SPLIT_DAYS)}")
    path = Path(split_dir or SPLIT_DIR) / f"{name}_ids.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python scripts/make_split.py")
    return pd.read_csv(path)[ID_COL]


def split_id(split_dir: Path | str | None = None) -> str:
    """Short hash of the three ID files. Logged with every run."""
    split_dir = Path(split_dir or SPLIT_DIR)
    h = hashlib.sha256()
    for name in SPLIT_DAYS:
        h.update((split_dir / f"{name}_ids.csv").read_bytes())
    return "chrono-v1-" + h.hexdigest()[:10]
