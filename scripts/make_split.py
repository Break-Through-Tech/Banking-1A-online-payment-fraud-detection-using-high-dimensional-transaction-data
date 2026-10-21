"""Freeze the chronological split (Task #6).

Writes splits/{train,valid,test}_ids.csv from the canonical Parquet. Run once;
the files are committed and every model loads them, so all results are on
identical rows. Re-running on the same data reproduces the files exactly.

    python scripts/make_split.py
    python scripts/make_split.py --data path/to/train_merged_v1.parquet
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from fraud import split
from fraud.config import PARQUET_PATH


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=PARQUET_PATH)
    ap.add_argument("--split-dir", type=Path)
    args = ap.parse_args(argv)
    df = pd.read_parquet(args.data, columns=[split.ID_COL, split.TIME_COL, "isFraud"])
    print(json.dumps(split.save_split(df, args.split_dir), indent=2))
    print("split_id:", split.split_id(args.split_dir))


if __name__ == "__main__":
    main()
