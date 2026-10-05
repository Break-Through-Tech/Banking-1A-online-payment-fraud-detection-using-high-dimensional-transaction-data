"""
Build the canonical Parquet (Task #5) and freeze the chronological split (Task #6).

    python scripts/make_split.py                       # from data/raw CSVs
    python scripts/make_split.py --data merged.parquet # from an existing Parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import data, split  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, help="Existing merged Parquet")
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--split-dir", type=Path)
    args = ap.parse_args(argv)

    path = args.data or data.build_canonical(args.raw_dir)
    df = pd.read_parquet(path, columns=[split.ID_COL, split.TIME_COL, "isFraud"])
    summary = split.save_split(df, args.split_dir)
    print(json.dumps(summary, indent=2))
    print("split_id:", split.split_id(args.split_dir))


if __name__ == "__main__":
    main()
