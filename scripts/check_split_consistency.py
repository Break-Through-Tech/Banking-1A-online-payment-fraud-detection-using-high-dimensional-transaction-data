"""Split consistency check (Nov Task 6, issue #67).

Confirms that the day rule in fraud.split and the frozen ID files in splits/
give exactly the same TransactionID sets, and that every saved validation
score file covers exactly the validation rows. Reads only TransactionID and
TransactionDT, never labels or features, so the sealed test split stays sealed.

    python scripts/check_split_consistency.py

Prints PASS or FAIL per check, saves the same lines to reports/split_check.txt,
and exits with code 1 if anything fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import PARQUET_PATH
from fraud.split import ID_COL, SPLIT_DAYS, TIME_COL, load_split, make_split, split_id

REPORT = Path("reports/split_check.txt")
EXPECTED_SPLIT_ID = "chrono-v1-c3eab22dc2"
EXPECTED_SIZES = {"train": 420_066, "valid": 85_044, "test": 85_430}
SCORE_FILES = {
    "CatBoost baseline": Path("reports/catboost_baseline/valid_scores.parquet"),
    "LightGBM tuned": Path("reports/lightgbm_tuned/valid_scores.parquet"),
    "Logistic regression baseline": Path("reports/linear_rf_baseline/lr_valid_scores.parquet"),
    "Random forest baseline": Path("reports/linear_rf_baseline/rf_valid_scores.parquet"),
    "XGBoost tuned": Path("reports/xgboost_tuned/valid_scores.npy"),
}

lines = []
failures = 0


def report(ok, text):
    global failures
    if ok is None:
        tag = "SKIP"
    elif ok:
        tag = "PASS"
    else:
        tag = "FAIL"
        failures += 1
    line = f"{tag}  {text}"
    print(line)
    lines.append(line)


def check_split_files(df):
    """Day rule vs frozen files, one line per split."""
    derived = make_split(df)
    for name in SPLIT_DAYS:
        frozen = set(load_split(name))
        rule = set(df.loc[derived == name, ID_COL])
        only_frozen, only_rule = len(frozen - rule), len(rule - frozen)
        ok = not only_frozen and not only_rule and len(frozen) == EXPECTED_SIZES[name]
        report(ok, f"{name}: frozen file {len(frozen):,} rows, day rule {len(rule):,} rows, "
                   f"only in file {only_frozen}, only in day rule {only_rule}, expected {EXPECTED_SIZES[name]:,}")


def check_score_file(label, path, valid_ids):
    """A score file must hold one finite score for every validation row and nothing else."""
    if not path.exists():
        report(None, f"{label}: {path} not found on this machine")
        return
    if path.suffix == ".npy":
        scores = np.load(path)
        ok = len(scores) == len(valid_ids) and np.isfinite(scores).all()
        report(ok, f"{label}: {len(scores):,} scores in TransactionID order (expected {len(valid_ids):,}), "
                   f"all finite: {bool(np.isfinite(scores).all())}")
        return
    frame = pd.read_parquet(path)
    ids = frame[ID_COL]
    missing = len(valid_ids.difference(ids))
    extra = len(pd.Index(ids).difference(valid_ids))
    dupes = int(ids.duplicated().sum())
    finite = bool(np.isfinite(frame["score"].to_numpy(dtype=float)).all())
    ok = not missing and not extra and not dupes and finite
    report(ok, f"{label}: {len(ids):,} rows, missing {missing}, extra {extra}, duplicates {dupes}, "
               f"all finite: {finite}")


def main():
    print(f"Reading {ID_COL} and {TIME_COL} from {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH, columns=[ID_COL, TIME_COL])

    sid = split_id()
    report(sid == EXPECTED_SPLIT_ID, f"split id of the frozen files: {sid} (expected {EXPECTED_SPLIT_ID})")
    check_split_files(df)

    valid_ids = pd.Index(load_split("valid"))
    for label, path in SCORE_FILES.items():
        check_score_file(label, path, valid_ids)

    summary = "All checks passed." if not failures else f"{failures} check(s) FAILED."
    print(summary)
    lines.append(summary)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved to {REPORT}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
