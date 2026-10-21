"""
Download and verify the IEEE-CIS data (Task #1).

Before running, each member must:
  1. Create a Kaggle account and accept the competition rules at
     https://www.kaggle.com/c/ieee-fraud-detection/rules
  2. Create an API token (Kaggle > Settings > Create New Token) and save it
     as ~/.kaggle/kaggle.json (chmod 600), or set KAGGLE_USERNAME / KAGGLE_KEY.

Then:
    python scripts/setup_data.py            # download + verify
    python scripts/setup_data.py --verify   # verify files already in data/raw

The script exits non-zero if the zip is corrupt, a file is missing, or a CSV
has fewer rows than expected (a truncated download).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
COMPETITION = "ieee-fraud-detection"

# Data rows (header excluded) in the official files.
EXPECTED_ROWS = {
    "train_transaction.csv": 590_540,
    "train_identity.csv": 144_233,
    "test_transaction.csv": 506_691,
    "test_identity.csv": 141_907,
}
EXPECTED_HEADERS = {
    "train_transaction.csv": ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"],
    "train_identity.csv": ["TransactionID", "id_01"],
}


def count_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for _ in fh) - 1


def download(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(raw_dir)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(
            "Kaggle download failed. Check that your token is set up and that you "
            "accepted the competition rules."
        )
    return raw_dir / f"{COMPETITION}.zip"


def extract(zip_path: Path, raw_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            sys.exit(f"Corrupt member in {zip_path.name}: {bad}. Delete it and re-download.")
        zf.extractall(raw_dir)


def verify(raw_dir: Path, files=EXPECTED_ROWS) -> list[str]:
    problems = []
    for name, expected in files.items():
        path = raw_dir / name
        if not path.exists():
            problems.append(f"missing {name}")
            continue
        header = path.open().readline().strip().split(",")
        for col in EXPECTED_HEADERS.get(name, []):
            if col not in header:
                problems.append(f"{name}: column {col} not in header")
        rows = count_rows(path)
        status = "ok" if rows == expected else "MISMATCH"
        print(f"{name:<24} {rows:>9,} rows (expected {expected:,})  {status}")
        if rows != expected:
            problems.append(f"{name}: {rows:,} rows, expected {expected:,}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--verify", action="store_true", help="Skip download, only verify")
    ap.add_argument("--train-only", action="store_true", help="Only check the labeled train files")
    args = ap.parse_args(argv)

    if not args.verify:
        zip_path = download(args.raw_dir)
        extract(zip_path, args.raw_dir)

    files = {k: v for k, v in EXPECTED_ROWS.items() if not args.train_only or k.startswith("train")}
    problems = verify(args.raw_dir, files)
    if problems:
        print("\nVerification FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("\nAll files present and complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
