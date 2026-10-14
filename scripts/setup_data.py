"""Download the IEEE-CIS raw files from Kaggle and verify them.

Usage:
    python scripts/setup_data.py                   // download if needed, then verify
    python scripts/setup_data.py --force           // download again even if files exist
    python scripts/setup_data.py --write-manifest  // record row counts and hashes of a good download
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

COMPETITION = "ieee-fraud-detection"
RAW_DIR = Path(os.environ.get("FRAUD_DATA_DIR", "data")) / "raw"
MANIFEST = Path(__file__).with_name("data_manifest.json")
FILES = ["train_transaction.csv", "train_identity.csv"]


def fail(message):
    """Print an error and stop with exit code 1, so a broken download can't go unnoticed."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def check_credentials():
    has_token = os.environ.get("KAGGLE_API_TOKEN")
    has_legacy = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if not (has_token or has_legacy):
        fail("No Kaggle credentials found. Set KAGGLE_API_TOKEN (see README).")
    if shutil.which("kaggle") is None:
        fail("The kaggle command was not found. Activate your venv and run: pip install kaggle")


def download():
    check_credentials()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} into {RAW_DIR} ...")
    result = subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)])
    if result.returncode != 0:
        fail("Kaggle download failed. Did you accept the competition rules on kaggle.com?")
    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if not zip_path.exists():
        fail(f"Expected {zip_path} after the download, but it is missing.")
    with zipfile.ZipFile(zip_path) as archive:
        for name in FILES:
            archive.extract(name, RAW_DIR)


def sha256(path):
    """Fingerprint a file. Any change to its bytes changes this value."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(name):
    path = RAW_DIR / name
    if not path.exists():
        fail(f"Missing file: {path}")
    try:
        rows = len(pd.read_csv(path, usecols=["TransactionID"]))
    except Exception as error:
        fail(f"Could not read {path}: {error}")
    return {"rows": rows, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if files exist")
    parser.add_argument("--write-manifest", action="store_true", help="save row counts and hashes of the current files")
    args = parser.parse_args()

    if args.force or not all((RAW_DIR / name).exists() for name in FILES):
        download()

    actual = {name: describe(name) for name in FILES}

    if args.write_manifest:
        MANIFEST.write_text(json.dumps(actual, indent=2) + "\n")
        print(f"Wrote {MANIFEST}. Check the row counts, then commit it.")
        return

    if not MANIFEST.exists():
        fail(f"{MANIFEST} not found. Run once with --write-manifest on a known-good download.")
    expected = json.loads(MANIFEST.read_text())
    for name in FILES:
        if actual[name] != expected.get(name):
            fail(f"{name} does not match the manifest.\n  expected: {expected.get(name)}\n  actual:   {actual[name]}")

    for name in FILES:
        print(f"OK  {name}: {actual[name]['rows']:,} rows, fingerprint matches")
    print("Raw data is present and verified.")


if __name__ == "__main__":
    main()