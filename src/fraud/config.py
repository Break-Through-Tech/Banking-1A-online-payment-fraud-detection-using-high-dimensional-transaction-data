import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("FRAUD_DATA_DIR", "data"))
PARQUET_PATH = DATA_DIR / "train_merged_v1.parquet"
REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "splits"