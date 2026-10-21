"""
Results registry and run conventions (Task #9).

One JSON line per run in results/registry.jsonl. The file is append-only:
never edit or delete a line. If a run was wrong, log a new run and put the
reason in `notes`.

Required fields for every record:
    model, feature_set, params, split_id, seed, metrics
Added automatically:
    run_id, timestamp_utc, git_commit, git_dirty, author

Conventions
-----------
- feature_set: "<name>_v<N>", e.g. "raw_all_v1". Bump N when the feature list changes.
- split_id: from src.split.split_id(). "derived:..." means the frozen files were missing.
- metrics: the dict from src.evaluation.evaluate, nested under "valid" (and
  "test" only for November finalists).
"""

from __future__ import annotations

import getpass
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "results" / "registry.jsonl"
REQUIRED = ("model", "feature_set", "params", "split_id", "seed", "metrics")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def log_run(record: dict, path: Path | str | None = None) -> Path:
    missing = [k for k in REQUIRED if k not in record]
    if missing:
        raise ValueError(f"Registry record missing required fields: {missing}")
    if "valid" not in record["metrics"]:
        raise ValueError("metrics must contain a 'valid' entry")

    full = {
        "run_id": uuid.uuid4().hex[:12],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "author": getpass.getuser(),
        **record,
    }
    path = Path(path or REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(full, default=str, sort_keys=True) + "\n")
    return path


def load_registry(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path or REGISTRY_PATH)
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.json_normalize(records)


def leaderboard(path: Path | str | None = None, split_id: str | None = None) -> pd.DataFrame:
    """Best run per model on the validation split, sorted by AUPRC."""
    df = load_registry(path)
    if df.empty:
        return df
    if split_id:
        df = df[df["split_id"] == split_id]
    cols = {
        "metrics.valid.auprc": "auprc",
        "metrics.valid.roc_auc": "roc_auc",
        "metrics.valid.recall_at_1pct_review": "recall@1%",
        "metrics.valid.recall_at_5pct_review": "recall@5%",
        "metrics.valid.ks": "ks",
    }
    keep = ["model", "feature_set", "seed", "split_id", "run_id", "timestamp_utc"] + [c for c in cols if c in df]
    df = df[keep].rename(columns=cols)
    best = df.sort_values("auprc", ascending=False).groupby("model", as_index=False).head(1)
    return best.reset_index(drop=True)
