"""
Shared plumbing for every baseline (Tasks #17, #18, #19, #20).

Each model script does three things:
    train, valid, split_id = load_train_valid(...)
    ... fit the model, predict scores on valid ...
    finish_run("lightgbm", valid, scores, params=..., ...)

so all models see the same rows, the same features, and the same metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluation, registry, split as split_mod
from src.data import ID_COL, PROCESSED_PATH, TARGET, TIME_COL

DROP_COLS = [TARGET, ID_COL, TIME_COL]
REPO_ROOT = Path(__file__).resolve().parents[2]


def base_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", type=Path, default=PROCESSED_PATH)
    p.add_argument("--split-dir", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature-set", default="raw_all_v1")
    p.add_argument("--registry", type=Path)
    p.add_argument("--report-dir", type=Path, help="Write capture table + plots here")
    p.add_argument("--sample", type=float, help="Train on a fraction of train rows (quick runs)")
    p.add_argument("--notes", default="")
    return p


def load_train_valid(data_path, split_dir=None, sample: float | None = None, seed: int = 42):
    """Return train and valid frames plus the split identifier.

    Uses the frozen split files when present. Falls back to the day-based
    rule and marks the split_id as derived so registry comparisons can
    tell them apart.
    """
    df = pd.read_parquet(data_path)
    try:
        train_ids = split_mod.load_split("train", split_dir)
        valid_ids = split_mod.load_split("valid", split_dir)
        train = df[df[ID_COL].isin(train_ids)]
        valid = df[df[ID_COL].isin(valid_ids)]
        sid = split_mod.split_id(split_dir)
    except FileNotFoundError:
        labels = split_mod.make_split(df)
        train, valid = df[labels == "train"], df[labels == "valid"]
        sid = "derived:chrono-v1"
    if sample:
        train = train.sample(frac=sample, random_state=seed)
    return train.reset_index(drop=True), valid.reset_index(drop=True), sid


def feature_columns(df: pd.DataFrame, extra_drop=()) -> list[str]:
    drop = set(DROP_COLS) | set(extra_drop)
    return [c for c in df.columns if c not in drop]


def categorical_columns(df: pd.DataFrame, cols) -> list[str]:
    return [c for c in cols if not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]


def categorize(train: pd.DataFrame, others: list[pd.DataFrame], cat_cols) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Pandas categoricals with levels learned on train; unseen levels become NaN."""
    train = train.copy()
    others = [o.copy() for o in others]
    for c in cat_cols:
        levels = pd.Index(train[c].dropna().astype(str).unique())
        train[c] = pd.Categorical(train[c].astype("string"), categories=levels)
        for o in others:
            v = o[c].astype("string")
            o[c] = pd.Categorical(v.where(v.isin(levels)), categories=levels)
    return train, others


def pos_weight(y) -> float:
    y = np.asarray(y)
    return float((len(y) - y.sum()) / max(y.sum(), 1))


def finish_run(model: str, valid: pd.DataFrame, scores, *, params: dict, feature_set: str, seed: int,
               split_id: str, n_features: int, extra: dict | None = None, registry_path=None,
               report_dir=None, notes: str = "") -> dict:
    metrics = evaluation.evaluate(valid[TARGET].values, scores, amount=valid["TransactionAmt"].values)
    record = {
        "model": model,
        "feature_set": feature_set,
        "n_features": n_features,
        "params": params,
        "split_id": split_id,
        "seed": seed,
        "metrics": {"valid": metrics},
        "notes": notes,
        **(extra or {}),
    }
    path = registry.log_run(record, registry_path)
    print(f"[{model}] {evaluation.headline(metrics)}")
    print(f"[{model}] logged to {path}")
    if report_dir:
        evaluation.save_standard_report(valid[TARGET].values, scores, valid["TransactionAmt"].values,
                                        report_dir, model)
    return record
