"""
XGBoost baseline for the Banking-1A fraud detection project (Task #19).

What it does
------------
1. Loads the merged transaction + identity data (the canonical Parquet from
   Task #5, or the raw Kaggle CSVs as a fallback).
2. Applies the frozen chronological split from Task #6 (TransactionID lists
   in data/splits/). If those files are not there yet, it derives the same
   day-based split from TransactionDT so work is not blocked.
3. Trains a gradient boosted tree model with class-imbalance handling
   (scale_pos_weight = negatives / positives on the training split) and
   early stopping on validation AUPRC.
4. Scores the validation split with the shared evaluation module (Task #7)
   when it exists, otherwise with the local fallback metrics below.
5. Appends one record to the results registry (Task #9).

The blind test split is never loaded unless --use-test is passed. Per the
team plan it stays sealed until November.

Usage
-----
    python -m src.models.xgboost_baseline --data data/processed/merged.parquet
    python -m src.models.xgboost_baseline --raw-dir data/raw --imbalance none
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"
SECONDS_PER_DAY = 86_400

# Day ranges agreed in Task #6 (inclusive bounds).
SPLIT_DAYS = {
    "train": (0, 121),
    "valid": (122, 151),
    "test": (152, 182),
}

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "max_cat_to_onehot": 1,
    "max_bin": 256,
}

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_merged(data_path: Path | None = None, raw_dir: Path | None = None) -> pd.DataFrame:
    """Return the merged transaction + identity table.

    Prefers the canonical Parquet. Falls back to merging the raw Kaggle CSVs
    with a left join on TransactionID and a has_identity flag, which matches
    the convention in Task #4.
    """
    if data_path is not None:
        data_path = Path(data_path)
        if data_path.suffix == ".parquet":
            return pd.read_parquet(data_path)
        return pd.read_csv(data_path)

    raw_dir = Path(raw_dir or REPO_ROOT / "data" / "raw")
    trans = pd.read_csv(raw_dir / "train_transaction.csv")
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    ident["has_identity"] = 1
    df = trans.merge(ident, on=ID_COL, how="left")
    df["has_identity"] = df["has_identity"].fillna(0).astype("int8")
    return df


def add_day_index(df: pd.DataFrame) -> pd.Series:
    """Day index counted from the first transaction in the labeled data."""
    return ((df[TIME_COL] - df[TIME_COL].min()) // SECONDS_PER_DAY).astype(int)


def assign_split(df: pd.DataFrame, split_dir: Path | None = None) -> pd.Series:
    """Label each row train / valid / test.

    Uses the frozen ID lists from Task #6 when present so every model is
    scored on identical rows. Otherwise rebuilds the split from the day index.
    """
    split_dir = Path(split_dir or REPO_ROOT / "data" / "splits")
    files = {name: split_dir / f"{name}_ids.csv" for name in SPLIT_DAYS}

    if all(f.exists() for f in files.values()):
        split = pd.Series("unassigned", index=df.index)
        for name, f in files.items():
            ids = pd.read_csv(f)[ID_COL]
            split[df[ID_COL].isin(ids)] = name
        source = f"frozen:{split_dir}"
    else:
        day = add_day_index(df)
        split = pd.Series("unassigned", index=df.index)
        for name, (lo, hi) in SPLIT_DAYS.items():
            split[(day >= lo) & (day <= hi)] = name
        source = "derived:TransactionDT"

    split.attrs["source"] = source
    return split


def prepare_features(train: pd.DataFrame, others: list[pd.DataFrame], drop: list[str]):
    """Build feature matrices.

    Object columns become pandas categoricals so XGBoost can split on them
    natively. Category levels are learned from the training split only;
    unseen levels in validation or test become missing.
    """
    feature_cols = [c for c in train.columns if c not in drop]
    X_train = train[feature_cols].copy()
    X_others = [o[feature_cols].copy() for o in others]

    # Anything non-numeric (object, string, category) is treated as categorical.
    cat_cols = [
        c for c in feature_cols
        if not (pd.api.types.is_numeric_dtype(X_train[c]) or pd.api.types.is_bool_dtype(X_train[c]))
    ]
    for c in cat_cols:
        levels = pd.Index(X_train[c].dropna().astype(str).unique())
        X_train[c] = pd.Categorical(X_train[c].astype("string"), categories=levels)
        for X in X_others:
            vals = X[c].astype("string")
            vals = vals.where(vals.isin(levels))  # unseen level -> missing
            X[c] = pd.Categorical(vals, categories=levels)

    return X_train, X_others, feature_cols, cat_cols


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _recall_at_review_rate(y_true: np.ndarray, y_score: np.ndarray, rate: float) -> float:
    """Share of fraud caught if analysts review the top `rate` of transactions."""
    k = max(1, int(np.ceil(rate * len(y_score))))
    top = np.argsort(-y_score)[:k]
    total = y_true.sum()
    return float(y_true[top].sum() / total) if total else float("nan")


def _dollar_capture(y_true, y_score, amount, rates=(0.01, 0.05, 0.10, 0.20)):
    """Fraud case share and fraud dollar share captured in the top buckets."""
    order = np.argsort(-y_score)
    fraud_amt = amount * y_true
    out = {}
    for r in rates:
        k = max(1, int(np.ceil(r * len(y_score))))
        top = order[:k]
        out[f"top{int(r * 100)}pct"] = {
            "fraud_case_share": float(y_true[top].sum() / max(y_true.sum(), 1)),
            "fraud_dollar_share": float(fraud_amt[top].sum() / max(fraud_amt.sum(), 1e-9)),
        }
    return out


def fallback_evaluate(y_true, y_score, amount=None, threshold: float = 0.5) -> dict:
    """Metric set from the Task #7 spec. Replaced by src.evaluation once merged."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_score)

    metrics = {
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": threshold,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "recall_at_1pct_review": _recall_at_review_rate(y_true, y_score, 0.01),
        "recall_at_5pct_review": _recall_at_review_rate(y_true, y_score, 0.05),
        "ks": float(np.max(tpr_curve - fpr_curve)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if amount is not None:
        metrics["dollar_capture"] = _dollar_capture(y_true, y_score, np.asarray(amount, dtype=float))
    return metrics


def evaluate(y_true, y_score, amount=None, threshold: float = 0.5) -> dict:
    """Use the shared evaluation module when it exists, else the fallback."""
    try:
        from src.evaluation import evaluate as shared_evaluate  # Task #7
    except ImportError:
        return fallback_evaluate(y_true, y_score, amount, threshold)
    try:
        return shared_evaluate(y_true, y_score, amount=amount, threshold=threshold)
    except TypeError:
        return shared_evaluate(y_true, y_score)


# ---------------------------------------------------------------------------
# Results registry
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def log_run(record: dict, registry_path: Path | None = None) -> Path:
    """Append one run to the registry (Task #9). Uses src.registry if merged."""
    try:
        from src.registry import log_run as shared_log_run  # Task #9
        return shared_log_run(record)
    except ImportError:
        pass
    registry_path = Path(registry_path or REPO_ROOT / "results" / "registry.jsonl")
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return registry_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_xgboost(
    df: pd.DataFrame,
    split: pd.Series,
    imbalance: str = "scale_pos_weight",
    params: dict | None = None,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    seed: int = 42,
    use_test: bool = False,
    feature_set: str = "raw_all_v1",
):
    """Train on the train split, early stop on valid, return model and metrics."""
    params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}

    train = df[split == "train"]
    valid = df[split == "valid"]
    test = df[split == "test"] if use_test else None

    drop = [TARGET, ID_COL, TIME_COL, "day", "split"]
    X_train, others, feature_cols, cat_cols = prepare_features(
        train, [valid] + ([test] if use_test else []), drop
    )
    X_valid = others[0]
    y_train, y_valid = train[TARGET].values, valid[TARGET].values

    if imbalance == "scale_pos_weight":
        pos = y_train.sum()
        params["scale_pos_weight"] = float((len(y_train) - pos) / max(pos, 1))
    elif imbalance != "none":
        raise ValueError("imbalance must be 'scale_pos_weight' or 'none'")

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

    start = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
    )
    train_seconds = time.time() - start

    best_range = (0, booster.best_iteration + 1)
    valid_score = booster.predict(dvalid, iteration_range=best_range)
    amount = valid[AMOUNT_COL].values if AMOUNT_COL in valid else None
    results = {"valid": evaluate(y_valid, valid_score, amount=amount)}

    if use_test:
        X_test = others[1]
        dtest = xgb.DMatrix(X_test, label=test[TARGET].values, enable_categorical=True)
        test_score = booster.predict(dtest, iteration_range=best_range)
        results["test"] = evaluate(test[TARGET].values, test_score, amount=test[AMOUNT_COL].values)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "xgboost",
        "task": "#19 baseline",
        "feature_set": feature_set,
        "n_features": len(feature_cols),
        "n_categorical": len(cat_cols),
        "imbalance": imbalance,
        "params": params,
        "best_iteration": int(booster.best_iteration),
        "split_id": split.attrs.get("source", "unknown"),
        "split_sizes": {k: int((split == k).sum()) for k in SPLIT_DAYS},
        "seed": seed,
        "git_commit": _git_commit(),
        "train_seconds": round(train_seconds, 1),
        "metrics": results,
    }
    return booster, record, valid_score


def main(argv=None):
    p = argparse.ArgumentParser(description="XGBoost baseline (Task #19)")
    p.add_argument("--data", type=Path, help="Merged Parquet/CSV from Task #5")
    p.add_argument("--raw-dir", type=Path, help="Folder with the raw Kaggle CSVs")
    p.add_argument("--split-dir", type=Path, help="Folder with {train,valid,test}_ids.csv")
    p.add_argument("--imbalance", choices=["scale_pos_weight", "none"], default="scale_pos_weight")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--early-stopping", type=int, default=100)
    p.add_argument("--gpu", action="store_true", help="Train on CUDA (e.g. Colab)")
    p.add_argument("--use-test", action="store_true", help="Score the sealed blind test split (November only)")
    p.add_argument("--save-model", type=Path, help="Optional path to save the booster (.json)")
    p.add_argument("--registry", type=Path, help="Override registry path")
    args = p.parse_args(argv)

    df = load_merged(args.data, args.raw_dir)
    split = assign_split(df, args.split_dir)
    print(f"Split source: {split.attrs['source']}")
    print(split.value_counts().to_string())

    params = {"device": "cuda"} if args.gpu else {}
    booster, record, _ = train_xgboost(
        df,
        split,
        imbalance=args.imbalance,
        params=params,
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stopping,
        seed=args.seed,
        use_test=args.use_test,
    )

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(args.save_model)

    path = log_run(record, args.registry)
    print(json.dumps(record["metrics"], indent=2))
    print(f"Logged run to {path}")


if __name__ == "__main__":
    main()
