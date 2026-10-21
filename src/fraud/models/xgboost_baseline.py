"""
XGBoost baseline for the Banking-1A fraud detection project (Task #19).

What it does
------------
1. Loads the merged transaction + identity data (the canonical Parquet from
   Task #5, or the raw Kaggle CSVs as a fallback).
2. Applies the frozen chronological split from Task #6 (TransactionID lists
   in splits/). If those files are not there yet, it derives the same
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
    python -m fraud.models.xgboost_baseline                      # uses data/train_merged_v1.parquet
    python -m fraud.models.xgboost_baseline --feature-set fe_v2 --learning-rate 0.1
    python -m fraud.models.xgboost_baseline --raw-dir data/raw --imbalance none
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
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
    # Cap the histogram cache. It doesn't change the model, only memory: deep
    # trees (depth 10+) otherwise cache one histogram per node and the full-data
    # fit ran out of memory on an 8 GB machine.
    "max_cached_hist_node": 256,
}

from fraud.config import DATA_DIR, PARQUET_PATH, REPO_ROOT, SPLITS_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _float32_dtypes(path: Path) -> dict:
    """Read the anonymized V, C, D, dist and numeric id columns as float32.

    Read as float64, train_transaction.csv alone takes about 2 GB, and the
    merge copies it again, which runs out of memory on 8 GB machines and
    free Colab. float32 halves that with no effect on tree splits.
    """
    header = pd.read_csv(path, nrows=0).columns
    pattern = re.compile(r"^(V|C|D|dist|id[_-]0)\d+$")
    return {c: "float32" for c in header if pattern.match(c)}


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

    if data_path is None and raw_dir is None and PARQUET_PATH.exists():
        data_path = PARQUET_PATH  # canonical file from scripts/build_dataset.py
    raw_dir = Path(raw_dir or DATA_DIR / "raw")
    trans = pd.read_csv(raw_dir / "train_transaction.csv", dtype=_float32_dtypes(raw_dir / "train_transaction.csv"))
    ident = pd.read_csv(raw_dir / "train_identity.csv", dtype=_float32_dtypes(raw_dir / "train_identity.csv"))
    ident.columns = [c.replace("id-", "id_") for c in ident.columns]
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
    split_dir = Path(split_dir or SPLITS_DIR)
    files = {name: split_dir / f"{name}_ids.csv" for name in SPLIT_DAYS}

    if all(f.exists() for f in files.values()):
        split = pd.Series("unassigned", index=df.index)
        for name, f in files.items():
            ids = pd.read_csv(f)[ID_COL]
            split[df[ID_COL].isin(ids)] = name
        try:
            from fraud.split import split_id  # Task #6: same id every model logs
            source = split_id(split_dir)
        except ImportError:
            # Same hash fraud.split.split_id() produces, so runs stay comparable.
            h = hashlib.sha256()
            for name in SPLIT_DAYS:
                h.update(files[name].read_bytes())
            source = "chrono-v1-" + h.hexdigest()[:10]
    else:
        day = add_day_index(df)
        split = pd.Series("unassigned", index=df.index)
        for name, (lo, hi) in SPLIT_DAYS.items():
            split[(day >= lo) & (day <= hi)] = name
        source = "derived:chrono-v1"

    split.attrs["source"] = source
    return split


# ---------------------------------------------------------------------------
# Feature set fe_v2 (all statistics fitted on the train split only)
# ---------------------------------------------------------------------------

FREQ_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2",
             "P_emaildomain", "R_emaildomain", "DeviceInfo", "id_31"]
# D columns that count days since a past event, so they grow with calendar time.
# day - D turns them into a fixed point in time (literature review, section 1.4).
D_NORMALIZE = ["D1", "D2", "D4", "D10", "D11", "D15"]
FEATURE_SETS = ("raw_all_v1", "fe_v2")


def add_features_v2(train: pd.DataFrame, others: list[pd.DataFrame]):
    """Add leakage-safe features. Nothing here looks at labels, and frequency
    counts come from the train split only, so validation rows never inform
    their own features."""
    # Callers pass fresh split slices, so columns are added in place to avoid
    # another full copy of the data (memory is the limit on 8 GB machines).
    for col in FREQ_COLS:
        if col not in train:
            continue
        counts = train[col].astype("string").value_counts(dropna=True)
        for frame in [train, *others]:
            frame[f"{col}_freq"] = (frame[col].astype("string").map(counts).astype("float32")
                                    .fillna(0).where(frame[col].notna()))
    for frame in [train, *others]:
        day = frame[TIME_COL] / SECONDS_PER_DAY
        frame["hour"] = ((frame[TIME_COL] // 3600) % 24).astype("int8")
        frame["amt_cents"] = (frame[AMOUNT_COL] - np.floor(frame[AMOUNT_COL])).round(3).astype("float32")
        frame["amt_log"] = np.log1p(frame[AMOUNT_COL]).astype("float32")
        for d in D_NORMALIZE:
            if d in frame:
                frame[f"{d}_norm"] = (day - frame[d]).astype("float32")
                frame.drop(columns=d, inplace=True)
    return train, others


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
    top = np.argsort(-y_score, kind="stable")[:k]
    total = y_true.sum()
    return float(y_true[top].sum() / total) if total else float("nan")


def _fpr_at_review_rate(y_true: np.ndarray, y_score: np.ndarray, rate: float) -> float:
    """Share of legitimate customers flagged when reviewing the top `rate`."""
    k = max(1, int(np.ceil(rate * len(y_score))))
    top = np.zeros(len(y_score), dtype=bool)
    top[np.argsort(-y_score, kind="stable")[:k]] = True
    neg = y_true == 0
    return float(top[neg].mean()) if neg.any() else float("nan")


def _dollar_capture(y_true, y_score, amount, rates=(0.01, 0.05, 0.10, 0.20)):
    """Fraud case share and fraud dollar share captured in the top buckets."""
    order = np.argsort(-y_score, kind="stable")
    fraud_amt = amount * y_true
    out = {}
    for r in rates:
        k = max(1, int(np.ceil(r * len(y_score))))
        top = order[:k]
        case_share = float(y_true[top].sum() / max(y_true.sum(), 1))
        dollar_share = float(fraud_amt[top].sum() / max(fraud_amt.sum(), 1e-9))
        # Key names match fraud.evaluation (Task #7) so registry records line up.
        out[f"top_{int(r * 100)}pct"] = {
            "fraud_cases_captured": int(y_true[top].sum()),
            "fraud_case_capture_rate": case_share,
            "fraud_dollars_captured": float(fraud_amt[top].sum()),
            "fraud_dollar_capture_rate": dollar_share,
            "fraud_case_share": case_share,      # older name, kept for earlier records
            "fraud_dollar_share": dollar_share,
        }
    return out


def fallback_evaluate(y_true, y_score, amount=None, threshold: float = 0.5) -> dict:
    """Metric set from the Task #7 spec. Replaced by fraud.evaluation once merged."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_score)

    metrics = {
        "n": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": threshold,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "recall_at_1pct_review": _recall_at_review_rate(y_true, y_score, 0.01),
        "recall_at_5pct_review": _recall_at_review_rate(y_true, y_score, 0.05),
        "fpr_at_1pct_review": _fpr_at_review_rate(y_true, y_score, 0.01),
        "fpr_at_5pct_review": _fpr_at_review_rate(y_true, y_score, 0.05),
        "ks": float(np.max(tpr_curve - fpr_curve)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if amount is not None:
        metrics["dollar_capture"] = _dollar_capture(y_true, y_score, np.asarray(amount, dtype=float))
    return metrics


def evaluate(y_true, y_score, amount=None, threshold: float = 0.5) -> dict:
    """Always compute the full metric set locally, and attach the shared
    module's numbers (Task #7) under "shared" when it is available, so the
    two can be cross-checked. A broken or missing shared module never stops
    a training run."""
    metrics = fallback_evaluate(y_true, y_score, amount, threshold)
    try:
        import fraud.evaluation as shared  # Task #7
    except Exception as exc:  # ImportError, or SyntaxError in a half-merged module
        metrics["shared_evaluation"] = f"unavailable: {type(exc).__name__}"
        return metrics
    fn = getattr(shared, "evaluate", None) or getattr(shared, "evaluate_model", None)
    if fn is None:
        metrics["shared_evaluation"] = "unavailable: no evaluate function"
        return metrics
    try:
        metrics["shared"] = fn(np.asarray(y_true), np.asarray(y_score))
    except Exception as exc:
        metrics["shared_evaluation"] = f"failed: {exc}"
    return metrics


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
    """Append one run to the registry (Task #9). Uses fraud.registry if merged."""
    try:
        from fraud.registry import log_run as shared_log_run  # Task #9
        return shared_log_run(record, registry_path)
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
    log_train_metric: bool = False,
):
    """Train on the train split, early stop on valid, return model and metrics."""
    params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}

    train = df[split == "train"]
    valid = df[split == "valid"]
    test = df[split == "test"] if use_test else None

    drop = [TARGET, ID_COL, TIME_COL, "day", "split"]
    feat_train, feat_others = train, [valid] + ([test] if use_test else [])
    if feature_set == "fe_v2":
        feat_train, feat_others = add_features_v2(feat_train, feat_others)
        # Cap how many categories one split may group, to stop the model
        # memorizing rare DeviceInfo / email values.
        params.setdefault("max_cat_threshold", 32)
    elif feature_set != "raw_all_v1":
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    X_train, others, feature_cols, cat_cols = prepare_features(feat_train, feat_others, drop)
    del feat_train, feat_others
    X_valid = others[0]
    y_train, y_valid = train[TARGET].values, valid[TARGET].values

    if imbalance == "scale_pos_weight":
        pos = y_train.sum()
        params["scale_pos_weight"] = float((len(y_train) - pos) / max(pos, 1))
    elif imbalance != "none":
        raise ValueError("imbalance must be 'scale_pos_weight' or 'none'")

    # Free the pandas copies before training. On the full data the first
    # version kept ~3 GB of DataFrames alive and was killed for running out of
    # memory on an 8 GB machine. (QuantileDMatrix uses less memory but trained
    # about 3x slower with our categorical columns, so we keep DMatrix.)
    n_features, n_cat = len(feature_cols), len(cat_cols)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
    X_test = others[1] if use_test else None
    del X_train, X_valid, others, train
    gc.collect()

    evals = [(dvalid, "valid")]  # scoring the train set every round doubles training time
    if log_train_metric:
        evals.insert(0, (dtrain, "train"))

    start = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
    )
    del dtrain
    gc.collect()
    train_seconds = time.time() - start

    best_range = (0, booster.best_iteration + 1)
    valid_score = booster.predict(dvalid, iteration_range=best_range)
    amount = valid[AMOUNT_COL].values if AMOUNT_COL in valid else None
    results = {"valid": evaluate(y_valid, valid_score, amount=amount)}

    if use_test:
        dtest = xgb.DMatrix(X_test, label=test[TARGET].values, enable_categorical=True)
        test_score = booster.predict(dtest, iteration_range=best_range)
        results["test"] = evaluate(test[TARGET].values, test_score, amount=test[AMOUNT_COL].values)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "xgboost",
        "task": "#19 baseline",
        "feature_set": feature_set,
        "n_features": n_features,
        "n_categorical": n_cat,
        "imbalance": imbalance,
        "params": params,
        "best_iteration": int(booster.best_iteration),
        "hit_round_limit": bool(booster.best_iteration + 1 >= num_boost_round),
        "split_id": split.attrs.get("source", "unknown"),
        "split_sizes": {k: int((split == k).sum()) for k in SPLIT_DAYS},
        "seed": seed,
        "git_commit": _git_commit(),
        "train_seconds": round(train_seconds, 1),
        "metrics": results,
    }
    return booster, record, valid_score


def write_report(booster, valid: pd.DataFrame, valid_score, out_dir: Path) -> None:
    """Capture table by risk bucket and the 30 most important features (by gain)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = valid[TARGET].values.astype(int)
    amt = valid[AMOUNT_COL].values.astype(float)
    order = np.argsort(-valid_score, kind="stable")
    rows = []
    for r in (0.01, 0.05, 0.10, 0.20):
        top = order[: max(1, int(np.ceil(r * len(order))))]
        rows.append({
            "bucket": f"top {r:.0%}",
            "transactions": len(top),
            "fraud_cases": int(y[top].sum()),
            "fraud_case_share": y[top].sum() / y.sum(),
            "fraud_dollar_share": (amt[top] * y[top]).sum() / (amt * y).sum(),
            "precision": y[top].mean(),
        })
    pd.DataFrame(rows).to_csv(out_dir / "xgboost_capture.csv", index=False)
    gain = pd.Series(booster.get_score(importance_type="total_gain"), name="total_gain")
    (gain / gain.sum()).sort_values(ascending=False).head(30).rename("gain_share").to_csv(
        out_dir / "xgboost_top_features.csv")


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
    p.add_argument("--learning-rate", type=float, help="Override learning rate (0.1 is ~2x faster on CPU)")
    p.add_argument("--report-dir", type=Path, help="Write the dollar capture table and top features here")
    p.add_argument("--notes", default="", help="Free text stored with the registry record")
    p.add_argument("--feature-set", choices=FEATURE_SETS, default="raw_all_v1",
                   help="raw_all_v1 = every raw column; fe_v2 = + frequency, time-of-day, amount and normalized D features")
    args = p.parse_args(argv)

    df = load_merged(args.data, args.raw_dir)
    split = assign_split(df, args.split_dir)
    print(f"Split source: {split.attrs['source']}")
    print(split.value_counts().to_string())

    params = {"device": "cuda"} if args.gpu else {}
    if args.learning_rate:
        params["learning_rate"] = args.learning_rate
    booster, record, valid_score = train_xgboost(
        df,
        split,
        imbalance=args.imbalance,
        params=params,
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stopping,
        seed=args.seed,
        use_test=args.use_test,
        feature_set=args.feature_set,
    )

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(args.save_model)

    if args.report_dir:
        write_report(booster, df[split == "valid"], valid_score, args.report_dir)

    record["notes"] = args.notes
    path = log_run(record, args.registry)
    print(json.dumps(record["metrics"], indent=2))
    print(f"Logged run to {path}")


if __name__ == "__main__":
    main()
