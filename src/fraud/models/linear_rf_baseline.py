"""Logistic regression and random forest baselines (Sept Task #17, issue #22).

The performance floor every other model must clear. Both models use the same
simple features, built from the train split only:
  * numeric columns as they are (missing values filled with the train median
    for logistic regression; the random forest handles missing values itself),
  * the Task #11 categorical columns replaced by how often each value appears
    in training (frequency encoding; unseen values get 0).

    python -m fraud.models.linear_rf_baseline --sample 50000      # smoke test
    python -m fraud.models.linear_rf_baseline                     # full run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fraud.categorical import CategoricalEncoder
from fraud.evaluation import evaluate
from fraud.split import load_split, split_id

DATA = Path("data/train_merged_v1.parquet")
RESULTS = Path("results/linear_rf_task17.json")
REGISTRY = Path("results/registry.jsonl")
REPORT_DIR = Path("reports/linear_rf_baseline")
DROP = ["TransactionID", "TransactionDT", "isFraud"]
EXPECTED_SIZES = {"train": 420_066, "valid": 85_044}


def frequency_encode(train, valid, cat_cols):
    """Replace each category with its share of training rows (fit on train only)."""
    for c in cat_cols:
        freq = train[c].astype(object).fillna("missing").value_counts(normalize=True)
        for frame in (train, valid):
            frame[c] = frame[c].astype(object).fillna("missing").map(freq).fillna(0.0).astype("float32")
    return train, valid


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description="Logistic regression and random forest baselines (issue #22)")
    p.add_argument("--sample", type=int, default=0, help="train on N random train rows (smoke test)")
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    print("Loading", DATA)
    df = pd.read_parquet(DATA)
    train = df[df["TransactionID"].isin(load_split("train"))]
    valid = df[df["TransactionID"].isin(load_split("valid"))]
    del df
    sizes = {"train": len(train), "valid": len(valid)}
    print("Split:", split_id(), sizes)
    if sizes != EXPECTED_SIZES:
        raise SystemExit(f"ERROR: split sizes {sizes}, expected {EXPECTED_SIZES}. Stop and check the split.")
    if a.sample:
        train = train.sample(n=a.sample, random_state=a.seed)
        print(f"Smoke test: training on {len(train):,} sampled rows")

    X_tr, y_tr = train.drop(columns=DROP).copy(), train["isFraud"].to_numpy()
    X_va, y_va = valid.drop(columns=DROP).copy(), valid["isFraud"].to_numpy()
    amount = valid["TransactionAmt"].to_numpy()

    task11 = [c for c in CategoricalEncoder().required_raw_cols if c in X_tr.columns]
    text_cols = [c for c in X_tr.columns if not pd.api.types.is_numeric_dtype(X_tr[c]) and c not in task11]
    cat_cols = task11 + text_cols
    X_tr, X_va = frequency_encode(X_tr, X_va, cat_cols)
    X_tr, X_va = X_tr.astype("float32"), X_va.astype("float32")
    print(f"Features: {X_tr.shape[1]} ({len(cat_cols)} frequency-encoded)")

    base = {
        "task": "Sept #17 baseline (issue #22)",
        "feature_set": "freq_encoded_v1",
        "split_id": split_id(),
        "split_sizes": {"train": int(len(X_tr)), "valid": int(len(X_va))},
        "n_features": int(X_tr.shape[1]),
        "n_categorical": len(cat_cols),
        "imbalance": "class_weight=balanced",
        "seed": a.seed,
        "git_commit": git_commit(),
        "notes": f"smoke test on {a.sample} train rows" if a.sample else "",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if a.sample else ""

    # Logistic regression: needs no missing values and features on one scale.
    medians = X_tr.median()
    scaler = StandardScaler().fit(X_tr.fillna(medians))
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    t0 = time.time()
    lr.fit(scaler.transform(X_tr.fillna(medians)), y_tr)
    lr_secs = time.time() - t0
    lr_score = lr.predict_proba(scaler.transform(X_va.fillna(medians)))[:, 1]
    lr_metrics = evaluate(y_va, lr_score, amount=amount)
    print(f"\nLogistic regression: AUPRC {lr_metrics['auprc']:.4f} ({lr_secs / 60:.1f} min)")

    # Random forest: handles missing values directly (scikit-learn 1.4 or newer).
    rf = RandomForestClassifier(
        n_estimators=a.trees, max_features="sqrt", min_samples_leaf=50, max_samples=0.5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=a.seed,
    )
    t0 = time.time()
    rf.fit(X_tr, y_tr)
    rf_secs = time.time() - t0
    rf_score = rf.predict_proba(X_va)[:, 1]
    rf_metrics = evaluate(y_va, rf_score, amount=amount)
    print(f"Random forest: AUPRC {rf_metrics['auprc']:.4f} ({rf_secs / 60:.1f} min)")

    ids = valid["TransactionID"].to_numpy()
    pd.DataFrame({"TransactionID": ids, "score": lr_score}).to_parquet(
        REPORT_DIR / f"lr_valid_scores{suffix}.parquet", index=False)
    pd.DataFrame({"TransactionID": ids, "score": rf_score}).to_parquet(
        REPORT_DIR / f"rf_valid_scores{suffix}.parquet", index=False)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lr_record = {**base, "model": "logistic_regression",
                 "params": {"max_iter": 1000, "imputation": "train median", "scaling": "standard"},
                 "train_seconds": round(lr_secs, 1), "metrics": {"valid": lr_metrics}, "timestamp_utc": now}
    rf_record = {**base, "model": "random_forest",
                 "params": {"n_estimators": a.trees, "max_features": "sqrt", "min_samples_leaf": 50,
                            "max_samples": 0.5, "class_weight": "balanced_subsample"},
                 "train_seconds": round(rf_secs, 1), "metrics": {"valid": rf_metrics}, "timestamp_utc": now}

    out = REPORT_DIR / "smoke_test.json" if a.sample else RESULTS
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"logistic_regression": lr_record, "random_forest": rf_record}, indent=2, default=float))
    if not a.sample and REGISTRY.exists():
        with REGISTRY.open("a") as f:
            f.write(json.dumps(lr_record, default=float) + "\n")
            f.write(json.dumps(rf_record, default=float) + "\n")
        print("Appended both runs to", REGISTRY)
    print("Saved", out)


if __name__ == "__main__":
    main()