"""CatBoost baseline (Sept Task #20, issue #25).

Gradient boosted baseline with native categorical handling, class weighting
for the 1-in-29 imbalance, and early stopping on validation AUPRC.
Fits on the train split, stops on the validation split, never reads test.

    python -m fraud.models.catboost_baseline --sample 50000 --iterations 200   # smoke test
    python -m fraud.models.catboost_baseline                                  # full run
"""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from fraud.categorical import CategoricalEncoder
from fraud.evaluation import evaluate

DATA = Path("data/train_merged_v1.parquet")
RESULTS = Path("results/catboost_task20.json")
REPORT_DIR = Path("reports/catboost_baseline")
DROP = ["TransactionID", "TransactionDT", "isFraud"]
EXPECTED_SIZES = {"train": 420_066, "valid": 85_044}


def assign_split(df):
    """Label each row train / valid / test.

    Uses the frozen split files once PR #39 is merged (fraud.split exists).
    Before that, applies the same day rule the frozen files were built from.
    """
    try:
        from fraud.split import load_split, split_id
    except ImportError:
        day = (df["TransactionDT"] - df["TransactionDT"].min()) // 86400
        labels = np.select([day <= 121, day <= 151], ["train", "valid"], "test")
        return pd.Series(labels, index=df.index), "derived:day-rule"
    labels = np.where(
        df["TransactionID"].isin(load_split("train")), "train",
        np.where(df["TransactionID"].isin(load_split("valid")), "valid", "test"),
    )
    return pd.Series(labels, index=df.index), split_id()


def categorical_columns(X):
    """The same categorical set as Orator's LightGBM baseline (the Task #11 list,
    including number-coded categories), plus any other text columns, which
    CatBoost cannot read as numbers."""
    task11 = [c for c in CategoricalEncoder().required_raw_cols if c in X.columns]
    text_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c not in task11]
    return task11 + text_cols


def prepare(X, cat_cols):
    """CatBoost needs categories as plain strings with no missing values."""
    for c in cat_cols:
        X[c] = X[c].astype(object).fillna("missing").astype(str).astype(object)
    return X


def shared_metrics(y_true, y_score, amount):
    """Call the team's evaluation module, passing amounts if it accepts them."""
    if "amount" in inspect.signature(evaluate).parameters:
        return evaluate(y_true, y_score, amount=amount)
    return evaluate(y_true, y_score)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description="CatBoost baseline (issue #25)")
    p.add_argument("--sample", type=int, default=0, help="train on N random train rows (smoke test)")
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=-1, help="-1 uses every CPU core")
    a = p.parse_args()

    print("Loading", DATA)
    df = pd.read_parquet(DATA)
    split, sid = assign_split(df)
    sizes = {k: int(v) for k, v in split.value_counts().items()}
    print("Split:", sid, sizes)
    for name, n in EXPECTED_SIZES.items():
        if sizes.get(name) != n:
            raise SystemExit(f"ERROR: {name} has {sizes.get(name)} rows, expected {n}. Stop and check the split.")

    train = df[split == "train"]
    valid = df[split == "valid"]
    del df
    if a.sample:
        train = train.sample(n=a.sample, random_state=a.seed)
        print(f"Smoke test: training on {len(train):,} sampled rows")

    X_tr, y_tr = train.drop(columns=DROP), train["isFraud"].to_numpy()
    X_va, y_va = valid.drop(columns=DROP), valid["isFraud"].to_numpy()
    cat_cols = categorical_columns(X_tr)
    X_tr, X_va = prepare(X_tr, cat_cols), prepare(X_va, cat_cols)
    print(f"Features: {X_tr.shape[1]} ({len(cat_cols)} categorical)")

    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    model = CatBoostClassifier(
        iterations=a.iterations,
        learning_rate=a.learning_rate,
        depth=a.depth,
        loss_function="Logloss",
        eval_metric="PRAUC:use_weights=false",
        scale_pos_weight=spw,
        od_type="Iter",
        od_wait=200,
        use_best_model=True,
        random_seed=a.seed,
        thread_count=a.threads,
        verbose=100,
    )

    t0 = time.time()
    model.fit(Pool(X_tr, y_tr, cat_features=cat_cols), eval_set=Pool(X_va, y_va, cat_features=cat_cols))
    secs = time.time() - t0

    score = model.predict_proba(X_va)[:, 1]
    metrics = shared_metrics(y_va, score, valid["TransactionAmt"].to_numpy())

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"TransactionID": valid["TransactionID"].to_numpy(), "score": score}).to_parquet(
        REPORT_DIR / ("smoke_valid_scores.parquet" if a.sample else "valid_scores.parquet"), index=False
    )
    if not a.sample:
        model.save_model(str(REPORT_DIR / "catboost_baseline.cbm"))

    record = {
        "model": "catboost",
        "task": "#20 baseline (issue #25)",
        "feature_set": "native_raw_all_v1",
        "split_id": sid,
        "split_sizes": sizes,
        "n_features": int(X_tr.shape[1]),
        "n_categorical": len(cat_cols),
        "imbalance": "scale_pos_weight",
        "seed": a.seed,
        "params": {
            "iterations": a.iterations,
            "learning_rate": a.learning_rate,
            "depth": a.depth,
            "scale_pos_weight": round(spw, 2),
            "early_stopping_rounds": 200,
        },
        "best_iteration": int(model.get_best_iteration()),
        "train_seconds": round(secs, 1),
        "metrics": {"valid": metrics},
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": f"smoke test on {a.sample} train rows" if a.sample else "",
    }
    out = REPORT_DIR / "smoke_test.json" if a.sample else RESULTS
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float))

    print(f"\nTrained in {secs / 60:.1f} min, best iteration {record['best_iteration']}")
    print("Validation metrics:", json.dumps(metrics, indent=2, default=float))
    print("Saved", out)


if __name__ == "__main__":
    main()