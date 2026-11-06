"""Baseline ensemble and stacking (Oct Task #17, issue #53).

Combines saved validation scores from several models:
  * each model on its own,
  * averages that weight every model equally (nothing is fitted),
  * logistic stacks that learn a weight per model.
Each combination is built two ways: on percentile ranks, and on log-odds
(which keeps how confident each model is; Mohit's quick check found log-odds
blending worked better than ranks).

The stack is fitted on the first half of the validation window (days 122 to 136)
and every candidate is compared on the second half (days 137 to 151), so the
stack is never scored on rows it learned from. The test split is never read.

    python -m fraud.models.ensemble --scores xgb=PATH cat=PATH [lgbm=PATH ...]

Each PATH is either a .parquet file with TransactionID and score columns, or a
.npy array of validation scores in TransactionID order (how Mohit's scripts save).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fraud.evaluation import evaluate
from fraud.split import load_split, split_id

DATA = Path("data/train_merged_v1.parquet")
RESULTS = Path("results/ensemble_task17.json")
REGISTRY = Path("results/registry.jsonl")
REPORT_DIR = Path("reports/ensemble")
STACK_FIT_LAST_DAY = 136  # days 122-136 fit the stack, days 137-151 judge everything


def load_valid():
    """Validation rows in TransactionID order, with label, amount and day."""
    df = pd.read_parquet(DATA, columns=["TransactionID", "TransactionDT", "isFraud", "TransactionAmt"])
    df["day"] = (df["TransactionDT"] - df["TransactionDT"].min()) // 86400
    valid = df[df["TransactionID"].isin(load_split("valid"))]
    return valid.sort_values("TransactionID").set_index("TransactionID")


def load_scores(path, valid_ids):
    """Read one model's validation scores and line them up by TransactionID."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
        if len(arr) != len(valid_ids):
            raise SystemExit(f"ERROR: {path} has {len(arr)} scores, expected {len(valid_ids)}.")
        return pd.Series(arr, index=valid_ids)
    frame = pd.read_parquet(path)
    s = frame.set_index("TransactionID")["score"]
    missing = valid_ids.difference(s.index)
    if len(missing):
        raise SystemExit(f"ERROR: {path} is missing {len(missing)} validation rows.")
    return s.reindex(valid_ids)


def to_logodds(s):
    """Probabilities become log-odds; scores already outside 0-1 are kept as they are."""
    if s.min() >= 0 and s.max() <= 1:
        p = s.clip(1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return s


def fit_stack(features, fit, y):
    stack = LogisticRegression(max_iter=1000)
    stack.fit(features[fit].to_numpy(), y[fit])
    weights = {n: round(float(w), 3) for n, w in zip(features.columns, stack.coef_[0])}
    return pd.Series(stack.predict_proba(features.to_numpy())[:, 1], index=features.index), weights


def metrics_for(rows, score):
    return evaluate(rows["isFraud"].to_numpy(), np.asarray(score), amount=rows["TransactionAmt"].to_numpy())


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description="Ensemble and stacking (issue #53)")
    p.add_argument("--scores", nargs="+", required=True, help="name=path pairs, at least two")
    p.add_argument("--registry", action="store_true", help="also append the result to results/registry.jsonl")
    a = p.parse_args()

    valid = load_valid()
    ids = valid.index
    print(f"Validation rows: {len(valid):,} (split {split_id()})")

    scores = {}
    for pair in a.scores:
        name, path = pair.split("=", 1)
        scores[name] = load_scores(path, ids)
        print(f"Loaded {name}: {path}")
    if len(scores) < 2:
        raise SystemExit("ERROR: an ensemble needs at least two models.")
    names = list(scores)

    # Percentile ranks put every model on the same 0-to-1 scale.
    ranks = pd.DataFrame({n: s.rank(pct=True) for n, s in scores.items()})
    print("\nHow similar the models' rankings are (1.00 = identical):")
    print(ranks.corr(method="spearman").round(3).to_string())

    fit = (valid["day"] <= STACK_FIT_LAST_DAY).to_numpy()
    judge = ~fit
    print(f"\nStack fitted on {fit.sum():,} rows (days 122-{STACK_FIT_LAST_DAY}); "
          f"all candidates judged on {judge.sum():,} rows (days {STACK_FIT_LAST_DAY + 1}-151)")

    logits = pd.DataFrame({n: to_logodds(s) for n, s in scores.items()})
    y = valid["isFraud"].to_numpy()
    rank_stack, rank_weights = fit_stack(ranks, fit, y)
    logit_stack, logit_weights = fit_stack(logits, fit, y)
    weights = {"rank_stack": rank_weights, "logodds_stack": logit_weights}

    candidates = {n: ranks[n] for n in names}
    candidates["rank_average"] = ranks.mean(axis=1)
    candidates["rank_stack"] = rank_stack
    candidates["logodds_average"] = logits.mean(axis=1)
    candidates["logodds_stack"] = logit_stack

    judged = valid[judge]
    results = {}
    print("\nCandidate          AUPRC   recall@5%   fraud $ @5%")
    for name, s in candidates.items():
        m = metrics_for(judged, s[judge])
        results[name] = m
        dollars = m.get("dollar_capture", {}).get("top_5pct", {}).get("fraud_dollar_capture_rate", float("nan"))
        print(f"{name:<18} {m['auprc']:.4f}   {m['recall_at_5pct_review']:.4f}      {dollars:.4f}")
    print("\nStack weights:", weights)

    best = max(results, key=lambda n: results[n]["auprc"])
    print(f"Best on the judging half: {best}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    combos = ["rank_average", "rank_stack", "logodds_average", "logodds_stack"]
    pd.DataFrame({"TransactionID": ids, **{c: candidates[c].to_numpy() for c in combos}}).to_parquet(
        REPORT_DIR / "valid_scores.parquet", index=False)

    record = {
        "model": "ensemble",
        "task": "Oct #17 ensemble / stacking (issue #53)",
        "feature_set": "model_scores",
        "split_id": split_id(),
        "split_sizes": {"stack_fit": int(fit.sum()), "judge": int(judge.sum())},
        "inputs": {n: str(path) for n, path in (pair.split("=", 1) for pair in a.scores)},
        "spearman": ranks.corr(method="spearman").round(4).to_dict(),
        "stack_weights": weights,
        "best_on_judge_half": best,
        "metrics": {"judge_half": results},
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": "All candidates scored on validation days 137-151; stack fitted on days 122-136.",
    }
    RESULTS.write_text(json.dumps(record, indent=2, default=float))
    if a.registry and REGISTRY.exists():
        with REGISTRY.open("a") as f:
            f.write(json.dumps(record, default=float) + "\n")
        print("Appended to", REGISTRY)
    print("Saved", RESULTS)


if __name__ == "__main__":
    main()