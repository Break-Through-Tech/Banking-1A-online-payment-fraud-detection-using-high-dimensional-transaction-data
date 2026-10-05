"""Smoke tests for the XGBoost baseline on small synthetic data."""

import numpy as np
import pandas as pd

from src.models import xgboost_baseline as xb


def _synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 183, n)
    amt = rng.lognormal(4, 1, n)
    prod = rng.choice(["W", "C", "H", "R", "S"], n)
    email = rng.choice(["gmail.com", "yahoo.com", "anonymous.com", None], n)
    v1 = rng.normal(size=n)
    logit = -3.5 + 1.5 * (prod == "C") + 1.2 * v1 + 0.8 * (email == "anonymous.com")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + n),
        "TransactionDT": 86_400 + days * 86_400 + rng.integers(0, 86_400, n),
        "isFraud": y,
        "TransactionAmt": amt,
        "ProductCD": prod,
        "P_emaildomain": email,
        "V1": np.where(rng.random(n) < 0.2, np.nan, v1),
    })


def test_split_days_cover_expected_ranges():
    df = _synthetic()
    split = xb.assign_split(df, split_dir="/nonexistent")
    day = xb.add_day_index(df)
    assert split.attrs["source"] == "derived:chrono-v1"
    assert day[split == "train"].max() <= 121
    assert day[split == "valid"].between(122, 151).all()
    assert day[split == "test"].min() >= 152


def test_frozen_split_files_take_priority(tmp_path):
    df = _synthetic()
    ids = df["TransactionID"].values
    pd.DataFrame({"TransactionID": ids[:4000]}).to_csv(tmp_path / "train_ids.csv", index=False)
    pd.DataFrame({"TransactionID": ids[4000:5000]}).to_csv(tmp_path / "valid_ids.csv", index=False)
    pd.DataFrame({"TransactionID": ids[5000:]}).to_csv(tmp_path / "test_ids.csv", index=False)
    split = xb.assign_split(df, split_dir=tmp_path)
    assert split.attrs["source"].startswith(("chrono-v1-", "frozen:"))
    assert (split == "train").sum() == 4000


def test_unseen_categories_become_missing():
    train = pd.DataFrame({"c": ["a", "b"], "x": [1, 2]})
    valid = pd.DataFrame({"c": ["a", "zzz"], "x": [3, 4]})
    _, [Xv], _, cats = xb.prepare_features(train, [valid], drop=[])
    assert cats == ["c"]
    assert pd.isna(Xv["c"].iloc[1])


def test_end_to_end_trains_and_logs(tmp_path):
    df = _synthetic()
    split = xb.assign_split(df, split_dir="/nonexistent")
    booster, record, scores = xb.train_xgboost(df, split, num_boost_round=50, early_stopping_rounds=10)
    m = record["metrics"]["valid"]
    assert "test" not in record["metrics"]  # blind test stays sealed
    assert record["params"]["scale_pos_weight"] > 1
    assert 0 < m["auprc"] <= 1 and m["roc_auc"] > 0.6
    assert set(m["dollar_capture"]) == {"top1pct", "top5pct", "top10pct", "top20pct"}
    path = xb.log_run(record, tmp_path / "registry.jsonl")
    assert path.read_text().count("\n") == 1


def test_metric_edge_values():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    m = xb.fallback_evaluate(y, s, amount=np.array([1, 1, 10, 20]))
    assert m["auprc"] == 1.0 and m["ks"] == 1.0 and m["fpr"] == 0.0
    assert m["recall_at_5pct_review"] == 0.5  # top 1 of 4 rows = 1 of 2 frauds
