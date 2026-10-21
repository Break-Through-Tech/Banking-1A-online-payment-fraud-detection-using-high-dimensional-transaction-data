"""Every baseline trains through the shared pipeline and lands in the registry (Tasks #17 to #21)."""

import subprocess
import sys

import pandas as pd
import pytest

from src import data, registry, split
from src.models import catboost_baseline, common, lightgbm_baseline, linear_rf_baseline, xgboost_baseline
from tests.synthetic import make_raw

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    raw = root / "raw"
    raw.mkdir()
    t, i = make_raw(n_clients=2500, seed=1)
    t.to_csv(raw / "train_transaction.csv", index=False)
    i.to_csv(raw / "train_identity.csv", index=False)
    pq = data.build_canonical(raw, root / "merged.parquet")
    split.save_split(pd.read_parquet(pq), root / "splits")
    train, valid, sid = common.load_train_valid(pq, root / "splits")
    return {"root": root, "pq": pq, "splits": root / "splits", "reg": root / "reg.jsonl",
            "train": train, "valid": valid, "sid": sid}


def _check(rec, ws, model):
    m = rec["metrics"]["valid"]
    assert rec["model"] == model and rec["split_id"] == ws["sid"]
    assert m["roc_auc"] > 0.6, f"{model} failed to learn the synthetic signal"
    assert m["auprc"] > ws["valid"]["isFraud"].mean()


@pytest.mark.parametrize("model", ["logreg", "rf"])
def test_linear_rf(ws, model):
    kw = {"n_estimators": 50} if model == "rf" else {}
    rec = linear_rf_baseline.run(model, ws["train"], ws["valid"], ws["sid"], registry_path=ws["reg"], **kw)
    _check(rec, ws, model)


def test_lightgbm(ws):
    rec = lightgbm_baseline.run(ws["train"], ws["valid"], ws["sid"], rounds=200, early_stopping=30,
                                registry_path=ws["reg"], params={"num_leaves": 15, "min_child_samples": 20})
    _check(rec, ws, "lightgbm")
    assert rec["params"]["scale_pos_weight"] > 1


def test_catboost(ws):
    rec = catboost_baseline.run(ws["train"], ws["valid"], ws["sid"], registry_path=ws["reg"],
                                params={"iterations": 150, "depth": 4, "verbose": 0})
    _check(rec, ws, "catboost")
    assert "card1" in catboost_baseline.CODE_COLS


def test_xgboost_uses_shared_modules(ws):
    df = pd.read_parquet(ws["pq"])
    s = xgboost_baseline.assign_split(df, ws["splits"])
    assert s.attrs["source"] == ws["sid"]
    _, rec, _ = xgboost_baseline.train_xgboost(df, s, num_boost_round=100, early_stopping_rounds=20)
    xgboost_baseline.log_run(rec, ws["reg"])  # routes through src.registry
    assert "dollar_capture" in rec["metrics"]["valid"] and "fpr_at_1pct_review" in rec["metrics"]["valid"]
    _check(rec, ws, "xgboost")


def test_compare_baselines_report(ws):
    out = ws["root"] / "review.md"
    r = subprocess.run([sys.executable, "scripts/compare_baselines.py", "--registry", str(ws["reg"]),
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    for model in ("logreg", "rf", "lightgbm", "catboost", "xgboost"):
        assert model in text
    assert len(registry.leaderboard(ws["reg"])) == 5
