"""End-to-end tests for the September pipeline on synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from src import data, entity, eda, evaluation, registry, split
from tests.synthetic import make_raw


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    raw = root / "raw"
    raw.mkdir()
    trans, ident = make_raw()
    trans.to_csv(raw / "train_transaction.csv", index=False)
    ident.rename(columns={"id_01": "id-01"}).to_csv(raw / "train_identity.csv", index=False)
    parquet = data.build_canonical(raw, root / "merged.parquet")
    split.save_split(pd.read_parquet(parquet), root / "splits")
    return {"root": root, "raw": raw, "parquet": parquet, "splits": root / "splits",
            "registry": root / "registry.jsonl", "n_trans": len(trans), "n_ident": len(ident)}


# --- Tasks #4 and #5 -------------------------------------------------------

def test_merge_keeps_every_transaction(workspace):
    df = pd.read_parquet(workspace["parquet"])
    assert len(df) == workspace["n_trans"]
    assert df["has_identity"].sum() == workspace["n_ident"]
    assert "id_01" in df  # id-01 normalized
    assert df.loc[df["has_identity"] == 0, "DeviceType"].isna().all()


def test_merge_rejects_duplicate_ids():
    t = pd.DataFrame({"TransactionID": [1, 1]})
    with pytest.raises(ValueError):
        data.merge_tables(t, pd.DataFrame({"TransactionID": [1]}))


def test_optimize_memory_shrinks_and_preserves_values():
    df = pd.DataFrame({"a": np.arange(1000, dtype="int64"), "b": np.random.rand(1000),
                       "c": ["x", "y"] * 500, "TransactionAmt": np.random.rand(1000)})
    out = data.optimize_memory(df, verbose=False)
    assert out.memory_usage(deep=True).sum() < df.memory_usage(deep=True).sum()
    assert out["a"].dtype.itemsize < 8 and str(out["c"].dtype) == "category"
    assert out["TransactionAmt"].dtype == "float64"
    np.testing.assert_allclose(out["b"], df["b"], rtol=1e-6)


# --- Task #6 ---------------------------------------------------------------

def test_split_is_chronological_disjoint_and_reproducible(workspace, tmp_path):
    df = pd.read_parquet(workspace["parquet"])
    ids = {n: set(split.load_split(n, workspace["splits"])) for n in split.SPLIT_DAYS}
    assert not (ids["train"] & ids["valid"]) and not (ids["valid"] & ids["test"])
    assert sum(len(v) for v in ids.values()) == len(df)
    t = df.set_index("TransactionID")["TransactionDT"]
    assert t[list(ids["train"])].max() < t[list(ids["valid"])].min()
    assert t[list(ids["valid"])].max() < t[list(ids["test"])].min()
    split.save_split(df, tmp_path)
    assert split.split_id(tmp_path) == split.split_id(workspace["splits"])


def test_load_data_by_split(workspace):
    v = data.load_data(workspace["parquet"], "valid", workspace["splits"])
    assert len(v) == len(split.load_split("valid", workspace["splits"]))


# --- Tasks #7 and #12 ------------------------------------------------------

def test_evaluate_known_values():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    s = np.array([.1, .2, .3, .4, .5, .6, .7, .95, .9, .99])
    m = evaluation.evaluate(y, s, amount=np.arange(1, 11), threshold=0.85)
    assert m["confusion"] == {"tn": 7, "fp": 1, "fn": 0, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == 1.0
    assert m["fpr"] == pytest.approx(1 / 8)
    assert m["recall_at_1pct_review"] == 0.5  # top 1 row
    assert m["ks"] == pytest.approx(1 - 1 / 8)
    cap = evaluation.dollar_capture_table(y, s, np.arange(1, 11))
    assert cap.loc[0, "fraud_dollar_share"] == pytest.approx(10 / 19)


def test_evaluate_rejects_bad_input():
    with pytest.raises(ValueError):
        evaluation.evaluate([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        evaluation.evaluate([0, 1], [0.1, np.nan])


def test_plots_written(tmp_path):
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    s = np.clip(y * 0.4 + rng.random(500) * 0.6, 0, 1)
    evaluation.save_standard_report(y, s, rng.random(500) * 100, tmp_path, "demo")
    assert {p.name for p in tmp_path.iterdir()} == {
        "demo_capture.csv", "demo_pr.png", "demo_confusion.png", "demo_scores.png"}


# --- Task #9 ---------------------------------------------------------------

def test_registry_requires_fields_and_appends(tmp_path):
    reg = tmp_path / "r.jsonl"
    with pytest.raises(ValueError):
        registry.log_run({"model": "x"}, reg)
    rec = {"model": "m", "feature_set": "f_v1", "params": {}, "split_id": "s", "seed": 1,
           "metrics": {"valid": {"auprc": 0.5, "roc_auc": 0.8, "recall_at_1pct_review": 0.1,
                                 "recall_at_5pct_review": 0.3, "ks": 0.4}}}
    registry.log_run(rec, reg)
    registry.log_run({**rec, "metrics": {"valid": {**rec["metrics"]["valid"], "auprc": 0.6}}}, reg)
    lines = reg.read_text().splitlines()
    assert len(lines) == 2 and "run_id" in json.loads(lines[0])
    assert registry.leaderboard(reg)["auprc"].iloc[0] == 0.6


# --- Tasks #8, #10, #16 ----------------------------------------------------

def test_eda_tables(workspace):
    df = pd.read_parquet(workspace["parquet"])
    miss = eda.missingness_by_column(df)
    assert miss.set_index("column").loc["V1", "group"] == "V"
    assert {"fraud_lift", "chi2_p"} <= set(miss.columns)
    blocks = eda.missing_pattern_blocks(df, "V")
    assert sorted(blocks["n_columns"]) == [3, 3]
    prof = eda.categorical_profile(df)
    assert "ProductCD" in prof["column"].values
    trend = eda.fraud_rate_trend_test(df)
    assert 0 <= trend["p_value"] <= 1
    drift = eda.feature_drift(df, columns=["TransactionAmt", "ProductCD", "D1"])
    assert drift.set_index("column").loc["D1", "psi"] > 0.1  # D1 grows with time


def test_psi_zero_for_identical():
    s = pd.Series(np.random.default_rng(0).normal(size=1000))
    assert eda.psi(s, s) == pytest.approx(0, abs=1e-9)


# --- Task #15 --------------------------------------------------------------

def test_entity_feasibility_finds_magic_uid(workspace):
    df = pd.read_parquet(workspace["parquet"])
    labels = split.make_split(df)
    rep = entity.feasibility_report(df, labels)
    assert rep.loc["uid_magic", "d1_consistency"] > 0.95
    assert rep.loc["uid_magic", "label_purity"] >= rep.loc["uid_card", "label_purity"]
    assert "uid_magic" in entity.verdict(rep)
