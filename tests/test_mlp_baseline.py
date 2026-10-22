"""Tabular MLP baseline (Oct Task #14) on small synthetic data."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fraud.models import mlp_baseline as mlp  # noqa: E402
from tests.test_xgboost_baseline import _synthetic  # noqa: E402


def test_preprocessor_fits_on_train_only():
    tr = pd.DataFrame({"card1": [1.0] * 30 + [2.0] * 5, "ProductCD": ["W"] * 35, "V1": [np.nan] * 5 + [1.0] * 30,
                       "V2": [np.nan] * 5 + [2.0] * 30, "C1": np.arange(35.0), "isFraud": 0})
    va = pd.DataFrame({"card1": [1.0, 2.0, 3.0, np.nan], "ProductCD": ["W", "X", None, "W"],
                       "V1": [np.nan, 1, 1, 1], "V2": [np.nan, 2, 2, 2], "C1": [0.0, 1, 2, 3], "isFraud": 0})
    pre = mlp.TabularPreprocessor(min_count=10).fit(tr)
    assert set(pre.cat_cols) == {"card1", "ProductCD"}
    assert len(pre.flag_cols) == 1  # V1 and V2 share one missing pattern
    cat, dense = pre.transform(va)
    c = pre.cat_cols.index("card1")
    assert list(cat[:, c]) == [2, 1, 1, 0]  # frequent, rare (5 < 10), unseen, missing
    assert dense.shape == (4, pre.n_dense) and np.isfinite(dense).all()


def test_focal_loss_downweights_easy_examples():
    f = mlp.FocalLoss(gamma=2.0, alpha=0.5)
    y = torch.tensor([1.0])
    assert f(torch.tensor([4.0]), y) < f(torch.tensor([0.0]), y) / 10


@pytest.mark.parametrize("loss", ["bce_weighted", "focal"])
def test_end_to_end_learns_and_logs(tmp_path, loss):
    df = _synthetic(6000)
    df["card1"] = np.random.default_rng(0).integers(1000, 1100, len(df)).astype(float)
    data = tmp_path / "d.parquet"
    df.to_parquet(data)
    rec = mlp.run(data, tmp_path / "nosplit", tmp_path / "r.jsonl", tmp_path / "rep", loss=loss, epochs=4,
                  batch_size=256, hidden=(32, 16), patience=2, threads=1)
    assert rec["model"] == "mlp" and rec["metrics"]["valid"]["roc_auc"] > 0.6
    assert json.loads((tmp_path / "r.jsonl").read_text())["params"]["loss"] == loss
    assert (tmp_path / "rep" / f"mlp_{loss}_valid_scores.npy").exists()
