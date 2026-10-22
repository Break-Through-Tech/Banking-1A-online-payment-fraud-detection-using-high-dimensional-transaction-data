"""End-to-end smoke test of scripts/analyze_tuned_xgboost.py on synthetic data."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from fraud.models import xgboost_baseline as xb
from tests.test_xgboost_baseline import _synthetic

ROOT = Path(__file__).resolve().parents[1]


def test_analysis_runs(tmp_path):
    df = _synthetic(8000)
    df["card1"] = np.random.default_rng(0).integers(1000, 1100, len(df)).astype(float)
    df["has_identity"] = (np.random.default_rng(1).random(len(df)) < 0.3).astype("int8")
    data = tmp_path / "d.parquet"
    df.to_parquet(data)
    split = xb.assign_split(df, tmp_path / "nosplit")
    booster, rec, score = xb.train_xgboost(df, split, num_boost_round=40, early_stopping_rounds=10,
                                           feature_set="fe_v2")
    mdir = tmp_path / "model"
    mdir.mkdir()
    booster.save_model(mdir / "xgboost_tuned.json")
    np.save(mdir / "valid_scores.npy", score)

    spec = importlib.util.spec_from_file_location("analyze", ROOT / "scripts" / "analyze_tuned_xgboost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    mod.main(["--data", str(data), "--split-dir", str(tmp_path / "nosplit"), "--model-dir", str(mdir),
              "--out", str(out), "--legit-sample", "500"])
    for f in ("operating_points.csv", "shap_global_importance.csv", "reasons_flagged_by_family.csv",
              "dollar_capture_by_strategy.png", "missed_vs_caught.csv", "calibration.png"):
        assert (out / f).exists(), f
    assert json.loads((out / "summary.json").read_text())["scale_pos_weight"] > 1
