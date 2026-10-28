"""Smoke test for the Optuna sweep (Oct Task #8) on small synthetic data."""
import json

import numpy as np

from fraud.models import xgboost_tuning as tune
from tests.test_xgboost_baseline import _synthetic


def test_sweep_resumes_and_refits(tmp_path, monkeypatch):
    df = _synthetic(8000)
    df["card1"] = np.random.default_rng(0).integers(1000, 1100, len(df)).astype(float)
    data = tmp_path / "d.parquet"
    df.to_parquet(data)
    monkeypatch.setattr(tune, "MAX_ROUNDS", 60)
    monkeypatch.setattr(tune, "REPORT_EVERY", 10)
    args = ["--data", str(data), "--split-dir", str(tmp_path / "nosplit"), "--sample-rows", "3000",
            "--storage", f"sqlite:///{tmp_path / 's.db'}", "--registry", str(tmp_path / "r.jsonl"),
            "--out", str(tmp_path / "summary.json")]
    tune.main(args + ["--trials", "3", "--refit-seeds", ""])
    tune.main(args + ["--trials", "2", "--refit-seeds", "1,2"])  # resumes the same study
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["trials_total"] == 5
    assert s["best_trial_valid_auprc_on_sample"] > 0
    assert len(s["refit_seeds"]) == 2 and s["refit_valid"]["auprc"]["std"] is not None
    assert 1.0 <= s["best_params"]["scale_pos_weight"] <= s["base_spw"] + 1e-9
    assert (tmp_path / "r.jsonl").read_text().count("\n") == 2
    assert (tmp_path / "xgboost_optuna_task8_trials.csv").exists()
