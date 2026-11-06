"""Refit Mohit's tuned XGBoost (Oct Task #8) with one seed to get validation
scores for the ensemble (#53). Reads the tuned settings from his committed
results file, so no Optuna database is needed. Logs to a scratch registry
inside reports/ so the shared registry is not duplicated."""

import json
from pathlib import Path

from fraud.models import xgboost_tuning as xt

params = json.loads(Path("results/xgboost_optuna_task8.json").read_text())["best_params"]
out = Path("reports/xgboost_tuned")
out.mkdir(parents=True, exist_ok=True)
xt.refit(params, [42], registry=out / "registry_rerun.jsonl", report_dir=out,
         notes="one-seed rerun for ensemble scores (Leo, #53)")
print("Saved", out / "valid_scores.npy")