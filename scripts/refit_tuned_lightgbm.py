"""Refit Orator's tuned LightGBM (Oct Task #10) with one seed to get validation
scores for the ensemble (#53). Reuses his feature preparation and final-training
functions unchanged; the tuned settings come from a JSON file given on the
command line, because results/lightgbm_optuna_task10.json stores only metrics.

    python scripts/refit_tuned_lightgbm.py reports/lgbm_best_params.json
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.models.lightgbm_baseline import prepare_native_features
from fraud.models.lightgbm_tuning import train_final_model
from fraud.split import load_split

params = json.loads(Path(sys.argv[1]).read_text())
df = pd.read_parquet("data/train_merged_v1.parquet")
train = df[df["TransactionID"].isin(load_split("train"))]
valid = df[df["TransactionID"].isin(load_split("valid"))].sort_values("TransactionID")
del df

X_train, (X_valid,), _, _, categorical_cols = prepare_native_features(train, [valid])
print(f"Features: {X_train.shape[1]} ({len(categorical_cols)} categorical)")

model, scores, used = train_final_model(
    X_train, train["isFraud"].to_numpy(), X_valid, valid["isFraud"].to_numpy(),
    categorical_cols, params, seed=42,
)
print(f"Validation AUPRC: {average_precision_score(valid['isFraud'], scores):.4f} (Orator's refined run: 0.6683)")

out = Path("reports/lightgbm_tuned")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"TransactionID": valid["TransactionID"].to_numpy(), "score": scores}).to_parquet(
    out / "valid_scores.parquet", index=False)
print("Saved", out / "valid_scores.parquet")