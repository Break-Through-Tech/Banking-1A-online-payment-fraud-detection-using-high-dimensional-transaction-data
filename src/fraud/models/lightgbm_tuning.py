"""
LightGBM hyperparameter tuning for Banking-1A fraud detection (October Task #10).

Workflow
--------
1. Load the same merged dataset and chronological split used by the baseline.
2. Use the same native categorical feature representation as Task #4.
3. Create a stratified 150,000-row sample from the training split.
4. Run a broad Optuna search on the sample.
5. Run a refined Optuna search around the promising parameter region.
6. Retrain the best refined configuration on the full training split.
7. Evaluate on the unchanged validation split.
8. Log the final tuned run.

The blind test split remains sealed by default.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud.models.lightgbm_baseline import (
    AMOUNT_COL,
    SPLIT_DAYS,
    TARGET,
    _git_commit,
    assign_split,
    evaluate,
    load_merged,
    log_run,
    prepare_native_features,
)


BASELINE_AUPRC = 0.6620552457417546

BASELINE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 256,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "lambda_l1": 0.0001,
    "lambda_l2": 1.0,
}


def create_tuning_sample(
    X_train: pd.DataFrame,
    y_train,
    sample_size: int = 150_000,
    seed: int = 42,
):
    """Create the stratified sample used for faster Optuna search."""
    if sample_size >= len(X_train):
        return X_train.copy(), y_train.copy()

    X_tune, _, y_tune, _ = train_test_split(
        X_train,
        y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=seed,
    )

    return X_tune, y_tune


def build_objective(
    X_tune,
    y_tune,
    X_valid,
    y_valid,
    categorical_cols,
    search_type: str,
    seed: int = 42,
):
    """Return an Optuna objective for broad or refined LightGBM tuning."""

    positives = y_tune.sum()
    negatives = len(y_tune) - positives
    scale_pos_weight = float(
        negatives / max(positives, 1)
    )

    def objective(trial):
        if search_type == "broad":
            tuning_params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.02,
                    0.10,
                    log=True,
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    64,
                    384,
                    step=64,
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    50,
                    250,
                    step=25,
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction",
                    0.4,
                    0.9,
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction",
                    0.6,
                    1.0,
                ),
                "lambda_l1": trial.suggest_float(
                    "lambda_l1",
                    1e-4,
                    10.0,
                    log=True,
                ),
                "lambda_l2": trial.suggest_float(
                    "lambda_l2",
                    1e-4,
                    10.0,
                    log=True,
                ),
            }

            max_rounds = 1500
            early_stopping = 75

        elif search_type == "refined":
            tuning_params = {
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    0.02,
                    0.05,
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    192,
                    384,
                    step=32,
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    50,
                    125,
                    step=25,
                ),
                "feature_fraction": trial.suggest_float(
                    "feature_fraction",
                    0.60,
                    0.80,
                ),
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction",
                    0.70,
                    0.90,
                ),
                "lambda_l1": trial.suggest_float(
                    "lambda_l1",
                    1e-4,
                    0.1,
                    log=True,
                ),
                "lambda_l2": trial.suggest_float(
                    "lambda_l2",
                    0.3,
                    3.0,
                    log=True,
                ),
            }

            max_rounds = 2500
            early_stopping = 100

        else:
            raise ValueError(
                "search_type must be 'broad' or 'refined'"
            )

        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "force_col_wise": True,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "scale_pos_weight": scale_pos_weight,
            "bagging_freq": 1,
            **tuning_params,
        }

        # Build matching train/validation datasets inside each trial.
        # This avoids LightGBM bin-mapper conflicts between trials.
        train_set = lgb.Dataset(
            X_tune,
            label=y_tune,
            categorical_feature=categorical_cols,
            free_raw_data=False,
        )

        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            categorical_feature=categorical_cols,
            reference=train_set,
            free_raw_data=False,
        )

        model = lgb.train(
            params,
            train_set,
            num_boost_round=max_rounds,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(
                    early_stopping,
                    verbose=False,
                )
            ],
        )

        trial.set_user_attr(
            "best_iteration",
            int(model.best_iteration),
        )

        return float(
            model.best_score["valid"]["average_precision"]
        )

    return objective


def create_study(
    name: str,
    storage: str | None,
    seed: int = 42,
):
    """Create or resume an Optuna study."""
    return optuna.create_study(
        study_name=name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=seed
        ),
    )


def train_final_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    categorical_cols,
    best_params: dict,
    seed: int = 42,
):
    """Retrain the best Optuna configuration on the full training split."""

    positives = y_train.sum()
    negatives = len(y_train) - positives

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "force_col_wise": True,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "scale_pos_weight": float(
            negatives / max(positives, 1)
        ),
        "bagging_freq": 1,
        **best_params,
    }

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        categorical_feature=categorical_cols,
        reference=train_set,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        train_set,
        num_boost_round=6000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(100),
        ],
    )

    scores = model.predict(
        X_valid,
        num_iteration=model.best_iteration,
    )

    return model, scores, params


def run_tuning(
    df: pd.DataFrame,
    split: pd.Series,
    sample_size: int = 150_000,
    broad_trials: int = 5,
    refined_trials: int = 10,
    storage_dir: Path | None = None,
    seed: int = 42,
):
    """Run broad search, refined search, and final full-data training."""

    train = df.loc[split == "train"].copy()
    valid = df.loc[split == "valid"].copy()

    (
        X_train,
        others,
        feature_cols,
        numeric_cols,
        categorical_cols,
    ) = prepare_native_features(
        train,
        [valid],
    )

    X_valid = others[0]

    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()

    X_tune, y_tune = create_tuning_sample(
        X_train,
        y_train,
        sample_size=sample_size,
        seed=seed,
    )

    storage_dir = Path(
        storage_dir or "results"
    )
    storage_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    broad_storage = (
        f"sqlite:///{storage_dir / 'optuna_lightgbm_broad.db'}"
    )

    refined_storage = (
        f"sqlite:///{storage_dir / 'optuna_lightgbm_refined.db'}"
    )

    broad_study = create_study(
        "lightgbm_task10_broad",
        broad_storage,
        seed,
    )

    broad_objective = build_objective(
        X_tune,
        y_tune,
        X_valid,
        y_valid,
        categorical_cols,
        search_type="broad",
        seed=seed,
    )

    broad_study.optimize(
        broad_objective,
        n_trials=broad_trials,
        n_jobs=1,
        gc_after_trial=True,
    )

    # Evaluate the Task #4 baseline settings on the same
    # 150k tuning sample for a fair search-stage comparison.
    broad_study.enqueue_trial(
        BASELINE_PARAMS
    )

    broad_study.optimize(
        broad_objective,
        n_trials=1,
        n_jobs=1,
        gc_after_trial=True,
    )

    refined_study = create_study(
        "lightgbm_task10_refined",
        refined_storage,
        seed,
    )

    refined_objective = build_objective(
        X_tune,
        y_tune,
        X_valid,
        y_valid,
        categorical_cols,
        search_type="refined",
        seed=seed,
    )

    refined_study.optimize(
        refined_objective,
        n_trials=refined_trials,
        n_jobs=1,
        gc_after_trial=True,
    )

    model, valid_scores, final_params = (
        train_final_model(
            X_train,
            y_train,
            X_valid,
            y_valid,
            categorical_cols,
            refined_study.best_params,
            seed,
        )
    )

    amount = (
        valid[AMOUNT_COL].to_numpy()
        if AMOUNT_COL in valid.columns
        else None
    )

    metrics = evaluate(
        y_valid,
        valid_scores,
        amount=amount,
    )

    record = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "model": "lightgbm",
        "task": "October Task #10 tuning",
        "feature_set": "native_raw_all_v1",
        "search_method": "optuna_tpe",
        "sample_size": len(X_tune),
        "broad_trials": broad_trials,
        "refined_trials": refined_trials,
        "n_features": len(feature_cols),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "baseline_auprc": BASELINE_AUPRC,
        "best_sampled_auprc": float(
            refined_study.best_value
        ),
        "best_params": refined_study.best_params,
        "final_params": final_params,
        "best_iteration": int(
            model.best_iteration
        ),
        "split_id": split.attrs.get(
            "source",
            "unknown",
        ),
        "split_sizes": {
            name: int((split == name).sum())
            for name in SPLIT_DAYS
        },
        "seed": seed,
        "git_commit": _git_commit(),
        "metrics": {
            "valid": metrics
        },
    }

    return (
        model,
        record,
        valid_scores,
        broad_study,
        refined_study,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Optuna LightGBM tuning "
            "(October Task #10)"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        help="Merged Parquet/CSV dataset",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Folder containing raw Kaggle CSVs",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=150_000,
    )

    parser.add_argument(
        "--broad-trials",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--refined-trials",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("results/optuna"),
    )

    parser.add_argument(
        "--save-model",
        type=Path,
    )

    parser.add_argument(
        "--registry",
        type=Path,
    )

    args = parser.parse_args(argv)

    df = load_merged(
        data_path=args.data,
        raw_dir=args.raw_dir,
    )

    split = assign_split(df)

    print(
        "Split source:",
        split.attrs.get("source"),
    )
    print(split.value_counts().to_string())

    (
        model,
        record,
        _,
        broad_study,
        refined_study,
    ) = run_tuning(
        df,
        split,
        sample_size=args.sample_size,
        broad_trials=args.broad_trials,
        refined_trials=args.refined_trials,
        storage_dir=args.study_dir,
        seed=args.seed,
    )

    if args.save_model:
        args.save_model.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save_model(
            str(args.save_model),
            num_iteration=model.best_iteration,
        )

    registry_path = log_run(
        record,
        args.registry,
    )

    print(
        "Broad best sampled AUPRC:",
        broad_study.best_value,
    )

    print(
        "Refined best sampled AUPRC:",
        refined_study.best_value,
    )

    print(
        "Final validation metrics:"
    )
    print(
        json.dumps(
            record["metrics"]["valid"],
            indent=2,
        )
    )

    print(
        f"Logged run to {registry_path}"
    )


if __name__ == "__main__":
    main()