"""
LightGBM baseline for the Banking-1A fraud detection project (Task #18).

This baseline uses LightGBM's native categorical handling rather than
one-hot encoding. The categorical feature definitions come from the Task #11
categorical analysis, including numeric-looking categorical codes such as
card1, card2, and addr1.

The model:
- uses the team's chronological train/validation split
- keeps the blind test split sealed by default
- handles class imbalance with scale_pos_weight
- early-stops on validation AUPRC
- evaluates using the shared/fallback Task #7 metric set
- logs the run to the results registry

Usage
-----
Canonical merged dataset:

    python -m fraud.models.lightgbm_baseline \
        --data data/train_merged_v1.parquet

Raw Kaggle files:

    python -m fraud.models.lightgbm_baseline \
        --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from fraud.categorical import CategoricalEncoder
from fraud.config import PARQUET_PATH, SPLITS_DIR
from fraud.models.xgboost_baseline import (
    AMOUNT_COL,
    ID_COL,
    SPLIT_DAYS,
    TARGET,
    TIME_COL,
    _git_commit,
    assign_split,
    evaluate,
    load_merged,
    log_run,
)


DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 256,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "force_col_wise": True,
}


def prepare_native_features(
    train: pd.DataFrame,
    others: list[pd.DataFrame],
):
    """
    Build LightGBM feature matrices using native categorical features.

    Task #11 defines which raw columns should be treated as categorical.
    Category levels are learned from the training split only. Categories
    appearing only in validation/test become missing.

    All remaining numeric predictors are stored as float32.
    """
    categorical_cols = [
        col
        for col in CategoricalEncoder().required_raw_cols
        if col in train.columns
    ]

    excluded = {
        TARGET,
        ID_COL,
        TIME_COL,
        "day",
        "split",
        *categorical_cols,
    }

    numeric_cols = [
        col
        for col in train.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(train[col])
    ]

    # Keep the same ordering used in the final Colab experiment:
    # numerical features first, then native categorical features.
    feature_cols = numeric_cols + categorical_cols

    X_train = train[feature_cols].copy()
    X_others = [
        frame[feature_cols].copy()
        for frame in others
    ]

    # Match the full-data experiment and reduce memory use.
    for col in numeric_cols:
        X_train[col] = X_train[col].astype("float32")

        for X in X_others:
            X[col] = X[col].astype("float32")

    # Learn categorical levels from training data only.
    for col in categorical_cols:
        train_values = train[col].astype("string")
        categories = pd.Index(
            train_values.dropna().unique()
        )

        X_train[col] = pd.Categorical(
            train_values,
            categories=categories,
        )

        for X, frame in zip(X_others, others):
            values = frame[col].astype("string")

            # Validation/test categories unseen during training become missing.
            values = values.where(
                values.isin(categories)
            )

            X[col] = pd.Categorical(
                values,
                categories=categories,
            )

    return (
        X_train,
        X_others,
        feature_cols,
        numeric_cols,
        categorical_cols,
    )


def train_lightgbm(
    df: pd.DataFrame,
    split: pd.Series,
    imbalance: str = "scale_pos_weight",
    params: dict | None = None,
    num_boost_round: int = 5000,
    early_stopping_rounds: int = 200,
    seed: int = 42,
    use_test: bool = False,
    feature_set: str = "native_raw_all_v1",
):
    """
    Train the Task #18 native-categorical LightGBM baseline.
    """
    params = {
        **DEFAULT_PARAMS,
        **(params or {}),
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
    }

    train = df.loc[split == "train"].copy()
    valid = df.loc[split == "valid"].copy()
    test = (
        df.loc[split == "test"].copy()
        if use_test
        else None
    )

    other_frames = [valid]

    if use_test:
        other_frames.append(test)

    (
        X_train,
        X_others,
        feature_cols,
        numeric_cols,
        categorical_cols,
    ) = prepare_native_features(
        train,
        other_frames,
    )

    X_valid = X_others[0]

    y_train = train[TARGET].to_numpy()
    y_valid = valid[TARGET].to_numpy()

    if imbalance == "scale_pos_weight":
        positives = y_train.sum()
        negatives = len(y_train) - positives

        params["scale_pos_weight"] = float(
            negatives / max(positives, 1)
        )

    elif imbalance != "none":
        raise ValueError(
            "imbalance must be 'scale_pos_weight' or 'none'"
        )

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

    start = time.time()

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(
                early_stopping_rounds
            ),
            lgb.log_evaluation(100),
        ],
    )

    train_seconds = time.time() - start

    valid_scores = booster.predict(
        X_valid,
        num_iteration=booster.best_iteration,
    )

    valid_amount = (
        valid[AMOUNT_COL].to_numpy()
        if AMOUNT_COL in valid.columns
        else None
    )

    results = {
        "valid": evaluate(
            y_valid,
            valid_scores,
            amount=valid_amount,
        )
    }

    if use_test:
        X_test = X_others[1]

        test_scores = booster.predict(
            X_test,
            num_iteration=booster.best_iteration,
        )

        test_amount = (
            test[AMOUNT_COL].to_numpy()
            if AMOUNT_COL in test.columns
            else None
        )

        results["test"] = evaluate(
            test[TARGET].to_numpy(),
            test_scores,
            amount=test_amount,
        )

    record = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds"),
        "model": "lightgbm",
        "task": "#18 baseline",
        "feature_set": feature_set,
        "n_features": len(feature_cols),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "imbalance": imbalance,
        "params": params,
        "best_iteration": int(
            booster.best_iteration
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
        "train_seconds": round(
            train_seconds,
            1,
        ),
        "metrics": results,
    }

    return booster, record, valid_scores


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Native categorical LightGBM baseline "
            "(Task #18)"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        help="Canonical merged Parquet/CSV",
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Folder containing raw Kaggle CSVs",
    )

    parser.add_argument(
        "--split-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Folder containing frozen split ID files",
    )

    parser.add_argument(
        "--imbalance",
        choices=[
            "scale_pos_weight",
            "none",
        ],
        default="scale_pos_weight",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--use-test",
        action="store_true",
        help=(
            "Score the sealed blind test split. "
            "Do not use during baseline development."
        ),
    )

    parser.add_argument(
        "--save-model",
        type=Path,
        help="Optional output path for the trained model",
    )

    parser.add_argument(
        "--registry",
        type=Path,
        help="Override results registry path",
    )

    args = parser.parse_args(argv)

    # Prefer the canonical Parquet when no explicit source is supplied.
    data_path = args.data

    if (
        data_path is None
        and args.raw_dir is None
        and PARQUET_PATH.exists()
    ):
        data_path = PARQUET_PATH

    df = load_merged(
        data_path=data_path,
        raw_dir=args.raw_dir,
    )

    split = assign_split(
        df,
        args.split_dir,
    )

    print(
        f"Split source: "
        f"{split.attrs.get('source')}"
    )
    print(split.value_counts().to_string())

    booster, record, _ = train_lightgbm(
        df,
        split,
        imbalance=args.imbalance,
        num_boost_round=args.rounds,
        early_stopping_rounds=(
            args.early_stopping
        ),
        seed=args.seed,
        use_test=args.use_test,
    )

    if args.save_model:
        args.save_model.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        booster.save_model(
            str(args.save_model)
        )

    registry_path = log_run(
        record,
        args.registry,
    )

    print(
        json.dumps(
            record["metrics"],
            indent=2,
        )
    )

    print(
        f"Logged run to {registry_path}"
    )


if __name__ == "__main__":
    main()