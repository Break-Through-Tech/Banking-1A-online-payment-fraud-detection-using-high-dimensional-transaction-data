"""
LightGBM baseline (Task #18).

Native categorical handling (levels learned on train), scale_pos_weight for
imbalance, early stopping on validation average precision.

    python -m src.models.lightgbm_baseline
    python -m src.models.lightgbm_baseline --imbalance none
"""

from __future__ import annotations

import time

import lightgbm as lgb

from src.models import common

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 256,
    "max_depth": -1,
    "min_child_samples": 100,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "cat_smooth": 10,
    "max_cat_to_onehot": 4,
    "verbosity": -1,
}


def run(train, valid, split_id, seed=42, imbalance="scale_pos_weight", rounds=5000, early_stopping=200,
        feature_set="raw_all_v1", registry_path=None, report_dir=None, notes="", params=None):
    cols = common.feature_columns(train)
    cat_cols = common.categorical_columns(train, cols)
    Xtr, [Xva] = common.categorize(train[cols], [valid[cols]], cat_cols)
    ytr, yva = train[common.TARGET].values, valid[common.TARGET].values

    params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
    if imbalance == "scale_pos_weight":
        params["scale_pos_weight"] = common.pos_weight(ytr)

    dtrain = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols, free_raw_data=False)
    dvalid = lgb.Dataset(Xva, yva, reference=dtrain, categorical_feature=cat_cols)
    start = time.time()
    booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvalid], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(200)])
    scores = booster.predict(Xva, num_iteration=booster.best_iteration)
    return common.finish_run("lightgbm", valid, scores, params=params, feature_set=feature_set, seed=seed,
                             split_id=split_id, n_features=len(cols),
                             extra={"imbalance": imbalance, "best_iteration": booster.best_iteration,
                                    "train_seconds": round(time.time() - start, 1)},
                             registry_path=registry_path, report_dir=report_dir, notes=notes)


def main(argv=None):
    p = common.base_argparser("LightGBM baseline (Task #18)")
    p.add_argument("--imbalance", choices=["scale_pos_weight", "none"], default="scale_pos_weight")
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--early-stopping", type=int, default=200)
    args = p.parse_args(argv)
    train, valid, sid = common.load_train_valid(args.data, args.split_dir, args.sample, args.seed)
    run(train, valid, sid, seed=args.seed, imbalance=args.imbalance, rounds=args.rounds,
        early_stopping=args.early_stopping, feature_set=args.feature_set, registry_path=args.registry,
        report_dir=args.report_dir, notes=args.notes)


if __name__ == "__main__":
    main()
