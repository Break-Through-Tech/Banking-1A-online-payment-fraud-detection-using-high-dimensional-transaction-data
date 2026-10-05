"""
CatBoost baseline (Task #20).

Uses CatBoost's native categorical handling (ordered target statistics), so
no manual encoding. Imbalance via auto_class_weights, early stopping on
validation PR-AUC.

    python -m src.models.catboost_baseline
    python -m src.models.catboost_baseline --gpu      # Colab
"""

from __future__ import annotations

import time

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.models import common

# Numeric code columns that CatBoost should treat as categories.
CODE_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]

DEFAULT_PARAMS = {
    "iterations": 5000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "eval_metric": "PRAUC",
    "od_type": "Iter",
    "od_wait": 200,
    "one_hot_max_size": 4,
    "verbose": 200,
}


def _as_strings(X, cols):
    X = X.copy()
    for c in cols:
        v = X[c]
        if pd.api.types.is_float_dtype(v):
            v = v.astype("Int64")  # 13926.0 -> "13926", not "13926.0"
        X[c] = v.astype("string").fillna("missing").astype(str)
    return X


def run(train, valid, split_id, seed=42, imbalance="Balanced", gpu=False, feature_set="raw_all_v1",
        registry_path=None, report_dir=None, notes="", params=None):
    cols = common.feature_columns(train)
    cat_cols = common.categorical_columns(train, cols) + [c for c in CODE_COLS if c in cols]
    Xtr, Xva = _as_strings(train[cols], cat_cols), _as_strings(valid[cols], cat_cols)
    ytr, yva = train[common.TARGET].values, valid[common.TARGET].values

    params = {**DEFAULT_PARAMS, **(params or {}), "random_seed": seed}
    if imbalance != "none":
        params["auto_class_weights"] = imbalance
    if gpu:
        params["task_type"] = "GPU"

    model = CatBoostClassifier(**params)
    start = time.time()
    model.fit(Pool(Xtr, ytr, cat_features=cat_cols), eval_set=Pool(Xva, yva, cat_features=cat_cols),
              use_best_model=True)
    scores = model.predict_proba(Xva)[:, 1]
    logged = {k: v for k, v in params.items() if k != "verbose"}
    return common.finish_run("catboost", valid, scores, params=logged, feature_set=feature_set, seed=seed,
                             split_id=split_id, n_features=len(cols),
                             extra={"imbalance": imbalance, "best_iteration": int(model.get_best_iteration()),
                                    "n_cat_features": len(cat_cols),
                                    "train_seconds": round(time.time() - start, 1)},
                             registry_path=registry_path, report_dir=report_dir, notes=notes)


def main(argv=None):
    p = common.base_argparser("CatBoost baseline (Task #20)")
    p.add_argument("--imbalance", choices=["Balanced", "SqrtBalanced", "none"], default="Balanced")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--iterations", type=int, default=5000)
    args = p.parse_args(argv)
    train, valid, sid = common.load_train_valid(args.data, args.split_dir, args.sample, args.seed)
    run(train, valid, sid, seed=args.seed, imbalance=args.imbalance, gpu=args.gpu,
        feature_set=args.feature_set, registry_path=args.registry, report_dir=args.report_dir,
        notes=args.notes, params={"iterations": args.iterations})


if __name__ == "__main__":
    main()
