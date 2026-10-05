"""
Logistic regression and random forest baselines (Task #17).

These set the performance floor every later model must clear.

    python -m src.models.linear_rf_baseline --model logreg
    python -m src.models.linear_rf_baseline --model rf --sample 0.5
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.models import common


def build_logreg(num_cols, cat_cols, seed: int, C: float = 0.1) -> Pipeline:
    """Median impute + missing flags + scaling for numerics; one-hot for
    categoricals with rare levels pooled (min 50 rows, fitted on train)."""
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                     sparse_output=True)),
        ]), cat_cols),
    ], sparse_threshold=0.3)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_rf(num_cols, cat_cols, seed: int, n_estimators: int = 300) -> Pipeline:
    """Random forest. sklearn trees accept NaN in numerics; categoricals are
    ordinal codes (unseen = -1, missing = -2)."""
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-2), cat_cols),
    ])
    clf = RandomForestClassifier(
        n_estimators=n_estimators, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def run(model: str, train, valid, split_id, seed=42, feature_set="raw_all_v1", registry_path=None,
        report_dir=None, notes="", **kw):
    cols = common.feature_columns(train)
    cat_cols = common.categorical_columns(train, cols)
    num_cols = [c for c in cols if c not in cat_cols]
    Xtr, Xva = train[cols].copy(), valid[cols].copy()
    for c in cat_cols:  # plain strings for sklearn encoders
        Xtr[c] = Xtr[c].astype("object").where(Xtr[c].notna(), np.nan)
        Xva[c] = Xva[c].astype("object").where(Xva[c].notna(), np.nan)

    pipe = build_logreg(num_cols, cat_cols, seed, **kw) if model == "logreg" else build_rf(num_cols, cat_cols, seed, **kw)
    start = time.time()
    pipe.fit(Xtr, train[common.TARGET].values)
    scores = pipe.predict_proba(Xva)[:, 1]
    params = {k: v for k, v in pipe.named_steps["clf"].get_params().items()
              if k in ("C", "class_weight", "max_iter", "n_estimators", "min_samples_leaf", "max_features")}
    return common.finish_run(model, valid, scores, params=params, feature_set=feature_set, seed=seed,
                             split_id=split_id, n_features=len(cols),
                             extra={"train_seconds": round(time.time() - start, 1)},
                             registry_path=registry_path, report_dir=report_dir, notes=notes)


def main(argv=None):
    p = common.base_argparser("Logistic regression / random forest baseline (Task #17)")
    p.add_argument("--model", choices=["logreg", "rf"], required=True)
    args = p.parse_args(argv)
    train, valid, sid = common.load_train_valid(args.data, args.split_dir, args.sample, args.seed)
    run(args.model, train, valid, sid, seed=args.seed, feature_set=args.feature_set,
        registry_path=args.registry, report_dir=args.report_dir, notes=args.notes)


if __name__ == "__main__":
    main()
