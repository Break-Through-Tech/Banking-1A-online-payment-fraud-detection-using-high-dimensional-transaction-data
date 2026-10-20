"""
XGBoost hyperparameter tuning with Optuna (October Task #8).

Starts from the September Task #19 baseline (feature set fe_v2) and searches
tree, regularization, sampling, imbalance and categorical-split settings.

How it works
------------
1. Build fe_v2 features once. Frequency counts come from the full train split.
2. Tune on a random sample of training rows (default 150,000, the same budget
   Orator used for LightGBM in Task #10) and score every trial on the full
   frozen validation split (85,044 rows). Each trial early-stops on validation
   AUPRC; Optuna's median pruner stops weak trials early.
3. Refit the best configuration on all 420,066 training rows with several
   seeds and log each refit to the results registry.

The study is stored in SQLite, so an interrupted sweep resumes where it
stopped (rerun the same command). The blind test split is never loaded.

Caveat: choosing hyperparameters on validation makes the validation score of
the winner slightly optimistic. The November blind-test run is the unbiased
check.

Usage
-----
    python -m fraud.models.xgboost_tuning --trials 60
    python -m fraud.models.xgboost_tuning --trials 0 --refit-seeds 42,43,44   # refit only
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from fraud.config import DATA_DIR, REPO_ROOT
from fraud.models import xgboost_baseline as xb

STUDY_NAME = "xgboost_task8_fe_v2"
TUNING_LR = 0.1          # fixed during the search to fit a CPU budget
MAX_ROUNDS = 2000
EARLY_STOP = 50
REPORT_EVERY = 25        # rounds between pruning checks


def build_matrices(data_path=None, split_dir=None, sample_rows=150_000, seed=42):
    """Return (dtrain_sample, dvalid, info) built from fe_v2 features."""
    df = xb.load_merged(data_path)
    split = xb.assign_split(df, split_dir)
    train = df[split == "train"]
    valid = df[split == "valid"]
    info = {"split_id": split.attrs["source"], "split_sizes": {k: int((split == k).sum()) for k in xb.SPLIT_DAYS}}
    del df
    gc.collect()

    train, [valid] = xb.add_features_v2(train, [valid])
    drop = [xb.TARGET, xb.ID_COL, xb.TIME_COL]
    if sample_rows and sample_rows < len(train):
        train = train.sample(n=sample_rows, random_state=seed)
    X_tr, [X_va], cols, cats = xb.prepare_features(train, [valid], drop)
    y_tr, y_va = train[xb.TARGET].values, valid[xb.TARGET].values
    info.update(n_features=len(cols), n_categorical=len(cats), tuning_rows=int(len(train)),
                tuning_fraud_rate=float(y_tr.mean()), base_spw=float((len(y_tr) - y_tr.sum()) / y_tr.sum()))
    dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    del X_tr, X_va, train, valid
    gc.collect()
    return dtr, dva, info


def suggest_params(trial: optuna.Trial, base_spw: float) -> dict:
    spw_power = trial.suggest_float("spw_power", 0.0, 1.0)  # 0 = no weighting, 1 = full ratio
    return {
        **{k: v for k, v in xb.DEFAULT_PARAMS.items() if k in ("objective", "eval_metric", "tree_method", "max_bin",
                                                                    "max_cached_hist_node")},
        "learning_rate": TUNING_LR,
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 100, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "max_cat_threshold": trial.suggest_int("max_cat_threshold", 8, 128, log=True),
        "max_cat_to_onehot": 1,
        "scale_pos_weight": float(base_spw ** spw_power),
    }


class PruneCallback(xgb.callback.TrainingCallback):
    """Report validation AUPRC to Optuna and stop the trial if it is pruned."""

    def __init__(self, trial):
        self.trial = trial

    def after_iteration(self, model, epoch, evals_log):
        if epoch % REPORT_EVERY == 0:
            score = evals_log["valid"]["aucpr"][-1]
            self.trial.report(score, epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"pruned at round {epoch}")
        return False


def objective_factory(dtr, dva, base_spw):
    def objective(trial):
        params = {**suggest_params(trial, base_spw), "seed": 42, "nthread": -1}
        start = time.time()
        booster = xgb.train(params, dtr, num_boost_round=MAX_ROUNDS, evals=[(dva, "valid")],
                            early_stopping_rounds=EARLY_STOP, verbose_eval=False,
                            callbacks=[PruneCallback(trial)])
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        trial.set_user_attr("seconds", round(time.time() - start, 1))
        trial.set_user_attr("scale_pos_weight", params["scale_pos_weight"])
        return float(booster.best_score)
    return objective


def best_params(study: optuna.Study, base_spw: float) -> dict:
    p = dict(study.best_params)
    spw = float(base_spw ** p.pop("spw_power"))
    return {**p, "scale_pos_weight": spw, "max_cat_to_onehot": 1}


def refit(params: dict, seeds, data_path=None, split_dir=None, registry=None, learning_rate=TUNING_LR,
          report_dir=None, notes=""):
    """Train the tuned configuration on the full train split, once per seed."""
    df = xb.load_merged(data_path)
    split = xb.assign_split(df, split_dir)
    records = []
    for seed in seeds:
        tuned = {**params, "learning_rate": learning_rate}
        # train_xgboost sets scale_pos_weight itself when imbalance="scale_pos_weight";
        # we pass the tuned value explicitly instead.
        booster, rec, score = xb.train_xgboost(
            df, split, imbalance="none", params=tuned, num_boost_round=5000, early_stopping_rounds=100,
            seed=seed, feature_set="fe_v2")
        rec["task"] = "Oct #8 tuned"
        rec["notes"] = notes
        rec["params"]["scale_pos_weight"] = tuned["scale_pos_weight"]
        rec["imbalance"] = f"scale_pos_weight={tuned['scale_pos_weight']:.2f} (tuned)"
        xb.log_run(rec, registry)
        records.append(rec)
        print(f"seed {seed}: AUPRC {rec['metrics']['valid']['auprc']:.4f} best round {rec['best_iteration'] + 1}")
        if report_dir and seed == seeds[0]:
            xb.write_report(booster, df[split == "valid"], score, report_dir)
            booster.save_model(Path(report_dir) / "xgboost_tuned.json")
            np.save(Path(report_dir) / "valid_scores.npy", score)
        del booster
        gc.collect()
    return records


def summarize(study, info, refits, baseline_auprc=None) -> dict:
    trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    done = trials[trials["state"] == "COMPLETE"]
    v = [r["metrics"]["valid"] for r in refits]
    mean = lambda k: float(np.mean([m[k] for m in v])) if v else None
    std = lambda k: float(np.std([m[k] for m in v], ddof=1)) if len(v) > 1 else None
    return {
        "model": "xgboost",
        "task": "October Task #8 hyperparameter tuning",
        "search_method": "Optuna TPE + median pruner",
        "feature_set": "fe_v2",
        **info,
        "trials_total": int(len(trials)),
        "trials_complete": int(len(done)),
        "trials_pruned": int((trials["state"] == "PRUNED").sum()),
        "tuning_learning_rate": TUNING_LR,
        "best_trial": int(study.best_trial.number),
        "best_trial_valid_auprc_on_sample": float(study.best_value),
        "best_params": best_params(study, info["base_spw"]),
        "baseline_auprc_full_train": baseline_auprc,
        "refit_seeds": [r["seed"] for r in refits],
        "refit_valid": {k: {"mean": mean(k), "std": std(k)} for k in
                        ("auprc", "roc_auc", "recall_at_1pct_review", "recall_at_5pct_review", "ks")} if v else None,
        "refit_best_iterations": [int(r["best_iteration"]) for r in refits],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _refit_command(argv: list[str]) -> list[str]:
    """Same arguments, but --trials 0 (refit only)."""
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a == "--trials":
            skip = True
            continue
        if a.startswith("--trials="):
            continue
        out.append(a)
    return [sys.executable, "-m", "fraud.models.xgboost_tuning", "--trials", "0", *out]


def main(argv=None):
    p = argparse.ArgumentParser(description="Optuna sweep for XGBoost (Oct Task #8)")
    p.add_argument("--data", type=Path)
    p.add_argument("--split-dir", type=Path)
    p.add_argument("--trials", type=int, default=60, help="New trials to run this call (0 = refit only)")
    p.add_argument("--sample-rows", type=int, default=150_000)
    p.add_argument("--storage", default=f"sqlite:///{(DATA_DIR / 'optuna_xgboost.db').resolve()}")
    p.add_argument("--refit-seeds", default="42,43,44")
    p.add_argument("--refit-lr", type=float, default=TUNING_LR)
    p.add_argument("--registry", type=Path)
    p.add_argument("--report-dir", type=Path)
    p.add_argument("--baseline-auprc", type=float, default=0.6637, help="Sept #19 fe_v2 full-train AUPRC")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "xgboost_optuna_task8.json")
    p.add_argument("--in-process-refit", action="store_true", help="Refit in this process (small data / tests)")
    args = p.parse_args(argv)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=args.storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=12, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=150, interval_steps=1))

    if args.trials or study.user_attrs.get("info") is None:
        dtr, dva, info = build_matrices(args.data, args.split_dir, args.sample_rows)
        print(json.dumps(info, indent=2))
        study.set_user_attr("info", info)
    else:
        dtr = dva = None
        info = study.user_attrs["info"]  # refit only: skip rebuilding the tuning matrices
    if not study.trials:
        # Trial 0 = the September fe_v2 baseline settings, so tuning must beat it.
        study.enqueue_trial({"max_depth": 8, "min_child_weight": 1.0, "gamma": 1e-3, "subsample": 0.8,
                             "colsample_bytree": 0.5, "colsample_bynode": 1.0, "reg_lambda": 1.0,
                             "reg_alpha": 1e-3, "max_cat_threshold": 32, "spw_power": 1.0})

    def log_trial(study, trial):
        print(f"trial {trial.number:3d} {trial.state.name:8s} value={trial.value} "
              f"best={study.best_value:.4f} ({trial.user_attrs.get('seconds', '-')}s)", flush=True)

    if args.trials:
        study.optimize(objective_factory(dtr, dva, info["base_spw"]), n_trials=args.trials, callbacks=[log_trial],
                       gc_after_trial=True)
    del dtr, dva
    gc.collect()

    params = best_params(study, info["base_spw"])
    print("best params:", json.dumps(params, indent=2))
    seeds = [int(s) for s in args.refit_seeds.split(",") if s]
    if args.trials and seeds and not args.in_process_refit:
        # Refit in a fresh process: after a long sweep the heap still holds
        # the tuning matrices, and the full-data refit was killed for running
        # out of memory on an 8 GB machine when run in the same process.
        cmd = _refit_command(argv if argv is not None else sys.argv[1:])
        print("refitting in a new process:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return
    refits = refit(params, seeds, args.data, args.split_dir, args.registry, args.refit_lr, args.report_dir,
                   notes=f"Oct #8 Optuna best trial {study.best_trial.number}; lr {args.refit_lr}") if seeds else []
    summary = summarize(study, info, refits, args.baseline_auprc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    trials_csv = args.out.with_name("xgboost_optuna_task8_trials.csv")
    study.trials_dataframe(attrs=("number", "value", "state", "duration", "params", "user_attrs")).to_csv(
        trials_csv, index=False)
    print(f"wrote {args.out} and {trials_csv}")


if __name__ == "__main__":
    main()
