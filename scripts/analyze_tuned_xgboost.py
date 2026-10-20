"""
SHAP attribution (Oct #18) and operating-point selection (Oct #19) for the
tuned XGBoost model produced by `python -m fraud.models.xgboost_tuning
--report-dir reports/xgboost_tuned`.

    python scripts/analyze_tuned_xgboost.py --model-dir reports/xgboost_tuned --out reports/oct_analysis

Writes CSV tables and PNG charts to --out. Uses the validation split only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from fraud import explain, thresholds  # noqa: E402
from fraud.models import xgboost_baseline as xb  # noqa: E402

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.titlesize": 10, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2})


def hbar(ax, labels, values, fmt, color=BLUE):
    y = range(len(labels))
    ax.barh(list(y), values, color=color, height=0.62)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(values):
        ax.text(v, i, " " + fmt(v), va="center", fontsize=8, color=INK)
    ax.set_xlim(0, max(values) * 1.2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--split-dir", type=Path)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--summary", type=Path, help="results/xgboost_optuna_task8.json (for scale_pos_weight)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--legit-sample", type=int, default=10_000)
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    booster = xgb.Booster()
    booster.load_model(args.model_dir / "xgboost_tuned.json")
    best = booster.attr("best_iteration")
    n_rounds = int(best) + 1 if best is not None else booster.num_boosted_rounds()
    spw = float(json.loads(booster.save_config())["learner"]["objective"]["reg_loss_param"]["scale_pos_weight"])

    df = xb.load_merged(args.data)
    split = xb.assign_split(df, args.split_dir)
    train, valid = df[split == "train"], df[split == "valid"]
    del df
    train, [valid] = xb.add_features_v2(train, [valid])
    _, [X_va], cols, _ = xb.prepare_features(train, [valid], [xb.TARGET, xb.ID_COL, xb.TIME_COL])
    del train
    y = valid[xb.TARGET].values.astype(int)
    amt = valid[xb.AMOUNT_COL].values.astype(float)
    t = valid[xb.TIME_COL].values
    score = booster.predict(xgb.DMatrix(X_va, enable_categorical=True),
                            iteration_range=(0, n_rounds))
    saved = args.model_dir / "valid_scores.npy"
    if saved.exists():
        assert np.allclose(np.load(saved), score, atol=1e-5), "rebuilt features do not match the trained model"

    # ---------------- Task #19: operating points ----------------
    p = thresholds.prior_correct(score, spw)
    days = (t.max() - t.min()) / 86_400
    cal = thresholds.calibration_table(y, p)
    cal.to_csv(out / "calibration.csv")
    ops = thresholds.operating_table(y, p, amt, days=days)
    ops.to_csv(out / "operating_points.csv", index=False)
    thresholds.cost_sensitivity(y, p, amt).to_csv(out / "cost_sensitivity.csv", index=False)
    thresholds.threshold_stability(y, p, amt, t).to_csv(out / "threshold_stability.csv", index=False)
    thresholds.threshold_stability(y, p, amt, t, strategy="expected_loss").to_csv(
        out / "threshold_stability_expected_loss.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for strat, color, label in (("score", ORANGE, "Rank by fraud probability"),
                                ("expected_loss", BLUE, "Rank by probability x amount")):
        sub = ops[ops["strategy"] == strat]
        ax.plot(sub["review_rate"] * 100, sub["fraud_dollar_share"] * 100, marker="o", color=color, lw=2, label=label)
        for r, v in zip(sub["review_rate"] * 100, sub["fraud_dollar_share"] * 100):
            ax.annotate(f"{v:.0f}%", (r, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)
    ax.set(xlabel="Share of transactions reviewed (%)", ylabel="Fraud dollars caught (%)", ylim=(0, 100))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Fraud dollars caught at each review budget")
    fig.tight_layout()
    fig.savefig(out / "dollar_capture_by_strategy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.plot([0, cal["mean_predicted"].max()], [0, cal["mean_predicted"].max()], ls="--", color=INK2, lw=1)
    ax.plot(cal["mean_predicted"], cal["observed_rate"], marker="o", color=BLUE, lw=2)
    ax.set(xlabel="Predicted fraud probability", ylabel="Observed fraud rate", title="Calibration (deciles)")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=200)
    plt.close(fig)

    # ---------------- Task #18: SHAP ----------------
    rng = np.random.default_rng(42)
    k1 = int(np.ceil(0.01 * len(y)))
    flagged = np.zeros(len(y), dtype=bool)
    flagged[np.argsort(-score, kind="stable")[:k1]] = True
    legit = np.flatnonzero(y == 0)
    pick = np.zeros(len(y), dtype=bool)
    pick[y == 1] = True
    pick[flagged] = True
    pick[rng.choice(legit, size=min(args.legit_sample, len(legit)), replace=False)] = True
    X_s = X_va.iloc[np.flatnonzero(pick)]
    if n_rounds < booster.num_boosted_rounds():
        booster = booster[:n_rounds]  # SHAP on exactly the trees used for scoring
    shap = explain.shap_values(booster, X_s)
    shap.index = np.flatnonzero(pick)
    margin = booster.predict(xgb.DMatrix(X_s, enable_categorical=True), output_margin=True,
                             iteration_range=(0, n_rounds))
    assert np.allclose(shap.sum(axis=1).values, margin, atol=1e-3), "SHAP values do not add up to the score"

    # weight legit rows back to their population share so global importance reflects all validation rows
    w = np.where(y[shap.index] == 1, 1.0, (y == 0).sum() / min(args.legit_sample, len(legit)))
    w = np.where(flagged[shap.index] & (y[shap.index] == 0), 1.0, w)
    absw = shap.drop(columns="bias").abs().mul(w, axis=0).sum() / w.sum()
    imp = pd.DataFrame({"mean_abs_shap": absw.sort_values(ascending=False)})
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp["family"] = [explain.family_of(f) for f in imp.index]
    imp.to_csv(out / "shap_global_importance.csv")
    fam = explain.family_importance(imp)
    fam.to_csv(out / "shap_family_importance.csv")

    flagged_rows = shap.loc[np.flatnonzero(flagged)]
    reasons = explain.top_reasons(flagged_rows, k=3)
    explain.reason_frequency(reasons).to_csv(out / "reasons_flagged_by_family.csv")
    explain.reason_frequency(reasons, by_family=False).head(25).to_csv(out / "reasons_flagged_by_feature.csv")

    # Missed fraud: largest frauds outside the 5% review budget
    k5 = int(np.ceil(0.05 * len(y)))
    in5 = np.zeros(len(y), dtype=bool)
    in5[np.argsort(-score, kind="stable")[:k5]] = True
    missed = np.flatnonzero((y == 1) & ~in5)
    caught = np.flatnonzero((y == 1) & in5)
    miss_summary = pd.DataFrame({
        "group": ["frauds caught in top 5%", "frauds missed at 5%"],
        "count": [len(caught), len(missed)],
        "median_amount": [np.median(amt[caught]), np.median(amt[missed])],
        "total_amount": [amt[caught].sum(), amt[missed].sum()],
        "has_identity_share": [valid["has_identity"].values[caught].mean(), valid["has_identity"].values[missed].mean()],
        "top_product": [valid["ProductCD"].astype(str).values[caught], valid["ProductCD"].astype(str).values[missed]],
    })
    miss_summary["top_product"] = miss_summary["top_product"].map(lambda a: pd.Series(a).value_counts().index[0])
    miss_summary.to_csv(out / "missed_vs_caught.csv", index=False)
    fam_gap = (shap.loc[missed].drop(columns="bias").T.groupby(explain.family_of).sum().T.mean()
               - shap.loc[caught].drop(columns="bias").T.groupby(explain.family_of).sum().T.mean())
    fam_gap.sort_values().rename("mean_shap_missed_minus_caught").to_csv(out / "missed_vs_caught_by_family.csv")

    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    hbar(ax, [f"{f}  ({top.loc[f, 'family']})" for f in top.index], (top["share"] * 100).tolist(), lambda v: f"{v:.1f}%")
    ax.set(xlabel="Share of mean |SHAP| (%)")
    ax.set_title("Tuned XGBoost: 15 features with the largest effect on scores")
    fig.tight_layout()
    fig.savefig(out / "shap_top_features.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    hbar(ax, fam.index.tolist(), (fam["share"] * 100).tolist(), lambda v: f"{v:.0f}%")
    ax.set(xlabel="Share of mean |SHAP| (%)")
    ax.set_title("Which kinds of evidence drive the score")
    fig.tight_layout()
    fig.savefig(out / "shap_families.png", dpi=200)
    plt.close(fig)

    rf = explain.reason_frequency(reasons)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    hbar(ax, rf.index.tolist(), (rf["share_of_reasons"] * 100).tolist(), lambda v: f"{v:.0f}%", color=ORANGE)
    ax.set(xlabel="Share of top-3 reasons among transactions flagged at the 1% review rate (%)")
    ax.set_title("Why transactions get flagged")
    fig.tight_layout()
    fig.savefig(out / "flag_reasons.png", dpi=200)
    plt.close(fig)

    summary = {"scale_pos_weight": spw, "validation_days": days, "rows": int(len(y)), "frauds": int(y.sum()),
               "shap_rows": int(len(shap)), "flagged_1pct": int(k1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(ops.round(4).to_string())


if __name__ == "__main__":
    main()
