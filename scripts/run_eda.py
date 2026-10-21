"""
Run the September EDA and entity feasibility checks (Tasks #8, #10, #15, #16).

    python scripts/run_eda.py --data data/processed/merged.parquet

Writes CSV tables, plots, and a short summary.md to reports/eda/.
Uses the train split only unless --all-rows is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import eda, entity, split  # noqa: E402
from src.data import PROCESSED_PATH  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "eda"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=PROCESSED_PATH)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--all-rows", action="store_true", help="Include valid/test rows (drift uses all rows regardless)")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(args.data)
    labels = split.make_split(full)
    train = full if args.all_rows else full[labels == "train"]
    print(f"Rows analysed: {len(train):,} ({'all' if args.all_rows else 'train only'})")

    # Task #8
    miss = eda.missingness_by_column(train)
    miss.to_csv(out / "missing_by_column.csv", index=False)
    groups = eda.missingness_by_group(miss)
    groups.to_csv(out / "missing_by_group.csv")
    vblocks = eda.missing_pattern_blocks(train, "V")
    vblocks.to_csv(out / "v_missing_blocks.csv", index=False)

    # Task #10
    cats = eda.categorical_profile(train)
    cats.to_csv(out / "categorical_profile.csv", index=False)
    for col in ("ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"):
        if col in train:
            eda.level_fraud_rates(train, col).to_csv(out / f"fraud_rate_by_{col}.csv")

    # Task #16 (drift needs the full timeline)
    weekly = eda.fraud_rate_over_time(full, "week")
    weekly.to_csv(out / "fraud_rate_weekly.csv")
    trend = eda.fraud_rate_trend_test(full)
    drift = eda.feature_drift(full)
    drift.to_csv(out / "feature_drift_psi.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(weekly.index, weekly["fraud_rate"] * 100, marker="o", ms=3)
    colors = {"train": "#4C78A8", "valid": "#F58518", "test": "#9D9D9D"}
    for name, (lo, hi) in split.SPLIT_DAYS.items():
        ax.axvspan(lo / 7, (hi + 1) / 7, color=colors[name], alpha=0.12, label=name)
    ax.set(xlabel="Week", ylabel="Fraud rate (%)", title="Weekly fraud rate across the split windows")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fraud_rate_weekly.png", dpi=150)
    plt.close(fig)

    # Task #15
    feas = entity.feasibility_report(full, labels)
    feas.to_csv(out / "entity_feasibility.csv")
    decision = entity.verdict(feas)

    top_lift = miss.dropna(subset=["fraud_lift"]).sort_values("fraud_lift", ascending=False).head(10)
    summary = [
        "# September EDA summary",
        "",
        f"Rows analysed: {len(train):,}. Fraud rate {train['isFraud'].mean():.2%}.",
        "",
        "## Missing values (Task #8)",
        groups.round(3).to_markdown(),
        "",
        f"V columns fall into {len(vblocks)} shared-missingness blocks.",
        "",
        "Columns where missingness most raises fraud rate:",
        top_lift[["column", "missing_share", "fraud_lift", "chi2_p"]].round(4).to_markdown(index=False),
        "",
        "## Categorical fields (Task #10)",
        cats[["column", "cardinality", "missing_share", "rare_levels", "suggested_encoding"]].round(3).to_markdown(index=False),
        "",
        "## Imbalance and drift (Task #16)",
        f"Fraud rate by 30-day block: {json.dumps(trend['rates'])}",
        f"Chi-square test of equal fraud rate across blocks: p = {trend['p_value']:.3g}",
        "",
        f"Features with large shift (PSI > 0.25, month 0 vs month 5): {int((drift['shift'] == 'large').sum())}",
        drift.head(15).round(3).to_markdown(index=False),
        "",
        "## Entity grouping (Task #15)",
        feas.round(3).to_markdown(),
        "",
        decision,
    ]
    (out / "summary.md").write_text("\n".join(summary))
    print(decision)
    print(f"Wrote reports to {out}")


if __name__ == "__main__":
    main()
