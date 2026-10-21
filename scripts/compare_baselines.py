"""
Milestone review (Task #21): compare every baseline in the registry on the
same validation rows and write a markdown summary for the October handoff.

    python scripts/compare_baselines.py
    python scripts/compare_baselines.py --registry results/registry.jsonl --out reports/september_baselines.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIN_GAIN = 0.01  # AUPRC gain needed to call a winner (docs/business_objective.md 5.4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "september_baselines.md")
    args = ap.parse_args(argv)

    runs = registry.load_registry(args.registry)
    if runs.empty:
        sys.exit("Registry is empty. Run the baselines first.")

    split_ids = runs["split_id"].value_counts()
    main_split = split_ids.index[0]
    if len(split_ids) > 1:
        print(f"WARNING: runs use {len(split_ids)} different splits; comparing only {main_split}")

    board = registry.leaderboard(args.registry, split_id=main_split)
    seeds = runs[runs["split_id"] == main_split].groupby("model")["metrics.valid.auprc"].agg(["count", "std"])
    board = board.merge(seeds.rename(columns={"count": "runs", "std": "auprc_std"}),
                        left_on="model", right_index=True, how="left")

    best, runner_up = board.iloc[0], (board.iloc[1] if len(board) > 1 else None)
    gap = best["auprc"] - runner_up["auprc"] if runner_up is not None else float("nan")
    if runner_up is None:
        verdict = f"Only one model logged: {best['model']}."
    elif gap >= MIN_GAIN:
        verdict = f"**{best['model']}** leads by {gap:.3f} AUPRC, above the {MIN_GAIN} bar. Carry it forward."
    else:
        verdict = (f"{best['model']} and {runner_up['model']} are within {gap:.3f} AUPRC, below the "
                   f"{MIN_GAIN} bar. Treat as a tie; pick on recall@1% and training cost.")

    floor = board[board["model"].isin(["logreg", "rf"])]["auprc"].max()
    lines = [
        "# September baseline review (Task #21)",
        "",
        f"Split: `{main_split}`. All metrics on the validation split.",
        "",
        board.drop(columns=["split_id"]).round(4).to_markdown(index=False),
        "",
        verdict,
        "",
        f"Performance floor (best of logistic regression / random forest): AUPRC {floor:.4f}"
        if floor == floor else "Performance floor not logged yet (Task #17).",
        "",
        "## October handoff checklist",
        "- [ ] Agree the tree configuration the MLP must beat (same features, same split)",
        "- [ ] Confirm Task #15 entity-grouping verdict: sequence models in or out",
        "- [ ] Freeze feature_set name for October (bump to v2 if features change)",
        "- [ ] Bring open questions from docs/literature_review.md section 5 to the advisor",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
