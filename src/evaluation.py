"""
Shared evaluation module (Tasks #7 and #12).

Every model calls `evaluate`. Nobody computes metrics by hand.

    evaluate(y_true, y_score, amount=None, threshold=0.5) -> dict
    dollar_capture_table(y_true, y_score, amount)        -> DataFrame
    threshold_at_review_rate(y_score, rate)              -> float
    plot_pr_curve / plot_confusion_matrix / plot_score_distribution

Metric definitions follow docs/business_objective.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

REVIEW_RATES = (0.01, 0.05)
CAPTURE_BUCKETS = (0.01, 0.05, 0.10, 0.20)


def _arrays(y_true, y_score):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must be the same length")
    if np.isnan(y_score).any():
        raise ValueError("y_score contains NaN")
    if y_true.sum() == 0:
        raise ValueError("No fraud cases in y_true; AUPRC is undefined")
    return y_true, y_score


def _top_k(n: int, rate: float) -> int:
    return max(1, int(np.ceil(rate * n)))


def threshold_at_review_rate(y_score, rate: float) -> float:
    """Score cut-off that flags the top `rate` share of transactions."""
    y_score = np.asarray(y_score, dtype=float)
    k = _top_k(len(y_score), rate)
    return float(np.sort(y_score)[::-1][k - 1])


def recall_at_review_rate(y_true, y_score, rate: float) -> float:
    y_true, y_score = _arrays(y_true, y_score)
    top = np.argsort(-y_score, kind="stable")[: _top_k(len(y_score), rate)]
    return float(y_true[top].sum() / y_true.sum())


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def dollar_capture_table(y_true, y_score, amount, buckets=CAPTURE_BUCKETS) -> pd.DataFrame:
    """Fraud case share and fraud dollar share in the top risk buckets (Task #12)."""
    y_true, y_score = _arrays(y_true, y_score)
    amount = np.asarray(amount, dtype=float).ravel()
    order = np.argsort(-y_score, kind="stable")
    fraud_dollars = amount * y_true
    rows = []
    for b in buckets:
        top = order[: _top_k(len(y_score), b)]
        rows.append({
            "bucket": f"top {b:.0%}",
            "transactions": len(top),
            "fraud_cases": int(y_true[top].sum()),
            "fraud_case_share": y_true[top].sum() / y_true.sum(),
            "fraud_dollars": float(fraud_dollars[top].sum()),
            "fraud_dollar_share": fraud_dollars[top].sum() / max(fraud_dollars.sum(), 1e-9),
            "precision": y_true[top].mean(),
        })
    return pd.DataFrame(rows)


def evaluate(y_true, y_score, amount=None, threshold: float = 0.5) -> dict:
    """Full metric set. `threshold` only affects the threshold-based metrics."""
    y_true, y_score = _arrays(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    out = {
        "n": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    for r in REVIEW_RATES:
        out[f"recall_at_{int(r * 100)}pct_review"] = recall_at_review_rate(y_true, y_score, r)
        t = threshold_at_review_rate(y_score, r)
        neg = y_true == 0
        out[f"fpr_at_{int(r * 100)}pct_review"] = float((y_score[neg] >= t).mean())

    if amount is not None:
        table = dollar_capture_table(y_true, y_score, amount)
        out["dollar_capture"] = {
            row["bucket"].replace(" ", "").replace("%", "pct"): {
                "fraud_case_share": float(row["fraud_case_share"]),
                "fraud_dollar_share": float(row["fraud_dollar_share"]),
            }
            for _, row in table.iterrows()
        }
    return out


def headline(metrics: dict) -> str:
    """One-line summary for logs."""
    return (
        f"AUPRC {metrics['auprc']:.4f} | ROC-AUC {metrics['roc_auc']:.4f} | "
        f"R@1% {metrics['recall_at_1pct_review']:.3f} | R@5% {metrics['recall_at_5pct_review']:.3f} | "
        f"KS {metrics['ks']:.3f}"
    )


# ---------------------------------------------------------------------------
# Standard plots (Task #12). Matplotlib is imported lazily.
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pr_curve(y_true, scores: dict[str, np.ndarray], out_path: Path | str):
    """PR curves for one or more models on the same rows."""
    plt = _plt()
    y_true = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{name} (AP {average_precision_score(y_true, s):.3f})")
    ax.axhline(y_true.mean(), ls="--", color="grey", lw=1, label=f"base rate {y_true.mean():.3f}")
    ax.set(xlabel="Recall (fraud caught)", ylabel="Precision", title="Precision-recall, validation")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(y_true, y_score, threshold: float, out_path: Path | str):
    plt = _plt()
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
           xlabel="Predicted", ylabel="Actual", title=f"Threshold {threshold:.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_score_distribution(y_true, y_score, out_path: Path | str):
    plt = _plt()
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 51) if y_score.min() >= 0 and y_score.max() <= 1 else 50
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.6, density=True, label="Legit")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.6, density=True, label="Fraud")
    ax.set(xlabel="Predicted fraud score", ylabel="Density", title="Score distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_standard_report(y_true, y_score, amount, out_dir: Path | str, name: str, review_rate: float = 0.01):
    """Write the capture table and the three standard plots for one model."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dollar_capture_table(y_true, y_score, amount).to_csv(out_dir / f"{name}_capture.csv", index=False)
    t = threshold_at_review_rate(y_score, review_rate)
    plot_pr_curve(y_true, {name: y_score}, out_dir / f"{name}_pr.png")
    plot_confusion_matrix(y_true, y_score, t, out_dir / f"{name}_confusion.png")
    plot_score_distribution(y_true, y_score, out_dir / f"{name}_scores.png")
    return out_dir
