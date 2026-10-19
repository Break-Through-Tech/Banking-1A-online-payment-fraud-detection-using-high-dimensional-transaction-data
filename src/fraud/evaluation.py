import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def _recall_at_review_rate(y_true, y_score, rate):
    total_fraud = int(np.sum(y_true))
    if total_fraud == 0:
        return 0.0

    n_review = min(len(y_true), max(1, int(np.ceil(len(y_true) * rate))))
    top_indices = np.argsort(y_score)[::-1][:n_review]
    return float(np.sum(y_true[top_indices]) / total_fraud)


def _dollar_capture(y_true, y_score, amount):
    results = {}
    total_fraud_dollars = float(np.sum(amount[y_true == 1]))

    for rate in (0.01, 0.05, 0.10, 0.20):
        n_review = min(len(y_true), max(1, int(np.ceil(len(y_true) * rate))))
        top_indices = np.argsort(y_score)[::-1][:n_review]
        captured = float(np.sum(amount[top_indices][y_true[top_indices] == 1]))

        results[f"top_{int(rate * 100)}pct"] = {
            "fraud_dollars_captured": captured,
            "fraud_dollar_capture_rate": (
                captured / total_fraud_dollars
                if total_fraud_dollars > 0 else 0.0
            ),
            "fraud_cases_captured": int(np.sum(y_true[top_indices] == 1)),
            "fraud_case_capture_rate": (
                float(np.sum(y_true[top_indices] == 1) / np.sum(y_true))
                if np.sum(y_true) > 0 else 0.0
            ),
        }

    return results


def evaluate(y_true, y_score, amount=None, threshold=0.5):
    """Evaluate fraud scores using shared project metrics."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score must contain only finite values.")

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_score)

    metrics = {
        "auprc": float(average_precision_score(y_true, y_score)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "recall_at_1pct_review": _recall_at_review_rate(y_true, y_score, 0.01),
        "recall_at_5pct_review": _recall_at_review_rate(y_true, y_score, 0.05),
        "ks": float(np.max(tpr_curve - fpr_curve)),
        "confusion": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
        },
    }

    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) == 2 else float("nan")
    )

    if amount is not None:
        amount = np.asarray(amount, dtype=float).ravel()
        if len(amount) != len(y_true):
            raise ValueError("amount must have the same length as y_true.")
        metrics["dollar_capture"] = _dollar_capture(y_true, y_score, amount)

    return metrics