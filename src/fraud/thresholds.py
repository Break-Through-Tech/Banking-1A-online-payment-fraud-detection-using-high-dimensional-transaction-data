"""
Operating-point selection against the fraud team's review budget (Oct Task #19).

Given validation labels, model scores and transaction amounts, this module

1. converts scores to calibrated fraud probabilities (undoing the
   scale_pos_weight inflation, so probabilities can be compared to dollars),
2. compares two ways of filling a fixed review budget:
     - "score":         review the transactions most likely to be fraud
     - "expected_loss": review the transactions with the largest
                        probability x amount, i.e. the most fraud dollars at risk
3. picks a threshold for each review rate (default 1% and 5%, from
   docs/business_objective.md section 4) and reports precision, recall, false
   positive rate, fraud dollars caught and expected cost,
4. checks that the threshold is stable by recomputing it on the first and
   second half of the validation window.

Cost model (business objective section 3, placeholders until the Challenge
Advisor confirms): a missed fraud costs its amount plus a $20 chargeback fee;
each review costs $5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHARGEBACK_FEE = 20.0
REVIEW_COST = 5.0
REVIEW_RATES = (0.005, 0.01, 0.02, 0.05, 0.10)


def prior_correct(score, pos_weight: float) -> np.ndarray:
    """Undo class weighting: a model trained with positive weight w outputs
    odds inflated by w, so p = s / (s + (1 - s) * w)."""
    s = np.clip(np.asarray(score, dtype=float), 1e-12, 1 - 1e-12)
    return s / (s + (1 - s) * pos_weight)


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed fraud rate by predicted-probability decile."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin")
    return pd.DataFrame({"mean_predicted": g["p"].mean(), "observed_rate": g["y"].mean(), "rows": g.size()})


def _ranking(strategy: str, p: np.ndarray, amount: np.ndarray) -> np.ndarray:
    if strategy == "score":
        key = p
    elif strategy == "expected_loss":
        key = p * (amount + CHARGEBACK_FEE)
    else:
        raise ValueError(strategy)
    return np.argsort(-key, kind="stable"), key


def operating_point(y, p, amount, rate: float, strategy: str = "score", days: float | None = None) -> dict:
    y = np.asarray(y).astype(int)
    amount = np.asarray(amount, dtype=float)
    order, key = _ranking(strategy, p, amount)
    k = max(1, int(np.ceil(rate * len(y))))
    flagged = np.zeros(len(y), dtype=bool)
    flagged[order[:k]] = True
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    fraud_dollars = float((amount * y).sum())
    caught_dollars = float((amount * y)[flagged].sum())
    missed_cost = float(((amount + CHARGEBACK_FEE) * y)[~flagged].sum())
    no_model_cost = float(((amount + CHARGEBACK_FEE) * y).sum())
    total_cost = missed_cost + REVIEW_COST * k
    out = {
        "strategy": strategy,
        "review_rate": rate,
        "flagged": k,
        "threshold": float(key[order[k - 1]]),
        "precision": tp / k,
        "recall": tp / max(y.sum(), 1),
        "fpr": fp / max((y == 0).sum(), 1),
        "fraud_caught": tp,
        "fraud_missed": fn,
        "fraud_dollars_caught": caught_dollars,
        "fraud_dollar_share": caught_dollars / max(fraud_dollars, 1e-9),
        "review_cost": REVIEW_COST * k,
        "missed_fraud_cost": missed_cost,
        "total_cost": total_cost,
        "savings_vs_no_model": no_model_cost - total_cost,
    }
    if days:
        out["reviews_per_day"] = k / days
    return out


def operating_table(y, p, amount, rates=REVIEW_RATES, strategies=("score", "expected_loss"), days=None) -> pd.DataFrame:
    return pd.DataFrame([operating_point(y, p, amount, r, s, days) for s in strategies for r in rates])


def cost_sensitivity(y, p, amount, rates=REVIEW_RATES, review_costs=(2.0, 5.0, 10.0, 25.0)) -> pd.DataFrame:
    """Which review rate minimizes total cost as the cost of a review changes."""
    global REVIEW_COST
    saved = REVIEW_COST
    rows = []
    try:
        for c in review_costs:
            REVIEW_COST = c
            t = operating_table(y, p, amount, rates, ("expected_loss",))
            best = t.loc[t["total_cost"].idxmin()]
            rows.append({"review_cost": c, "best_review_rate": best["review_rate"],
                         "total_cost": best["total_cost"], "savings_vs_no_model": best["savings_vs_no_model"]})
    finally:
        REVIEW_COST = saved
    return pd.DataFrame(rows)


def threshold_stability(y, p, amount, t, rates=(0.01, 0.05), strategy="score") -> pd.DataFrame:
    """Recompute operating points separately on the first and second half of
    the validation window (split by transaction time)."""
    t = np.asarray(t)
    mid = np.median(t)
    rows = []
    for name, mask in (("first half", t <= mid), ("second half", t > mid)):
        for r in rates:
            op = operating_point(np.asarray(y)[mask], np.asarray(p)[mask], np.asarray(amount)[mask], r, strategy)
            rows.append({"window": name, "review_rate": r, "threshold": op["threshold"],
                         "precision": op["precision"], "recall": op["recall"],
                         "fraud_dollar_share": op["fraud_dollar_share"]})
    return pd.DataFrame(rows)
