"""
Ensemble baseline for payment fraud detection.

Combines prediction probabilities from multiple trained models
using weighted or equal-weight averaging.
"""

from __future__ import annotations

import numpy as np

from fraud.evaluation import evaluate


def combine_predictions(predictions, weights=None):
    """
    Combine prediction scores from multiple models.

    Parameters
    ----------
    predictions : list of array-like
        Each array contains fraud probabilities from one model.
    weights : list of float, optional
        Model weights. If omitted, all models receive equal weight.

    Returns
    -------
    numpy.ndarray
        Combined fraud probabilities.
    """
    if not predictions:
        raise ValueError("At least one model prediction is required.")

    scores = [np.asarray(p, dtype=float).reshape(-1)
              for p in predictions]

    if any(len(p) != len(scores[0]) for p in scores):
        raise ValueError("All prediction arrays must have equal length.")

    if not all(np.isfinite(p).all() for p in scores):
        raise ValueError("Predictions must contain finite values.")

    if any(((p < 0) | (p > 1)).any() for p in scores):
        raise ValueError("Predictions must be probabilities between 0 and 1.")

    if weights is None:
        weights = np.ones(len(scores), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(weights) != len(scores):
        raise ValueError("There must be one weight per model.")

    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and non-negative.")

    if weights.sum() <= 0:
        raise ValueError("At least one weight must be positive.")

    weights = weights / weights.sum()

    combined = np.average(
        np.vstack(scores),
        axis=0,
        weights=weights
    )

    return combined


def evaluate_ensemble(
    y_true,
    predictions,
    amount=None,
    weights=None,
    threshold=0.5,
):
    """
    Combine model predictions and evaluate the ensemble
    using the project's shared evaluation module.
    """
    combined_scores = combine_predictions(
        predictions,
        weights=weights
    )

    metrics = evaluate(
        y_true,
        combined_scores,
        amount=amount,
        threshold=threshold,
    )

    return combined_scores, metrics
