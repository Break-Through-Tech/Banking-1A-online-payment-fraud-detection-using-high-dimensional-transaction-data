import numpy as np
import pytest

from fraud.models.ensemble import (
    combine_predictions,
    evaluate_ensemble,
)


def test_equal_weight_average():
    model_a = np.array([0.2, 0.8, 0.4])
    model_b = np.array([0.4, 0.6, 0.8])

    result = combine_predictions([model_a, model_b])

    np.testing.assert_allclose(
        result,
        [0.3, 0.7, 0.6]
    )


def test_weighted_average():
    model_a = np.array([0.2, 0.8])
    model_b = np.array([0.6, 0.4])

    result = combine_predictions(
        [model_a, model_b],
        weights=[3, 1]
    )

    np.testing.assert_allclose(
        result,
        [0.3, 0.7]
    )


def test_different_prediction_lengths():
    with pytest.raises(ValueError):
        combine_predictions([
            np.array([0.2, 0.8]),
            np.array([0.3])
        ])


def test_invalid_weights():
    with pytest.raises(ValueError):
        combine_predictions(
            [np.array([0.2, 0.8])],
            weights=[-1]
        )


def test_ensemble_evaluation():
    y_true = np.array([0, 1, 0, 1])
    predictions = [
        np.array([0.1, 0.9, 0.2, 0.8]),
        np.array([0.2, 0.8, 0.3, 0.7]),
    ]

    scores, metrics = evaluate_ensemble(
        y_true,
        predictions
    )

    assert len(scores) == len(y_true)
    assert "auprc" in metrics
    assert "roc_auc" in metrics
