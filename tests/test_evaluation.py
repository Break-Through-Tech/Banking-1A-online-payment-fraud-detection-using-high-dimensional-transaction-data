import numpy as np

from src.evaluation import evaluate_model


def test_evaluate_model():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])

    results = evaluate_model(y_true, y_scores)

    assert "AUPRC" in results
    assert "Precision" in results
    assert "Recall" in results
    assert "F1" in results
    assert "FPR" in results
    assert "Recall@1%" in results
    assert "Recall@5%" in results
    assert "KS" in results

    assert results["Precision"] == 1.0
    assert results["Recall"] == 1.0
    assert results["F1"] == 1.0
    assert results["FPR"] == 0.0
    assert results["KS"] == 1.0
