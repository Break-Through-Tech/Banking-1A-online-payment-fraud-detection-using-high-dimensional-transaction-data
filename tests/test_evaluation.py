import numpy as np

from fraud.evaluation import evaluate


def test_evaluate():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    results = evaluate(y_true, y_score)

    assert results["auprc"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1"] == 1.0
    assert results["fpr"] == 0.0
    assert results["recall_at_1pct_review"] == 0.5
    assert results["recall_at_5pct_review"] == 0.5
    assert results["ks"] == 1.0
    assert results["roc_auc"] == 1.0
    assert results["confusion"] == {
        "tn": 2,
        "fp": 0,
        "fn": 0,
        "tp": 2,
    }
