"""Operating-point selection (Oct Task #19)."""
import numpy as np
import pytest

from fraud import thresholds as th


def test_prior_correction_inverts_weighting():
    # A weighted model's odds are inflated by w; correcting returns the true rate.
    true_p = np.array([0.01, 0.1, 0.5])
    w = 27.0
    inflated = true_p * w / (true_p * w + (1 - true_p))
    assert th.prior_correct(inflated, w) == pytest.approx(true_p)


def test_expected_loss_catches_more_dollars_at_same_budget():
    # Two equally likely frauds; the budget fits one. Expected loss picks the big one.
    y = np.array([1, 1, 0, 0])
    p = np.array([0.6, 0.59, 0.1, 0.1])
    amt = np.array([10.0, 1000.0, 5.0, 5.0])
    a = th.operating_point(y, p, amt, 0.25, "score")
    b = th.operating_point(y, p, amt, 0.25, "expected_loss")
    assert a["fraud_dollars_caught"] == 10 and b["fraud_dollars_caught"] == 1000
    assert a["recall"] == b["recall"] == 0.5
    assert b["total_cost"] < a["total_cost"]


def test_cost_accounting():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.1, 0.05])
    amt = np.array([100.0, 50.0, 20.0, 40.0])
    op = th.operating_point(y, p, amt, 0.5, "score")
    assert op["flagged"] == 2 and op["fraud_caught"] == 1 and op["precision"] == 0.5
    assert op["missed_fraud_cost"] == 40 + th.CHARGEBACK_FEE
    assert op["total_cost"] == op["missed_fraud_cost"] + 2 * th.REVIEW_COST
    assert op["savings_vs_no_model"] == pytest.approx((100 + 40 + 2 * th.CHARGEBACK_FEE) - op["total_cost"])


def test_sensitivity_and_stability_shapes():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.04).astype(int)
    p = np.clip(0.03 + 0.5 * y * rng.random(4000) + 0.05 * rng.random(4000), 0, 1)
    amt = rng.lognormal(4, 1, 4000)
    sens = th.cost_sensitivity(y, p, amt)
    assert list(sens["review_cost"]) == [2.0, 5.0, 10.0, 25.0]
    assert th.REVIEW_COST == 5.0  # restored
    stab = th.threshold_stability(y, p, amt, np.arange(4000))
    assert set(stab["window"]) == {"first half", "second half"} and len(stab) == 4
    cal = th.calibration_table(y, p)
    assert len(cal) == 10
