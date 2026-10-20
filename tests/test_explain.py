"""TreeSHAP attribution helpers (Oct Task #18)."""
import numpy as np
import pandas as pd
import xgboost as xgb

from fraud import explain


def test_shap_sums_to_raw_margin_and_finds_signal():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"V1": rng.normal(size=3000), "C1": rng.normal(size=3000), "card1": rng.normal(size=3000)})
    y = (X["V1"] + 0.1 * rng.normal(size=3000) > 1.5).astype(int)
    b = xgb.train({"objective": "binary:logistic", "max_depth": 3}, xgb.DMatrix(X, label=y), 30)
    s = explain.shap_values(b, X, batch=1000)
    margin = b.predict(xgb.DMatrix(X), output_margin=True)
    assert np.allclose(s.sum(axis=1), margin, atol=1e-4)
    imp = explain.global_importance(s)
    assert imp.index[0] == "V1" and imp.loc["V1", "family"] == "Vesta features (V)"
    fam = explain.family_importance(imp)
    assert abs(fam["share"].sum() - 1) < 1e-9
    r = explain.top_reasons(s, k=2)
    assert list(r.columns[:2]) == ["reason_1", "reason_2"]
    assert explain.reason_frequency(r[y.values == 1]).index[0] == "Vesta features (V)"  # frauds are flagged by V1


def test_family_mapping():
    cases = {"TransactionAmt": "Amount", "amt_cents": "Amount", "card1_freq": "Card", "D15_norm": "Time deltas (D)",
             "P_emaildomain": "Email", "id_31": "Device / identity", "DeviceInfo_freq": "Device / identity",
             "C13": "Counts (C)", "M4": "Match checks (M)", "addr1": "Address", "hour": "Time of day"}
    assert {k: explain.family_of(k) for k in cases} == cases
