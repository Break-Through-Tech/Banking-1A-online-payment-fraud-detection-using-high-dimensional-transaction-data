"""
Feature attribution for the tuned XGBoost model (October Task #18).

Uses XGBoost's built-in TreeSHAP (`pred_contribs=True`), which gives exact
SHAP values for tree ensembles, so no extra library is needed. Each value is
a feature's contribution to one transaction's log-odds of fraud, and the
contributions plus a bias term add up to the model's raw score.

Outputs tie the attributions back to the business framing in
docs/business_objective.md:

- global importance (mean |SHAP|) per feature and per feature family,
- the most common top reasons among transactions the model would send to
  review at the 1% operating point,
- missed frauds: what the model saw for the largest frauds it scored low.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import xgboost as xgb

# Plain-language meaning of each feature family. Most columns are anonymized by
# Vesta, so these follow the Kaggle data description and our EDA, not exact
# definitions.
FAMILIES = [
    (r"^TransactionAmt$|^amt_", "Amount", "How much is being spent, and odd amounts (cents pattern)"),
    (r"^ProductCD$", "Product", "Which product or channel the purchase is for"),
    (r"^card\d(_freq)?$", "Card", "Card issuer, type and network, and how common the card group is"),
    (r"^addr\d(_freq)?$|^dist\d$", "Address", "Billing region and country, and distances between addresses"),
    (r"emaildomain", "Email", "Purchaser and recipient email providers"),
    (r"^C\d+$", "Counts (C)", "How many addresses, cards or accounts are linked to this payment (velocity)"),
    (r"^D\d+(_norm)?$", "Time deltas (D)", "Days since earlier events, such as the card's first use"),
    (r"^M\d$", "Match checks (M)", "Whether details such as name and address match"),
    (r"^V\d+$", "Vesta features (V)", "Vesta's engineered counts, rankings and relationships"),
    (r"^id_\d+(_freq)?$|^Device", "Device / identity", "Device, browser, OS and network fingerprint"),
    (r"^hour$", "Time of day", "Hour the transaction happened"),
    (r"^has_identity$", "Identity present", "Whether an identity record exists at all"),
]


def family_of(feature: str) -> str:
    for pattern, name, _ in FAMILIES:
        if re.search(pattern, feature):
            return name
    return "Other"


FAMILY_MEANING = {name: meaning for _, name, meaning in FAMILIES}


def shap_values(booster: xgb.Booster, X: pd.DataFrame, batch: int = 5000) -> pd.DataFrame:
    """TreeSHAP contributions, one column per feature plus 'bias'."""
    parts = []
    for i in range(0, len(X), batch):
        d = xgb.DMatrix(X.iloc[i:i + batch], enable_categorical=True)
        parts.append(booster.predict(d, pred_contribs=True))
    vals = np.vstack(parts)
    return pd.DataFrame(vals, columns=list(X.columns) + ["bias"], index=X.index)


def global_importance(shap: pd.DataFrame) -> pd.DataFrame:
    s = shap.drop(columns="bias")
    imp = s.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp["family"] = [family_of(f) for f in imp.index]
    return imp


def family_importance(imp: pd.DataFrame) -> pd.DataFrame:
    fam = imp.groupby("family").agg(share=("share", "sum"), features=("share", "size"))
    fam["meaning"] = [FAMILY_MEANING.get(f, "") for f in fam.index]
    return fam.sort_values("share", ascending=False)


def top_reasons(shap: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """For each row, the k features pushing hardest towards fraud."""
    s = shap.drop(columns="bias")
    vals = s.values
    idx = np.argsort(-vals, axis=1)[:, :k]
    cols = np.array(s.columns)
    out = pd.DataFrame({f"reason_{j + 1}": cols[idx[:, j]] for j in range(k)}, index=shap.index)
    for j in range(k):
        out[f"reason_{j + 1}_shap"] = vals[np.arange(len(vals)), idx[:, j]]
    return out


def reason_frequency(reasons: pd.DataFrame, by_family: bool = True) -> pd.DataFrame:
    cols = [c for c in reasons.columns if re.fullmatch(r"reason_\d", c)]
    flat = reasons[cols].stack()
    if by_family:
        flat = flat.map(family_of)
    counts = flat.value_counts()
    return pd.DataFrame({"count": counts, "share_of_reasons": counts / len(flat)})
