"""
Entity grouping feasibility (Task #15).

Question: can we build a stable card/client identifier from card1, addr1,
and the D columns? If yes, October can add past-only behavioral features and
possibly sequence models. If no, those stay out of scope.

Candidate keys (see docs/literature_review.md, section 1.1):

    uid_card      = card1
    uid_card_addr = card1 + addr1
    uid_magic     = card1 + addr1 + floor(day - D1)     # 1st place solution

How we judge a key
------------------
- coverage:        share of rows where the key can be built (no missing parts)
- multi_txn_share: share of rows whose key appears more than once (needed
                   for any history feature)
- median_size:     typical transactions per key
- d1_consistency:  within a key, how often consecutive transactions agree on
                   `day - D1` within 1 day (high = same client)
- label_purity:    share of keys whose transactions are all fraud or all
                   legit. Very high purity + many repeat keys is the label
                   propagation effect from the Vesta labeling logic.
- cross_split:     share of valid-split keys already seen in train. This is
                   how often a history feature would actually be available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400


def build_uids(df: pd.DataFrame) -> pd.DataFrame:
    day = (df["TransactionDT"] // SECONDS_PER_DAY).astype(int)
    out = pd.DataFrame(index=df.index)
    card1 = df["card1"].astype("Int64").astype("string")
    addr1 = df["addr1"].astype("Int64").astype("string")
    anchor = np.floor(day - df["D1"]).astype("Int64").astype("string")

    out["uid_card"] = card1
    out["uid_card_addr"] = card1 + "_" + addr1
    out["uid_magic"] = card1 + "_" + addr1 + "_" + anchor
    out["d1_anchor"] = day - df["D1"]
    return out


def score_key(df: pd.DataFrame, key: pd.Series, split: pd.Series | None = None) -> dict:
    present = key.notna()
    k = key[present]
    sizes = k.map(k.value_counts())
    res = {
        "coverage": float(present.mean()),
        "n_keys": int(k.nunique()),
        "multi_txn_share": float((sizes > 1).mean()),
        "median_size": float(k.value_counts().median()),
    }

    y = df.loc[present, "isFraud"]
    per_key = y.groupby(k).agg(["mean", "size"])
    multi = per_key[per_key["size"] > 1]
    res["label_purity"] = float(((multi["mean"] == 0) | (multi["mean"] == 1)).mean()) if len(multi) else np.nan
    res["fraud_keys_all_fraud"] = float((multi["mean"] == 1).sum() / max((multi["mean"] > 0).sum(), 1))

    if split is not None:
        train_keys = set(k[split[present] == "train"])
        valid = k[split[present] == "valid"]
        res["valid_rows_with_train_history"] = float(valid.isin(train_keys).mean()) if len(valid) else np.nan
    return res


def d1_consistency(df: pd.DataFrame, key: pd.Series, tol_days: float = 1.0) -> float:
    """Within a key, share of consecutive transactions whose day - D1 agree."""
    anchor = df["TransactionDT"] / SECONDS_PER_DAY - df["D1"]
    tmp = pd.DataFrame({"key": key, "anchor": anchor, "t": df["TransactionDT"]}).dropna()
    tmp = tmp.sort_values(["key", "t"])
    diff = tmp.groupby("key")["anchor"].diff().abs().dropna()
    return float((diff <= tol_days).mean()) if len(diff) else np.nan


def feasibility_report(df: pd.DataFrame, split: pd.Series | None = None) -> pd.DataFrame:
    uids = build_uids(df)
    rows = []
    for name in ("uid_card", "uid_card_addr", "uid_magic"):
        r = score_key(df, uids[name], split)
        r["d1_consistency"] = d1_consistency(df, uids[name])
        r["key"] = name
        rows.append(r)
    return pd.DataFrame(rows).set_index("key")


def verdict(report: pd.DataFrame, min_coverage=0.8, min_repeat=0.5, min_consistency=0.8) -> str:
    """Plain-language go / no-go for October sequence work."""
    m = report.loc["uid_magic"]
    ok = (m["coverage"] >= min_coverage and m["multi_txn_share"] >= min_repeat
          and m["d1_consistency"] >= min_consistency)
    return (
        f"uid_magic: coverage {m['coverage']:.0%}, repeat rows {m['multi_txn_share']:.0%}, "
        f"D1 consistency {m['d1_consistency']:.0%}. "
        + ("GO: usable for past-only aggregates; sequence models feasible for repeat clients."
           if ok else
           "NO-GO for sequence models: key is too sparse or unstable. Use aggregates on card1/addr1 only.")
    )
