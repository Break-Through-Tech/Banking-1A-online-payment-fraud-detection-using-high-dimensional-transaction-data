"""
Exploratory analysis helpers (Tasks #8, #10, #16).

All functions take the merged DataFrame and return tables, so results can be
saved, diffed, and reused in notebooks. Run everything at once with:

    python scripts/run_eda.py --data data/processed/merged.parquet

Analysis uses the TRAIN split only by default, so no decisions are made from
validation or test rows.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

TARGET = "isFraud"
TIME_COL = "TransactionDT"
SECONDS_PER_DAY = 86_400

CATEGORICAL_FIELDS = (
    ["ProductCD"] + [f"card{i}" for i in range(1, 7)]
    + ["addr1", "addr2", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 10)]
    + ["DeviceType", "DeviceInfo"]
)


# ---------------------------------------------------------------------------
# Task #8: missing values
# ---------------------------------------------------------------------------

def column_group(col: str) -> str:
    """Map a column to its documented feature group (V, C, D, M, id, card, ...)."""
    m = re.match(r"^(V|C|D|M|id_|card|addr|dist)\d", col)
    if m:
        return m.group(1).rstrip("_")
    if col.endswith("emaildomain"):
        return "email"
    if col.startswith("Device"):
        return "device"
    return "core"


def missingness_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Missing share per column, plus fraud rate when missing vs present.

    `fraud_lift` > 1 means rows where the column is missing are more likely
    fraud. The chi-square p-value tests whether missingness and the label
    are independent.
    """
    y = df[TARGET].values
    rows = []
    for col in df.columns:
        if col == TARGET:
            continue
        miss = df[col].isna().values
        share = miss.mean()
        row = {"column": col, "group": column_group(col), "missing_share": share}
        if 0 < share < 1:
            fr_miss, fr_present = y[miss].mean(), y[~miss].mean()
            table = np.array([[(y[miss] == 1).sum(), (y[miss] == 0).sum()],
                              [(y[~miss] == 1).sum(), (y[~miss] == 0).sum()]])
            p = stats.chi2_contingency(table)[1] if table.min() >= 0 and (table.sum(0) > 0).all() else np.nan
            row.update(fraud_rate_missing=fr_miss, fraud_rate_present=fr_present,
                       fraud_lift=fr_miss / fr_present if fr_present else np.nan, chi2_p=p)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("missing_share", ascending=False).reset_index(drop=True)


def missingness_by_group(by_col: pd.DataFrame) -> pd.DataFrame:
    return (by_col.groupby("group")
            .agg(columns=("column", "size"),
                 mean_missing=("missing_share", "mean"),
                 max_missing=("missing_share", "max"),
                 fully_present=("missing_share", lambda s: int((s == 0).sum())))
            .sort_values("mean_missing", ascending=False))


def missing_pattern_blocks(df: pd.DataFrame, prefix: str = "V", min_cols: int = 2) -> pd.DataFrame:
    """Group columns that share an identical missing-row count (a cheap proxy
    for an identical missing pattern). Used for V column reduction."""
    cols = [c for c in df.columns if re.fullmatch(rf"{prefix}\d+", c)]
    counts = df[cols].isna().sum()
    blocks = counts.groupby(counts).apply(lambda s: list(s.index))
    out = pd.DataFrame({"missing_rows": blocks.index, "columns": blocks.values})
    out["n_columns"] = out["columns"].str.len()
    return out[out["n_columns"] >= min_cols].sort_values("n_columns", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task #10: categorical cardinality and distributions
# ---------------------------------------------------------------------------

def categorical_profile(df: pd.DataFrame, fields=CATEGORICAL_FIELDS, rare_threshold: int = 50) -> pd.DataFrame:
    rows = []
    for col in [f for f in fields if f in df.columns]:
        s = df[col]
        vc = s.value_counts(dropna=True)
        fraud_by_level = df.groupby(s, observed=True)[TARGET].mean()
        rows.append({
            "column": col,
            "cardinality": int(s.nunique(dropna=True)),
            "missing_share": float(s.isna().mean()),
            "top_level": vc.index[0] if len(vc) else None,
            "top_share": float(vc.iloc[0] / s.notna().sum()) if len(vc) else np.nan,
            "rare_levels": int((vc < rare_threshold).sum()),
            "rows_in_rare_levels": float(vc[vc < rare_threshold].sum() / max(s.notna().sum(), 1)),
            "fraud_rate_min": float(fraud_by_level.min()) if len(fraud_by_level) else np.nan,
            "fraud_rate_max": float(fraud_by_level.max()) if len(fraud_by_level) else np.nan,
            "suggested_encoding": suggest_encoding(int(s.nunique(dropna=True))),
        })
    return pd.DataFrame(rows).sort_values("cardinality", ascending=False).reset_index(drop=True)


def suggest_encoding(cardinality: int) -> str:
    if cardinality <= 10:
        return "one-hot"
    if cardinality <= 100:
        return "one-hot after grouping rare levels"
    return "frequency encoding or native categorical (trees) / embedding (MLP)"


def level_fraud_rates(df: pd.DataFrame, col: str, min_count: int = 100) -> pd.DataFrame:
    g = df.groupby(df[col].astype("string").fillna("<missing>"), observed=True)[TARGET]
    out = pd.DataFrame({"count": g.size(), "fraud_rate": g.mean()})
    return out[out["count"] >= min_count].sort_values("fraud_rate", ascending=False)


# ---------------------------------------------------------------------------
# Task #16: class imbalance and temporal drift
# ---------------------------------------------------------------------------

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day"] = ((out[TIME_COL] - out[TIME_COL].min()) // SECONDS_PER_DAY).astype(int)
    out["week"] = out["day"] // 7
    out["month_30d"] = out["day"] // 30
    return out


def fraud_rate_over_time(df: pd.DataFrame, period: str = "week") -> pd.DataFrame:
    d = add_time_parts(df)
    g = d.groupby(period)[TARGET]
    return pd.DataFrame({"transactions": g.size(), "frauds": g.sum(), "fraud_rate": g.mean()})


def fraud_rate_trend_test(df: pd.DataFrame) -> dict:
    """Chi-square test that fraud rate is the same in every 30-day block."""
    d = add_time_parts(df)
    table = pd.crosstab(d["month_30d"], d[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p),
            "rates": d.groupby("month_30d")[TARGET].mean().round(4).to_dict()}


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population stability index. <0.1 stable, 0.1 to 0.25 moderate, >0.25 large shift."""
    expected, actual = expected.dropna(), actual.dropna()
    if expected.empty or actual.empty:
        return np.nan
    if pd.api.types.is_numeric_dtype(expected):
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    else:
        e = expected.astype(str).value_counts(normalize=True)
        a = actual.astype(str).value_counts(normalize=True).reindex(e.index, fill_value=0)
    e, a = e.clip(lower=1e-6), a.clip(lower=1e-6)
    return float(((a - e) * np.log(a / e)).sum())


def feature_drift(df: pd.DataFrame, columns=None, ref_months=(0,), cmp_months=(5,)) -> pd.DataFrame:
    """PSI of each column between a reference and a later 30-day block."""
    d = add_time_parts(df)
    columns = columns or [c for c in df.columns if c not in (TARGET, TIME_COL, "TransactionID")]
    ref = d[d["month_30d"].isin(ref_months)]
    cmp_ = d[d["month_30d"].isin(cmp_months)]
    rows = [{"column": c, "psi": psi(ref[c], cmp_[c])} for c in columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["shift"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "large"])
    return out
