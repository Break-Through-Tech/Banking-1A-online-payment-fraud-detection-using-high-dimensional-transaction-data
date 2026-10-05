"""
Small synthetic stand-in for the IEEE-CIS files, used by the tests.

It mimics the parts of the schema our code depends on: TransactionDT over
~183 days, ~3.5% fraud propagated across a client, card/addr/D1 so a UID can
be rebuilt, V columns with block missingness, M flags, and an identity table
that covers only part of the transactions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_raw(n_clients: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    clients = pd.DataFrame({
        "card1": rng.integers(1000, 18000, n_clients),
        "addr1": rng.choice(np.arange(100, 540), n_clients),
        "first_day": rng.integers(0, 150, n_clients),
        "product": rng.choice(list("WCHRS"), n_clients, p=[0.6, 0.15, 0.1, 0.08, 0.07]),
        "email": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com"], n_clients),
        "device": rng.choice(["Windows", "iOS Device", "MacOS", "SM-G9", "Trident/7.0"], n_clients),
        "risky": rng.random(n_clients) < 0.06,
    })
    rows = []
    tid = 2_987_000
    for i, c in clients.iterrows():
        n_txn = rng.integers(1, 12)
        days = np.sort(c.first_day + rng.integers(0, 40, n_txn))
        days = days[days <= 182]
        compromised_from = days[rng.integers(0, len(days))] if c.risky and len(days) else None
        for d in days:
            fraud = int(compromised_from is not None and d >= compromised_from)
            amt = rng.lognormal(5.0 if fraud else 4.2, 1.0)
            rows.append({
                "TransactionID": tid, "isFraud": fraud,
                "TransactionDT": 86_400 + int(d) * 86_400 + int(rng.integers(0, 86_400)),
                "TransactionAmt": round(float(amt), 3), "ProductCD": c["product"],
                "card1": c.card1, "card2": float(rng.choice([111, 321, 545, np.nan])),
                "card4": rng.choice(["visa", "mastercard"]), "card6": rng.choice(["debit", "credit"]),
                "addr1": float(c.addr1) if rng.random() > 0.1 else np.nan, "addr2": 87.0,
                "P_emaildomain": c.email if rng.random() > 0.15 else None,
                "C1": float(rng.poisson(2 + 5 * fraud)), "D1": float(d - c.first_day),
                "M4": rng.choice(["M0", "M1", "M2", None]), "M6": rng.choice(["T", "F", None]),
                "_device": c.device, "_client": i,
            })
            tid += 1
    trans = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
    n = len(trans)
    # V columns: two blocks that are missing together, signal in V1/V3.
    block_a = rng.random(n) < 0.3
    block_b = rng.random(n) < 0.6
    for j in range(1, 7):
        v = rng.normal(size=n) + (0.8 * trans["isFraud"].values if j in (1, 3) else 0)
        trans[f"V{j}"] = np.where(block_a if j <= 3 else block_b, np.nan, v)

    has_id = (rng.random(n) < 0.25) | (trans["isFraud"].values == 1) & (rng.random(n) < 0.6)
    ident = pd.DataFrame({
        "TransactionID": trans.loc[has_id, "TransactionID"].values,
        "id_01": rng.choice([-5.0, -10.0, 0.0], has_id.sum()),
        "id_12": rng.choice(["Found", "NotFound"], has_id.sum()),
        "DeviceType": rng.choice(["desktop", "mobile"], has_id.sum()),
        "DeviceInfo": trans.loc[has_id, "_device"].values,
    })
    trans = trans.drop(columns=["_device", "_client"])
    return trans, ident
