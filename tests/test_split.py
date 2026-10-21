"""Frozen chronological split (Task #6)."""
import numpy as np
import pandas as pd

from fraud import split


def _frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": 86_400 + rng.integers(0, 183 * 86_400, n),
        "isFraud": (rng.random(n) < 0.035).astype(int),
    })


def test_split_is_chronological_disjoint_and_reproducible(tmp_path):
    df = _frame()
    split.save_split(df, tmp_path / "a")
    split.save_split(df, tmp_path / "b")
    ids = {n: set(split.load_split(n, tmp_path / "a")) for n in split.SPLIT_DAYS}
    assert sum(map(len, ids.values())) == len(df)
    assert not (ids["train"] & ids["valid"]) and not (ids["valid"] & ids["test"])
    t = df.set_index("TransactionID")["TransactionDT"]
    assert t[list(ids["train"])].max() < t[list(ids["valid"])].min()
    assert split.split_id(tmp_path / "a") == split.split_id(tmp_path / "b")


def test_committed_split_files_match_expected_sizes():
    sizes = {n: len(split.load_split(n)) for n in split.SPLIT_DAYS}
    assert sizes == {"train": 420_066, "valid": 85_044, "test": 85_430}
    assert split.split_id() == "chrono-v1-c3eab22dc2"
