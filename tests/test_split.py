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


def test_split_id_ignores_line_endings(tmp_path):
    """Regression: Windows checkouts (CRLF) used to log a different split_id."""
    df = _frame()
    split.save_split(df, tmp_path / "lf")
    crlf = tmp_path / "crlf"
    crlf.mkdir()
    for name in split.SPLIT_DAYS:
        data = (tmp_path / "lf" / f"{name}_ids.csv").read_bytes()
        (crlf / f"{name}_ids.csv").write_bytes(data.replace(b"\n", b"\r\n"))
    assert b"\r\n" in (crlf / "train_ids.csv").read_bytes()
    assert split.split_id(crlf) == split.split_id(tmp_path / "lf")


def test_xgboost_fallback_hash_matches_split_module(tmp_path, monkeypatch):
    """The baseline's own hash (used if fraud.split is missing) must agree with split_id()."""
    import builtins

    from fraud.models import xgboost_baseline as xb
    df = _frame()
    df["TransactionAmt"] = 10.0
    split.save_split(df, tmp_path)
    for name in split.SPLIT_DAYS:  # also check it with CRLF files
        p = tmp_path / f"{name}_ids.csv"
        p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    real_import = builtins.__import__

    def no_split(name, *a, **k):
        if name == "fraud.split":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_split)
    assert xb.assign_split(df, tmp_path).attrs["source"] == split.split_id(tmp_path)
