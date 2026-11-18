"""One file format for saved model scores (Nov Task 6, issue #67).

Every final candidate saves its validation scores, and later its blind test
scores, as a Parquet file with two columns: TransactionID and score. Anything
that combines or compares models (the ensemble, the blind test, the figures)
then lines scores up by TransactionID instead of trusting row order.

    from fraud.scores import save_scores, load_scores
    save_scores(valid["TransactionID"], score, "reports/<model>/valid_scores.parquet")
    score = load_scores("reports/<model>/valid_scores.parquet", load_split("valid"))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraud.split import ID_COL


def save_scores(transaction_ids, scores, path):
    """Write one score per transaction to a Parquet file and return its path."""
    ids = np.asarray(transaction_ids)
    scores = np.asarray(scores, dtype=float).ravel()
    if len(ids) != len(scores):
        raise ValueError(f"{len(ids)} TransactionIDs but {len(scores)} scores.")
    if pd.Index(ids).has_duplicates:
        raise ValueError("TransactionIDs must be unique.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must all be finite numbers.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: ids, "score": scores}).to_parquet(path, index=False)
    return path


def load_scores(path, expected_ids):
    """Read a score file back as an array in the order of expected_ids.

    Raises an error if any expected TransactionID has no score, so a model
    scored on different rows can't slip into a comparison unnoticed.
    """
    frame = pd.read_parquet(path)
    by_id = frame.set_index(ID_COL)["score"]
    if by_id.index.has_duplicates:
        raise ValueError(f"{path} has duplicate TransactionIDs.")
    expected = pd.Index(expected_ids)
    missing = expected.difference(by_id.index)
    if len(missing):
        raise ValueError(f"{path} is missing scores for {len(missing)} of the {len(expected)} expected TransactionIDs.")
    return by_id.reindex(expected).to_numpy()
