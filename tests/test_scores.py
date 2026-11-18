"""Shared score file format (Nov Task 6)."""
import numpy as np
import pytest

from fraud.scores import load_scores, save_scores


def test_round_trip_lines_scores_up_by_id(tmp_path):
    path = save_scores([30, 10, 20], [0.3, 0.1, 0.2], tmp_path / "scores.parquet")

    # Asked for in a different order, each ID still gets its own score.
    np.testing.assert_allclose(load_scores(path, [10, 20, 30]), [0.1, 0.2, 0.3])


def test_missing_id_is_an_error(tmp_path):
    path = save_scores([10, 20], [0.1, 0.2], tmp_path / "scores.parquet")

    with pytest.raises(ValueError, match="missing scores for 1"):
        load_scores(path, [10, 20, 30])


def test_bad_input_is_refused(tmp_path):
    with pytest.raises(ValueError):
        save_scores([10, 20], [0.1], tmp_path / "a.parquet")
    with pytest.raises(ValueError):
        save_scores([10, 10], [0.1, 0.2], tmp_path / "b.parquet")
    with pytest.raises(ValueError):
        save_scores([10, 20], [0.1, float("nan")], tmp_path / "c.parquet")
