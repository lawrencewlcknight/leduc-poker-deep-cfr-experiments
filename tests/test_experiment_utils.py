"""Tests for the multi-seed aggregation and export utilities."""

from __future__ import annotations

import json

import numpy as np
import pytest

from deep_cfr_poker.experiment_utils import (
    DEFAULT_FINAL_WINDOW,
    final_window_mean,
    final_window_std,
    first_nodes_to_threshold,
    first_time_to_threshold,
    json_safe,
    normalised_auc,
    resolve_solver_batch_sizes,
    summarise_numeric_fields,
    _pad_to_length,
    _stack_padded,
)


def test_json_safe_handles_numpy_scalars_and_arrays():
    payload = {
        "i": np.int64(7),
        "f": np.float32(0.5),
        "arr": np.array([1.0, 2.0]),
        "nested": [np.int32(1), {"x": np.float64(0.25)}],
        "tuple": (np.int8(1), np.int8(2)),
    }
    safe = json_safe(payload)
    json.dumps(safe)  # should not raise
    assert safe["i"] == 7
    assert safe["f"] == pytest.approx(0.5)
    assert safe["arr"] == [1.0, 2.0]
    assert safe["nested"][0] == 1
    assert safe["nested"][1]["x"] == pytest.approx(0.25)
    assert safe["tuple"] == [1, 2]


def test_first_nodes_to_threshold_returns_first_index_below_threshold():
    nodes = np.array([1, 2, 3, 4, 5])
    metric = np.array([0.9, 0.4, 0.2, 0.1, 0.05])
    assert first_nodes_to_threshold(nodes, metric, 0.3) == 3.0


def test_first_nodes_to_threshold_returns_nan_when_unmet():
    metric = np.array([0.9, 0.8, 0.7])
    nodes = np.array([1, 2, 3])
    assert np.isnan(first_nodes_to_threshold(nodes, metric, 0.05))


def test_first_time_to_threshold_returns_first_time_below_threshold():
    times = np.array([10.0, 20.0, 30.0, 40.0])
    metric = np.array([0.6, 0.5, 0.2, 0.1])
    assert first_time_to_threshold(times, metric, 0.3) == 30.0


def test_final_window_mean_handles_short_arrays():
    assert np.isnan(final_window_mean(np.array([])))
    assert final_window_mean(np.array([1.0])) == pytest.approx(1.0)
    assert final_window_mean(
        np.array([1.0, 2.0, 3.0, 4.0]), window=DEFAULT_FINAL_WINDOW
    ) == pytest.approx(2.5)


def test_final_window_std_uses_last_window():
    assert final_window_std(np.array([])) == 0.0
    assert final_window_std(np.array([1.0])) == 0.0
    assert final_window_std(np.array([1.0, 2.0, 3.0]), window=2) == pytest.approx(
        np.std([2.0, 3.0], ddof=1)
    )


def test_normalised_auc_drops_nan_and_normalises_by_range():
    auc = normalised_auc(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([1.0, 2.0, float("nan"), 4.0]),
    )
    assert auc == pytest.approx(np.trapz([1.0, 2.0, 4.0], [0.0, 1.0, 3.0]) / 3.0)


def test_resolve_solver_batch_sizes_falls_back_to_positive_minibatches():
    assert resolve_solver_batch_sizes(
        {"batch_size_advantage": None, "batch_size_strategy": None},
        default_batch_size=32,
    ) == (32, 32)
    assert resolve_solver_batch_sizes(
        {"batch_size_advantage": 128, "batch_size_strategy": None}
    ) == (128, 128)
    assert resolve_solver_batch_sizes(
        {"batch_size_advantage": 128, "batch_size_strategy": 64}
    ) == (128, 64)
    assert resolve_solver_batch_sizes(
        {"batch_size_advantage": 0, "batch_size_strategy": 0},
        default_batch_size=32,
    ) == (32, 32)


def test_summarise_numeric_fields_skips_seed_and_handles_nan():
    rows = [
        {"seed": 1, "metric_a": 1.0, "metric_b": float("nan")},
        {"seed": 2, "metric_a": 2.0, "metric_b": 3.0},
        {"seed": 3, "metric_a": 3.0, "metric_b": 5.0},
    ]
    summary = summarise_numeric_fields(rows)
    assert "seed" not in summary
    assert summary["metric_a"]["mean"] == pytest.approx(2.0)
    assert summary["metric_a"]["n_finite"] == 3
    # NaNs are dropped before averaging.
    assert summary["metric_b"]["n_finite"] == 2
    assert summary["metric_b"]["mean"] == pytest.approx(4.0)


def test_summarise_numeric_fields_returns_empty_dict_for_empty_input():
    assert summarise_numeric_fields([]) == {}


def test_pad_to_length_pads_with_nan_for_floats():
    arr = np.array([1.0, 2.0, 3.0])
    padded = _pad_to_length(arr, 5)
    assert padded.shape == (5,)
    assert padded[0] == 1.0 and padded[2] == 3.0
    assert np.isnan(padded[3]) and np.isnan(padded[4])


def test_pad_to_length_pads_integers_with_zero():
    arr = np.array([1, 2, 3], dtype=np.int64)
    padded = _pad_to_length(arr, 5)
    assert padded.shape == (5,)
    np.testing.assert_array_equal(padded, np.array([1, 2, 3, 0, 0]))


def test_stack_padded_handles_ragged_inputs():
    arrays = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
    stacked = _stack_padded(arrays)
    assert stacked.shape == (2, 3)
    assert np.isnan(stacked[0, 2])
    assert stacked[1, 2] == 5.0
