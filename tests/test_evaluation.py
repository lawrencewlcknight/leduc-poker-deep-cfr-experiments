"""Tests for the evaluation module."""

from __future__ import annotations

import numpy as np
import pytest

from deep_cfr_poker.evaluation import (
    HeadToHeadRecord,
    aggregate_head_to_head,
    aggregate_strength,
    monotonicity_summary,
    strength_per_checkpoint,
)


def _records_for_three_checkpoint_run(seed: str = "0"):
    # Construct a small deterministic head-to-head matrix.
    #
    #         ckpt 100   ckpt 200   ckpt 300
    # ckpt 100   0.0      -0.10      -0.20
    # ckpt 200   0.10      0.0       -0.05
    # ckpt 300   0.20      0.05       0.0
    matrix = {
        100: {100: 0.0, 200: -0.10, 300: -0.20},
        200: {100: 0.10, 200: 0.0, 300: -0.05},
        300: {100: 0.20, 200: 0.05, 300: 0.0},
    }
    records = []
    for a, row in matrix.items():
        for b, value in row.items():
            records.append(
                HeadToHeadRecord(
                    seed=seed,
                    checkpoint_a=a,
                    checkpoint_b=b,
                    A_EV_as_player_0=value,
                    A_EV_as_player_1=value,
                    A_EV_seat_averaged=value,
                )
            )
    return records


def test_aggregate_head_to_head_diagonal_is_zero():
    records = _records_for_three_checkpoint_run()
    mean, win_fraction, iterations = aggregate_head_to_head(records, equivalence_epsilon=1e-3)
    assert iterations == [100, 200, 300]
    assert mean[0, 0] == pytest.approx(0.0)
    assert mean[1, 1] == pytest.approx(0.0)
    assert mean[2, 2] == pytest.approx(0.0)


def test_aggregate_head_to_head_off_diagonal_means():
    records = _records_for_three_checkpoint_run()
    mean, win_fraction, _ = aggregate_head_to_head(records, equivalence_epsilon=1e-3)
    # Single seed -> mean equals the underlying record value.
    assert mean[1, 0] == pytest.approx(0.10)
    assert mean[2, 0] == pytest.approx(0.20)
    assert mean[0, 1] == pytest.approx(-0.10)
    # win_fraction reports 1.0 wherever the entry is positive and >epsilon.
    assert win_fraction[2, 0] == pytest.approx(1.0)
    assert win_fraction[0, 2] == pytest.approx(0.0)


def test_aggregate_averages_across_multiple_seeds():
    seed_a = _records_for_three_checkpoint_run("a")
    seed_b_records = []
    # In seed b, all later-vs-earlier EVs are doubled.
    for record in _records_for_three_checkpoint_run("b"):
        seed_b_records.append(
            HeadToHeadRecord(
                seed=record.seed,
                checkpoint_a=record.checkpoint_a,
                checkpoint_b=record.checkpoint_b,
                A_EV_as_player_0=record.A_EV_as_player_0 * 2,
                A_EV_as_player_1=record.A_EV_as_player_1 * 2,
                A_EV_seat_averaged=record.A_EV_seat_averaged * 2,
            )
        )
    mean, _, _ = aggregate_head_to_head(seed_a + seed_b_records, equivalence_epsilon=1e-3)
    # 200 vs 100: seed a says 0.10, seed b says 0.20 -> mean 0.15.
    assert mean[1, 0] == pytest.approx(0.15)


def test_monotonicity_summary_classifies_strict_improvement():
    records = _records_for_three_checkpoint_run()
    _, _, iterations = aggregate_head_to_head(records, equivalence_epsilon=1e-3)
    # Build a per-seed matrix from the same records.
    matrix = np.array(
        [
            [0.0, -0.10, -0.20],
            [0.10, 0.0, -0.05],
            [0.20, 0.05, 0.0],
        ]
    )
    summaries = monotonicity_summary(
        {"0": matrix}, iterations, equivalence_epsilon=1e-3
    )
    assert len(summaries) == 1
    summary = summaries[0]
    # All later-vs-earlier EVs (0.10, 0.20, 0.05) are positive and above epsilon.
    assert summary.all_pairs_clear_improvement_rate == pytest.approx(1.0)
    assert summary.adjacent_clear_improvement_rate == pytest.approx(1.0)
    assert summary.mean_later_vs_earlier_ev == pytest.approx((0.10 + 0.20 + 0.05) / 3)
    assert summary.worst_monotonicity_violation_ev == pytest.approx(0.05)


def test_monotonicity_summary_detects_regression():
    matrix = np.array(
        [
            [0.0, -0.05, 0.10],   # ckpt 0 vs ckpt 1, ckpt 0 vs ckpt 2
            [0.05, 0.0, -0.20],   # ckpt 1 vs ckpt 0, ckpt 1 vs ckpt 2
            [-0.10, 0.20, 0.0],   # ckpt 2 vs ckpt 0 NEGATIVE -> regression
        ]
    )
    summary = monotonicity_summary(
        {"0": matrix}, [100, 200, 300], equivalence_epsilon=1e-3
    )[0]
    assert summary.all_pairs_clear_regression_rate > 0
    assert summary.worst_monotonicity_violation_ev == pytest.approx(-0.10)


def test_strength_per_checkpoint_includes_correct_columns():
    matrix = np.array(
        [
            [0.0, -0.10, -0.20],
            [0.10, 0.0, -0.05],
            [0.20, 0.05, 0.0],
        ]
    )
    strength = strength_per_checkpoint({"0": matrix}, [100, 200, 300])
    by_ckpt = {row.checkpoint: row for row in strength}
    assert by_ckpt[100].mean_EV_vs_earlier_checkpoints != by_ckpt[100].mean_EV_vs_earlier_checkpoints  # NaN
    assert np.isnan(by_ckpt[100].mean_EV_vs_earlier_checkpoints)
    assert by_ckpt[300].mean_EV_vs_earlier_checkpoints == pytest.approx((0.20 + 0.05) / 2)
    assert by_ckpt[300].EV_vs_previous_checkpoint == pytest.approx(0.05)


def test_aggregate_strength_returns_one_row_per_checkpoint():
    matrix_a = np.array(
        [
            [0.0, -0.10],
            [0.10, 0.0],
        ]
    )
    matrix_b = np.array(
        [
            [0.0, -0.05],
            [0.05, 0.0],
        ]
    )
    strength = strength_per_checkpoint({"a": matrix_a, "b": matrix_b}, [100, 200])
    aggregated = aggregate_strength(strength)
    assert len(aggregated) == 2
    by_ckpt = {row["checkpoint"]: row for row in aggregated}
    assert by_ckpt[200]["EV_vs_previous_checkpoint_n"] == 2
    # Mean of 0.10 and 0.05 = 0.075.
    assert by_ckpt[200]["EV_vs_previous_checkpoint_mean"] == pytest.approx(0.075)
