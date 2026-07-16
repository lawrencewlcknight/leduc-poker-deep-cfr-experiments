"""Tests for pure Deep CFR parallel-orchestration helpers."""

from __future__ import annotations

import pytest

from deep_cfr_poker.parallel_utils import (
    WORKER_SEED_STRIDE,
    equivalence_summary,
    partition_total,
    worker_seed,
)


def test_partition_total_splits_exactly_with_small_imbalance():
    assert partition_total(10, 3) == [4, 3, 3]
    assert partition_total(2, 4) == [1, 1, 0, 0]
    assert sum(partition_total(321, 7)) == 321
    assert max(partition_total(321, 7)) - min(partition_total(321, 7)) <= 1


def test_worker_seed_is_deterministic_and_distinct():
    assert worker_seed(1234, 0) == 1234 + WORKER_SEED_STRIDE
    assert worker_seed(1234, 1) != worker_seed(1234, 0)
    with pytest.raises(ValueError):
        worker_seed(1234, -1)


def test_equivalence_summary_reports_margin_membership():
    summary = equivalence_summary([0.01, -0.02, 0.0], margin=0.05)
    assert summary["n"] == 3
    assert summary["all_seed_deltas_within_margin"] is True
    assert summary["margin"] == 0.05

    wide = equivalence_summary([0.1], margin=0.05)
    assert wide["all_seed_deltas_within_margin"] is False
