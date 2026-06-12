"""Tests for the reservoir replay buffer."""

from __future__ import annotations

import random

import numpy as np
import pytest

from deep_cfr_poker.replay import AdvantageMemory, ReservoirBuffer, StrategyMemory


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ReservoirBuffer(0)
    with pytest.raises(ValueError):
        ReservoirBuffer(-3)


def test_buffer_fills_up_to_capacity_then_replaces():
    capacity = 10
    buf = ReservoirBuffer(capacity)
    for i in range(capacity):
        buf.add(i)
    assert len(buf) == capacity
    assert sorted(list(buf)) == list(range(capacity))

    # After capacity, len stays fixed but contents may change.
    for i in range(capacity, capacity * 5):
        buf.add(i)
    assert len(buf) == capacity
    assert buf.add_calls == capacity * 5


def test_uniform_sampling_property_over_many_runs():
    """Each of the first N elements should be retained ~k/N of the time."""
    capacity = 5
    stream_length = 50
    trials = 4000
    counts = np.zeros(stream_length, dtype=np.int64)

    for trial in range(trials):
        random.seed(trial)
        buf = ReservoirBuffer(capacity)
        for i in range(stream_length):
            buf.add(i)
        for kept in buf:
            counts[kept] += 1

    expected = capacity / stream_length * trials
    # Each element should be retained roughly capacity/stream_length of the time.
    # Allow a generous margin (5x standard deviations of a binomial(p)).
    p = capacity / stream_length
    sd = np.sqrt(trials * p * (1 - p))
    deviation = np.abs(counts - expected)
    assert np.all(deviation < 5 * sd), (
        f"counts {counts} deviate from expected {expected} by more than 5 sd ({sd:.2f})"
    )


def test_sample_raises_when_too_many_requested():
    buf = ReservoirBuffer(5)
    for i in range(3):
        buf.add(i)
    with pytest.raises(ValueError):
        buf.sample(4)


def test_sample_returns_distinct_elements():
    buf = ReservoirBuffer(20)
    for i in range(20):
        buf.add(i)
    samples = buf.sample(10)
    assert len(set(samples)) == 10


def test_state_dict_round_trip_preserves_contents_and_counters():
    buf = ReservoirBuffer(7)
    for i in range(20):
        buf.add(("info", i, np.array([i, i + 1], dtype=np.float32)))

    state = buf.state_dict()
    restored = ReservoirBuffer(7)
    restored.load_state_dict(state)

    assert restored.capacity == buf.capacity
    assert restored.add_calls == buf.add_calls
    assert len(restored) == len(buf)
    # Element identities (lengths and per-element shapes) should match.
    for original, copy in zip(list(buf), list(restored)):
        assert original[0] == copy[0]
        assert original[1] == copy[1]
        np.testing.assert_array_equal(original[2], copy[2])


def test_load_state_dict_rejects_overfull_payload():
    buf = ReservoirBuffer(3)
    bad_state = {"capacity": 3, "add_calls": 100, "data": [1, 2, 3, 4]}
    with pytest.raises(ValueError):
        buf.load_state_dict(bad_state)


def test_seeded_runs_are_deterministic():
    def collect():
        random.seed(2026)
        buf = ReservoirBuffer(5)
        for i in range(50):
            buf.add(i)
        return list(buf)

    assert collect() == collect()


def test_namedtuples_have_expected_fields():
    assert AdvantageMemory._fields == ("info_state", "iteration", "advantage")
    assert StrategyMemory._fields == ("info_state", "iteration", "strategy_action_probs")
