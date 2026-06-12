"""Replay buffers and memory records for Deep CFR.

The reservoir buffer implements Algorithm R (Vitter, 1985): given a stream of
elements and a fixed capacity ``k``, every element seen so far is retained with
uniform probability ``k / n`` after ``n`` adds.

Both ``add`` and ``sample`` use Python's :mod:`random` module so that callers
seeding either ``random`` or ``numpy.random`` get a single, well-defined source
of randomness. This avoids the previous mix of ``np.random.randint`` and
``random.sample`` which made determinism harder to reason about.
"""

from __future__ import annotations

import collections
import random
from typing import Any, Iterable, Iterator, List

import numpy as np


# ``info_state`` is the OpenSpiel information-state tensor for the player whose
# regret we are storing; ``iteration`` is the Deep CFR iteration index at which
# the sample was collected (used for linear-CFR weighting); ``advantage`` is the
# per-action sampled regret vector with zeros for illegal actions.
AdvantageMemory = collections.namedtuple(
    "AdvantageMemory", ["info_state", "iteration", "advantage"]
)

# ``strategy_action_probs`` is the regret-matched policy at the info state for
# the non-traversing player (the average policy target).
StrategyMemory = collections.namedtuple(
    "StrategyMemory", ["info_state", "iteration", "strategy_action_probs"]
)


class ReservoirBuffer:
    """Uniform reservoir sampling over a stream of replay data."""

    def __init__(self, reservoir_buffer_capacity: int) -> None:
        capacity = int(reservoir_buffer_capacity)
        if capacity <= 0:
            raise ValueError(
                f"ReservoirBuffer capacity must be positive, got {capacity}"
            )
        self._reservoir_buffer_capacity = capacity
        self._data: List[Any] = []
        self._add_calls = 0

    @property
    def capacity(self) -> int:
        return self._reservoir_buffer_capacity

    @property
    def add_calls(self) -> int:
        return self._add_calls

    def add(self, element: Any) -> None:
        """Adds ``element`` to the buffer with reservoir-sampling semantics."""
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            # Algorithm R: replace a uniformly chosen slot with probability
            # capacity / (add_calls + 1).
            idx = random.randint(0, self._add_calls)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples: int) -> List[Any]:
        """Returns ``num_samples`` distinct elements drawn uniformly at random."""
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_samples > len(self._data):
            raise ValueError(
                f"Cannot sample {num_samples} elements from a buffer of size "
                f"{len(self._data)}"
            )
        return random.sample(self._data, num_samples)

    def clear(self) -> None:
        self._data = []
        self._add_calls = 0

    def state_dict(self) -> dict:
        """Returns a checkpointable representation of the buffer."""
        return {
            "capacity": int(self._reservoir_buffer_capacity),
            "add_calls": int(self._add_calls),
            "data": list(self._data),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restores buffer contents from :meth:`state_dict`."""
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError(
                f"Loaded buffer capacity must be positive, got {capacity}"
            )
        self._reservoir_buffer_capacity = capacity
        self._add_calls = int(state["add_calls"])
        self._data = list(state["data"])
        if len(self._data) > self._reservoir_buffer_capacity:
            raise ValueError(
                "Loaded buffer has more elements than its capacity: "
                f"{len(self._data)} > {self._reservoir_buffer_capacity}"
            )

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)
