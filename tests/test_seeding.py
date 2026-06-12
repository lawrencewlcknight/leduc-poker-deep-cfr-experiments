"""Tests for the seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deep_cfr_poker.seeding import set_seed


def test_set_seed_seeds_all_global_rngs():
    set_seed(1234)
    py_value = random.random()
    np_value = float(np.random.random())
    torch_value = float(torch.rand(1).item())

    set_seed(1234)
    assert random.random() == py_value
    assert float(np.random.random()) == np_value
    assert float(torch.rand(1).item()) == torch_value


def test_set_seed_sets_pythonhashseed():
    set_seed(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_set_seed_pins_cudnn_determinism():
    set_seed(7)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_different_seeds_produce_different_streams():
    set_seed(1)
    a = float(np.random.random())
    set_seed(2)
    b = float(np.random.random())
    assert a != b
