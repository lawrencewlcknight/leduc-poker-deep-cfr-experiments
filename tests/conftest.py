"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def leduc_game():
    """Returns an OpenSpiel Leduc poker game, skipping if pyspiel is missing."""
    pyspiel = pytest.importorskip("pyspiel")
    return pyspiel.load_game("leduc_poker")
