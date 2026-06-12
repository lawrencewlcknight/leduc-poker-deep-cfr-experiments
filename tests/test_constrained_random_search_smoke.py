"""Tiny smoke test for the constrained random-search experiment."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_constrained_random_search.config import (
    DEFAULT_CONFIG,
)
from experiments.leduc_poker.deep_cfr_constrained_random_search.run import (
    _apply_quick_test_defaults,
    run_staged_random_search,
)


@pytest.mark.smoke
def test_constrained_random_search_quick_mode_writes_expected_artifacts(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    _apply_quick_test_defaults(config)

    info = run_staged_random_search(config=config, run_dir=tmp_path)

    for path in (
        info["summary_csv"],
        info["curve_csv"],
        info["aggregate_summary"],
        info["metadata"],
        info["screening_seed_summary"],
        info["confirmation_seed_summary"],
        info["screening_config_summary"],
        info["confirmation_config_summary"],
        info["search_configurations"],
        info["paired_differences_csv"],
        info["paired_difference_summary"],
        info["search_curves_npz"],
    ):
        assert path is not None
        assert path.exists()

    aggregate = json.loads(info["aggregate_summary"].read_text(encoding="utf-8"))
    assert "screening" in aggregate
    assert "confirmation" in aggregate
    assert "experiment2_baseline" in aggregate["confirmation"]["by_config_label"]

    curve_header = info["curve_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "stage" in curve_header
    assert "config_label" in curve_header

    config_header = info["search_configurations"].read_text(encoding="utf-8").splitlines()[0]
    assert "selected_for_confirmation" in config_header

    traces = list((tmp_path / "traces").glob("*.json"))
    assert len(traces) == 4
