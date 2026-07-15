"""Smoke tests for targeted composite-baseline HP ablations."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from deep_cfr_poker.experiment_utils import run_single_seed
from experiments.leduc_poker.deep_cfr_composite_advantage_fitting_ablation.config import (
    ADVANTAGE_FITTING_VARIANTS,
    DEFAULT_CONFIG as ADVANTAGE_FITTING_CONFIG,
    DEFAULT_SEEDS as ADVANTAGE_FITTING_SEEDS,
)
from experiments.leduc_poker.deep_cfr_composite_advantage_fitting_ablation.run import (
    build_config as build_advantage_fitting_config,
)
from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    BASELINE_VARIANT_ID,
    _augment_result,
    _variant_config,
    export_ablation_results,
)
from experiments.leduc_poker.deep_cfr_composite_learning_rate_ablation.config import (
    DEFAULT_CONFIG as LEARNING_RATE_CONFIG,
    DEFAULT_SEEDS as LEARNING_RATE_SEEDS,
    LEARNING_RATE_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_composite_learning_rate_ablation.run import (
    build_config as build_learning_rate_config,
)
from experiments.leduc_poker.deep_cfr_composite_policy_extraction_ablation.config import (
    DEFAULT_CONFIG as POLICY_EXTRACTION_CONFIG,
    DEFAULT_SEEDS as POLICY_EXTRACTION_SEEDS,
    POLICY_EXTRACTION_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_composite_policy_extraction_ablation.run import (
    build_config as build_policy_extraction_config,
)
from experiments.leduc_poker.deep_cfr_composite_replay_memory_ablation.config import (
    DEFAULT_CONFIG as REPLAY_MEMORY_CONFIG,
    DEFAULT_SEEDS as REPLAY_MEMORY_SEEDS,
    REPLAY_MEMORY_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_composite_replay_memory_ablation.run import (
    build_config as build_replay_memory_config,
)


def _default_args(**overrides):
    values = {
        "variant_ids": None,
        "experiment_name": None,
        "iterations": None,
        "traversals": None,
        "evaluation_interval": None,
        "policy_network_layers": None,
        "advantage_network_layers": None,
        "learning_rate": None,
        "batch_size_advantage": None,
        "batch_size_strategy": None,
        "memory_capacity": None,
        "reinitialize_advantage_networks": None,
        "policy_network_train_steps": None,
        "advantage_network_train_steps": None,
        "policy_network_train_every": None,
        "final_policy_network_train_steps": None,
        "compute_exploitability": None,
        "target_standardize_epsilon": None,
        "priority_alpha": None,
        "priority_epsilon": None,
        "baseline_variant_id": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_composite_hp_ablation_default_variant_sets():
    cases = (
        (
            build_policy_extraction_config,
            [
                "composite_best_baseline",
                "policy_train_every_10",
                "policy_train_every_50",
                "policy_steps_400",
                "strategy_batch_2048",
            ],
        ),
        (
            build_advantage_fitting_config,
            [
                "composite_best_baseline",
                "advantage_steps_100",
                "advantage_steps_400",
                "advantage_batch_512",
                "advantage_batch_2048",
                "advantage_batch_2048_steps_400",
            ],
        ),
        (
            build_replay_memory_config,
            [
                "composite_best_baseline",
                "memory_capacity_1m",
                "memory_capacity_2m",
                "memory_capacity_5m",
            ],
        ),
        (
            build_learning_rate_config,
            [
                "composite_best_baseline",
                "learning_rate_0_001",
                "learning_rate_0_0015",
                "learning_rate_0_002",
                "learning_rate_0_004",
            ],
        ),
    )
    for build_config, expected_ids in cases:
        config = build_config(_default_args())
        assert config["target_processing"] == "standardize"
        assert config["average_strategy_weighting"] == "uniform"
        assert config["advantage_replay_sampling"] == "uniform"
        assert config["baseline_variant_id"] == BASELINE_VARIANT_ID
        assert [variant["variant_id"] for variant in config["ablation_variants"]] == (
            expected_ids
        )


def test_composite_hp_ablation_default_seed_sets():
    assert POLICY_EXTRACTION_SEEDS == [1234, 2025, 31415]
    assert ADVANTAGE_FITTING_SEEDS == [1234, 2025, 31415]
    assert LEARNING_RATE_SEEDS == [1234, 2025, 31415]
    assert REPLAY_MEMORY_SEEDS == [1234, 2025, 31415, 27182, 16180]


SMOKE_CASES = (
    (POLICY_EXTRACTION_CONFIG, (POLICY_EXTRACTION_VARIANTS[0], POLICY_EXTRACTION_VARIANTS[1])),
    (ADVANTAGE_FITTING_CONFIG, (ADVANTAGE_FITTING_VARIANTS[0], ADVANTAGE_FITTING_VARIANTS[3])),
    (REPLAY_MEMORY_CONFIG, (REPLAY_MEMORY_VARIANTS[0], REPLAY_MEMORY_VARIANTS[1])),
    (LEARNING_RATE_CONFIG, (LEARNING_RATE_VARIANTS[0], LEARNING_RATE_VARIANTS[3])),
)


@pytest.mark.smoke
@pytest.mark.parametrize("default_config,variants", SMOKE_CASES)
def test_composite_hp_ablation_writes_expected_artifacts(
    tmp_path, default_config, variants
):
    smoke_config = deepcopy(default_config)
    smoke_config.update(
        {
            "experiment_name": "composite_hp_ablation_smoke",
            "num_iterations": 3,
            "num_traversals": 4,
            "evaluation_interval": 1,
            "policy_network_layers": (8, 8),
            "advantage_network_layers": (8, 8),
            "learning_rate": 0.003,
            "batch_size_advantage": 2,
            "batch_size_strategy": 2,
            "memory_capacity": 256,
            "policy_network_train_steps": 1,
            "advantage_network_train_steps": 1,
            "policy_network_train_every": 1,
            "compute_exploitability": True,
            "exploitability_threshold": 0.5,
            "ablation_variants": variants,
            "baseline_variant_id": BASELINE_VARIANT_ID,
        }
    )

    results = []
    for variant in smoke_config["ablation_variants"]:
        variant_config = _variant_config(smoke_config, variant)
        result = run_single_seed(1234, variant_config, export_dir=tmp_path)
        results.append(
            _augment_result(
                result,
                variant_config,
                final_window=2,
                exploitability_threshold=smoke_config["exploitability_threshold"],
            )
        )

    info = export_ablation_results(
        results,
        tmp_path,
        smoke_config,
        [1234],
        experiment_note="Composite HP smoke test.",
    )

    for path in (
        info["summary_csv"],
        info["curve_csv"],
        info["aggregate_summary"],
        info["metadata"],
        info["ablation_curves_npz"],
        info["paired_differences_csv"],
        info["paired_difference_summary"],
    ):
        assert path is not None
        assert path.exists()

    aggregate = json.loads(info["aggregate_summary"].read_text(encoding="utf-8"))
    assert "by_variant_id" in aggregate
    assert str(variants[1]["variant_id"]) in aggregate["by_variant_id"]

    summary_header = info["summary_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "hp_family" in summary_header
    assert "hp_value" in summary_header
    assert "average_strategy_weighting" in summary_header

    paired = json.loads(info["paired_difference_summary"].read_text(encoding="utf-8"))
    assert str(variants[1]["variant_id"]) in paired
    assert "delta_final_exploitability_vs_baseline" in paired[
        str(variants[1]["variant_id"])
    ]
