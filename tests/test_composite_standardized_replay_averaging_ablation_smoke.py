"""Smoke test for averaging on the standardised composite baseline."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from deep_cfr_poker.experiment_utils import run_single_seed
from experiments.leduc_poker.deep_cfr_composite_standardized_replay_averaging_ablation.config import (
    REPLAY_AVERAGING_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_composite_standardized_replay_averaging_ablation.run import (
    _augment_result,
    _variant_config,
    build_config,
    export_ablation_results,
)


SMOKE_CONFIG = {
    "experiment_name": "composite_standardized_replay_averaging_smoke",
    "game_name": "leduc_poker",
    "num_iterations": 3,
    "num_traversals": 4,
    "evaluation_interval": 1,
    "policy_network_layers": (8, 8),
    "advantage_network_layers": (8, 8),
    "policy_network_type": "mlp",
    "advantage_network_type": "residual_layer_norm_centered_advantage_mlp",
    "learning_rate": 0.003,
    "batch_size_advantage": 2,
    "batch_size_strategy": 2,
    "memory_capacity": 256,
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 1,
    "advantage_network_train_steps": 1,
    "policy_network_train_every": 1,
    "policy_training_mode": "intermittent",
    "final_policy_network_train_steps": None,
    "compute_exploitability": True,
    "exploitability_threshold": 0.5,
    "target_processing": "standardize",
    "target_clip_value": 1.0,
    "target_standardize_epsilon": 1e-6,
    "advantage_replay_sampling": "uniform",
    "average_strategy_weighting": "linear",
    "priority_alpha": 1.0,
    "priority_epsilon": 1e-6,
    "ablation_variants": (
        REPLAY_AVERAGING_VARIANTS[0],
        REPLAY_AVERAGING_VARIANTS[2],
    ),
    "baseline_variant_id": "composite_std_uniform_replay_linear_avg_baseline",
}


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
        "compute_exploitability": None,
        "target_standardize_epsilon": None,
        "priority_alpha": None,
        "priority_epsilon": None,
        "baseline_variant_id": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_composite_standardized_replay_averaging_default_variants():
    config = build_config(_default_args())

    assert config["target_processing"] == "standardize"
    assert [variant["variant_id"] for variant in config["ablation_variants"]] == [
        "composite_std_uniform_replay_linear_avg_baseline",
        "composite_std_uniform_replay_uniform_avg",
    ]


@pytest.mark.smoke
def test_composite_standardized_replay_averaging_writes_expected_artifacts(tmp_path):
    results = []
    for variant in SMOKE_CONFIG["ablation_variants"]:
        variant_config = _variant_config(SMOKE_CONFIG, variant)
        result = run_single_seed(1234, variant_config, export_dir=tmp_path)
        results.append(
            _augment_result(
                result,
                variant_config,
                final_window=2,
                exploitability_threshold=SMOKE_CONFIG["exploitability_threshold"],
            )
        )

    info = export_ablation_results(results, tmp_path, SMOKE_CONFIG, [1234])

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
    assert "composite_std_uniform_replay_uniform_avg" in aggregate["by_variant_id"]

    summary_header = info["summary_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "target_processing" in summary_header
    assert "average_strategy_weighting" in summary_header

    paired = json.loads(info["paired_difference_summary"].read_text(encoding="utf-8"))
    assert "composite_std_uniform_replay_uniform_avg" in paired
    assert "delta_final_exploitability_vs_baseline" in paired[
        "composite_std_uniform_replay_uniform_avg"
    ]
