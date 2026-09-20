"""End-to-end smoke coverage for Experiment 28."""

from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.single_deep_cfr_comparison.config import (
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
)
from experiments.leduc_poker.single_deep_cfr_comparison.recover_failed_run import (
    export_recovery,
    recover_seed,
)
from experiments.leduc_poker.single_deep_cfr_comparison.run import (
    _export,
    _load_worker_results,
    _run_seed,
)


def test_default_config_uses_final_candidate_and_both_sd_cfr_weightings():
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert DEFAULT_CONFIG["num_iterations"] == 1050
    assert DEFAULT_CONFIG["num_traversals"] == 320
    assert DEFAULT_CONFIG["policy_network_train_every"] == 10
    assert DEFAULT_CONFIG["batch_size_advantage"] == 2048
    assert DEFAULT_CONFIG["memory_capacity"] == int(5e6)
    assert DEFAULT_CONFIG["learning_rate"] == pytest.approx(0.004)
    assert DEFAULT_CONFIG["advantage_network_type"] == (
        "residual_layer_norm_centered_advantage_mlp"
    )
    assert DEFAULT_CONFIG["target_processing"] == "standardize"
    assert DEFAULT_CONFIG["advantage_replay_sampling"] == "uniform"
    assert DEFAULT_CONFIG["average_strategy_weighting"] == "uniform"
    assert DEFAULT_CONFIG["sd_cfr_weightings"] == ("uniform", "linear")
    assert DEFAULT_CONFIG["primary_sd_cfr_weighting"] == "uniform"


@pytest.mark.smoke
def test_paired_sd_cfr_smoke_writes_playable_and_analysis_outputs(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "experiment_name": "single_deep_cfr_smoke",
            "num_iterations": 3,
            "num_traversals": 4,
            "evaluation_interval": 1,
            "policy_network_train_every": 1,
            "policy_network_layers": (8, 8),
            "advantage_network_layers": (8, 8),
            "memory_capacity": 256,
            "batch_size_advantage": 2,
            "batch_size_strategy": 2,
            "policy_network_train_steps": 1,
            "advantage_network_train_steps": 1,
        }
    )
    result = _run_seed(1234, config, tmp_path)
    _export([result], [], config, [1234], tmp_path)

    assert result["summary"]["num_archived_networks_per_player"] == 3
    assert len(result["curves"]) == 3
    assert "sd_cfr_uniform_final_exploitability" in result["summary"]
    assert "sd_cfr_linear_final_exploitability" in result["summary"]
    assert "sd_cfr_uniform_exploitability" in result["curves"][0]
    assert "sd_cfr_linear_exploitability" in result["curves"][0]
    assert (tmp_path / "sd_cfr_archives/seed_1234_sd_cfr_archive.pt").exists()
    assert (tmp_path / "policy_snapshots/seed_1234_deep_cfr_policy.pt").exists()
    assert (tmp_path / "seed_results/seed_1234_checkpoint_curves.csv").exists()
    assert (tmp_path / "seed_results/seed_1234_result.json").exists()
    restored = _load_worker_results(tmp_path, [1234])
    assert restored[0]["seed"] == 1234
    assert restored[0]["summary"] == result["summary"]
    assert restored[0]["curves"] == result["curves"]
    recovered = recover_seed(
        1234,
        tmp_path / "sd_cfr_archives/seed_1234_sd_cfr_archive.pt",
        tmp_path / "policy_snapshots/seed_1234_deep_cfr_policy.pt",
    )
    assert recovered["summary"]["deep_cfr_final_exploitability"] == pytest.approx(
        result["summary"]["deep_cfr_final_exploitability"]
    )
    assert recovered["summary"][
        "sd_cfr_uniform_final_exploitability"
    ] == pytest.approx(result["summary"]["sd_cfr_uniform_final_exploitability"])
    recovery_dir = tmp_path / "recovered_analysis"
    export_recovery([recovered], tmp_path, recovery_dir)
    assert (recovery_dir / "recovered_seed_summary.csv").exists()
    assert (recovery_dir / "recovered_sd_cfr_curves.csv").exists()
    assert (recovery_dir / "recovery_manifest.json").exists()
    for filename in (
        "seed_summary.csv",
        "checkpoint_curves.csv",
        "aggregate_summary.json",
        "paired_difference_summary.json",
        "experiment_metadata.json",
        "exploitability_by_iteration.png",
        "exploitability_by_nodes.png",
        "exploitability_by_training_time.png",
        "final_exploitability_paired.png",
    ):
        assert (tmp_path / filename).exists(), filename
