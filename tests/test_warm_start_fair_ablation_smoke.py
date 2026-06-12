"""Tiny smoke test for the fair warm-start ablation export path."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_warm_start_fair_ablation.run import (
    export_results,
    run_continuous_baseline,
    run_warm_start_arm,
)


SMOKE_CONFIG = {
    "experiment_name": "warm_start_smoke",
    "game_name": "leduc_poker",
    "total_iterations": 3,
    "warm_start_boundary": 1,
    "num_traversals": 4,
    "evaluation_interval": 1,
    "policy_network_layers": (8, 8),
    "advantage_network_layers": (8, 8),
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
    "arms": (
        {"arm": "baseline_continuous"},
        {"arm": "warm_start"},
    ),
}


@pytest.mark.smoke
def test_warm_start_fair_ablation_writes_expected_artifacts(tmp_path):
    baseline = run_continuous_baseline(1234, SMOKE_CONFIG)
    warm = run_warm_start_arm(1234, SMOKE_CONFIG, tmp_path)
    info = export_results([baseline, warm], tmp_path, SMOKE_CONFIG, [1234])

    for path in (
        info["summary_csv"],
        info["curve_csv"],
        info["paired_summary_csv"],
        info["paired_checkpoint_differences_csv"],
        info["aggregate_summary"],
        info["paired_aggregate_summary"],
        info["ablation_curves_npz"],
    ):
        assert path.exists()

    assert (tmp_path / "checkpoints" / "seed_1234_iter_1_full.pt").exists()

    metadata = json.loads((tmp_path / "experiment_metadata.json").read_text(encoding="utf-8"))
    assert metadata["experiment_config"]["warm_start_boundary"] == 1

    paired = info["paired_summary_csv"].read_text(encoding="utf-8")
    assert "delta_final_exploitability_warm_minus_baseline" in paired.splitlines()[0]

    curves = info["curve_csv"].read_text(encoding="utf-8")
    assert "policy_training_events" in curves.splitlines()[0]
    assert "warm_start" in curves
