"""Contract and end-to-end smoke coverage for Experiment 30."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.single_deep_cfr_uniform_36h_trajectory.config import (
    TRAINING_CONFIG as EXPERIMENT_29_CONFIG,
)
from experiments.leduc_poker.single_deep_cfr_paper_aligned_36h_trajectory.cloud import (
    aggregate_workers,
    run_evaluation_worker,
    run_training_worker,
)
from experiments.leduc_poker.single_deep_cfr_paper_aligned_36h_trajectory.config import (
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    TRAINING_CONFIG,
    build_config,
)
from experiments.leduc_poker.single_deep_cfr_paper_aligned_36h_trajectory.run import (
    read_csv,
    read_json,
)


def test_experiment_30_changes_only_the_registered_paper_aligned_fields():
    expected_changes = {
        "experiment_number",
        "experiment_name",
        "algorithm",
        "source_configuration",
        "num_traversals",
        "advantage_network_train_steps",
        "learning_rate",
        "reset_advantage_optimizer_each_iteration",
    }
    all_fields = set(EXPERIMENT_29_CONFIG) | set(TRAINING_CONFIG)
    actual_changes = {
        field
        for field in all_fields
        if EXPERIMENT_29_CONFIG.get(field) != TRAINING_CONFIG.get(field)
    }
    assert actual_changes == expected_changes

    for field in (
        "advantage_network_layers",
        "advantage_network_type",
        "learning_rate_schedule",
        "batch_size_advantage",
        "memory_capacity",
        "reinitialize_advantage_networks",
        "target_processing",
        "advantage_replay_sampling",
    ):
        assert TRAINING_CONFIG[field] == EXPERIMENT_29_CONFIG[field], field
    assert TRAINING_CONFIG["num_traversals"] == 1_500
    assert TRAINING_CONFIG["advantage_network_train_steps"] == 750
    assert TRAINING_CONFIG["learning_rate"] == 0.001
    assert TRAINING_CONFIG["reset_advantage_optimizer_each_iteration"] is True
    assert PRODUCTION_SEEDS == (1234, 2025, 31415, 27182, 16180)
    assert TRAINING_CONFIG["training_time_budget_seconds"] == 36 * 3600
    assert TRAINING_CONFIG["checkpoint_interval_seconds"] == 30 * 60
    assert TRAINING_CONFIG["primary_sd_cfr_weighting"] == "uniform"
    assert TRAINING_CONFIG["sd_cfr_weightings"] == ("uniform", "linear")
    assert TRAINING_CONFIG["policy_training_mode"] == "disabled"
    assert TRAINING_CONFIG["collect_strategy_replay"] is False
    assert TRAINING_CONFIG["compute_exploitability"] is False


def test_smoke_config_keeps_scientific_semantics_but_reduces_cost():
    config = build_config(smoke=True)
    assert config["seeds"] == list(SMOKE_SEEDS)
    assert config["training_time_budget_seconds"] > 0
    assert config["policy_training_mode"] == "disabled"
    assert config["collect_strategy_replay"] is False
    assert config["primary_sd_cfr_weighting"] == "uniform"
    assert config["reset_advantage_optimizer_each_iteration"] is True


@pytest.mark.smoke
def test_experiment_30_end_to_end_smoke(tmp_path):
    training = run_training_worker(
        task_index=0,
        seeds=SMOKE_SEEDS,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    assert training["status"] == "complete"
    training_task = tmp_path / "training" / "task_000_seed_1234"
    training_result = read_json(training_task / "seed_1234/training_result.json")
    assert training_result["num_archived_networks_per_player"] >= 1
    assert training_result["active_training_seconds"] >= 0.01
    assert training_result["num_time_checkpoints"] >= 1
    assert (training_task / training_result["archive_path"]).is_file()

    evaluation = run_evaluation_worker(
        task_index=0,
        seeds=SMOKE_SEEDS,
        training_root=tmp_path,
        output_root=tmp_path,
        smoke=True,
        resume=False,
    )
    assert evaluation["status"] == "complete"
    evaluation_task = tmp_path / "evaluation" / "task_000_seed_1234"
    evaluation_result = read_json(evaluation_task / "seed_1234/evaluation_result.json")
    rows = read_csv(evaluation_task / evaluation_result["metrics_path"])
    assert rows
    assert sum(row["is_final_endpoint"].lower() == "true" for row in rows) == 1
    assert all("sd_cfr_uniform_exploitability" in row for row in rows)
    assert all("sd_cfr_linear_exploitability" in row for row in rows)

    aggregate = aggregate_workers(
        evaluation_root=tmp_path,
        seeds=SMOKE_SEEDS,
        output_dir=tmp_path / "analysis",
        smoke=True,
    )
    assert aggregate["status"] == "complete"
    assert "final" in {
        row["endpoint_id"]
        for row in read_csv(tmp_path / "analysis/endpoint_aggregate_summary.csv")
    }
    for filename in (
        "aggregate_summary.json",
        "checkpoint_seed_metrics.csv",
        "trajectory_by_training_time_summary.csv",
        "trajectory_by_nodes_summary.csv",
        "endpoint_seed_metrics.csv",
        "endpoint_aggregate_summary.csv",
        "seed_summary.csv",
        "exploitability_by_training_time.png",
        "exploitability_by_nodes.png",
        "exploitability_by_iteration.png",
        "uniform_minus_linear_by_training_time.png",
    ):
        assert (tmp_path / "analysis" / filename).is_file(), filename
