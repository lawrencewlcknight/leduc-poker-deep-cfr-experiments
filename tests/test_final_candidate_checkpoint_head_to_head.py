"""Tests for the final-candidate checkpoint head-to-head experiment."""

from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.analyse import (
    run_analysis,
)
from experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.config import (
    CHECKPOINT_SCHEDULE,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
)
from experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.statistics import (
    build_inference_tables,
    exact_one_sided_sign_flip_p,
)
from experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.train import (
    run_training,
    validate_config,
)


def test_default_config_is_the_five_seed_final_candidate():
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert CHECKPOINT_SCHEDULE == (210, 420, 630, 840, 1050)
    assert DEFAULT_CONFIG["num_iterations"] == 1050
    assert DEFAULT_CONFIG["num_traversals"] == 320
    assert DEFAULT_CONFIG["policy_network_train_every"] == 10
    assert DEFAULT_CONFIG["batch_size_advantage"] == 2048
    assert DEFAULT_CONFIG["memory_capacity"] == int(5e6)
    assert DEFAULT_CONFIG["learning_rate"] == pytest.approx(0.004)
    assert DEFAULT_CONFIG["target_processing"] == "standardize"
    assert DEFAULT_CONFIG["advantage_replay_sampling"] == "uniform"
    assert DEFAULT_CONFIG["average_strategy_weighting"] == "uniform"
    assert DEFAULT_CONFIG["temporal_x_axis"] == "nodes_touched"


def test_checkpoint_schedule_must_align_with_policy_fits():
    invalid = deepcopy(DEFAULT_CONFIG)
    invalid["checkpoint_schedule"] = (211, 420, 630, 840, 1050)
    with pytest.raises(ValueError, match="average-policy training"):
        validate_config(invalid)


def test_exact_sign_flip_test_and_seed_level_tables():
    assert exact_one_sided_sign_flip_p([1, 1, 1, 1, 1]) == pytest.approx(1 / 32)
    rows = []
    schedule = (10, 20, 30)
    for seed, scale in (("1", 1.0), ("2", 2.0)):
        for later, earlier, ev in ((20, 10, 0.01), (30, 10, 0.03), (30, 20, 0.02)):
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_a": later,
                    "checkpoint_b": earlier,
                    "A_EV_seat_averaged": ev * scale,
                }
            )
    seed_rows, summary_rows, pair_rows = build_inference_tables(rows, schedule)
    assert len(seed_rows) == 2
    assert seed_rows[0]["num_later_vs_earlier_pairs"] == 3
    assert summary_rows[0]["mean_ev"] == pytest.approx(0.03)
    assert summary_rows[1]["mean_ev"] == pytest.approx(0.0225)
    assert summary_rows[2]["mean_ev"] == pytest.approx(0.045)
    assert len(pair_rows) == 3
    assert all("holm_adjusted_p" in row for row in pair_rows)


@pytest.mark.smoke
def test_train_and_analyse_writes_exact_node_based_outputs(tmp_path):
    config = deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "experiment_name": "final_candidate_checkpoint_head_to_head_smoke",
            "num_iterations": 10,
            "checkpoint_schedule": (2, 4, 6, 8, 10),
            "num_traversals": 2,
            "evaluation_interval": 2,
            "policy_network_train_every": 2,
            "policy_network_layers": (8, 8),
            "advantage_network_layers": (8, 8),
            "batch_size_advantage": 2,
            "batch_size_strategy": 2,
            "memory_capacity": 256,
            "policy_network_train_steps": 1,
            "advantage_network_train_steps": 1,
        }
    )
    seeds = [1234, 2025]
    outcome = run_training(config=config, seeds=seeds, run_dir=tmp_path)
    assert not outcome["failed"]
    assert len(outcome["metrics_rows"]) == 10
    assert len(list((tmp_path / "snapshots").glob("*.pt"))) == 10

    outputs = run_analysis(config=config, run_dir=tmp_path)
    for key in (
        "head_to_head_pairwise",
        "head_to_head_primary_effect_by_seed",
        "head_to_head_inference_summary",
        "head_to_head_pairwise_inference",
        "aggregate_summary",
    ):
        assert outputs[key].exists()

    for filename in (
        "head_to_head_mean_matrix.png",
        "head_to_head_later_vs_earlier.png",
        "head_to_head_strength_vs_earlier_by_nodes.png",
        "head_to_head_strength_vs_previous_by_nodes.png",
        "exploitability_by_nodes.png",
        "average_policy_value_by_nodes.png",
        "head_to_head_primary_effect_by_seed.png",
    ):
        assert (tmp_path / filename).exists(), filename
    assert not (tmp_path / "exploitability_by_checkpoint.png").exists()
