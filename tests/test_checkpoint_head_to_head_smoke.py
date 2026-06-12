"""End-to-end smoke test for the checkpoint head-to-head experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.analyse import run_analysis
from experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.config import DEFAULT_CONFIG
from experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.train import run_training


SMOKE_CONFIG = {
    **DEFAULT_CONFIG,
    "experiment_name": "checkpoint_head_to_head_smoke",
    "checkpoint_schedule": (3, 6),
    "policy_train_every_by_target": {3: 1, 6: 1},
    "num_traversals": 4,
    "policy_network_layers": (8, 8),
    "advantage_network_layers": (8, 8),
    "memory_capacity": 256,
    "batch_size_advantage": 2,
    "batch_size_strategy": 2,
    "policy_network_train_steps": 1,
    "advantage_network_train_steps": 1,
    "compute_exploitability": True,
    "run_monte_carlo_validation": False,
}


@pytest.mark.smoke
def test_train_then_analyse_creates_expected_artifacts(tmp_path):
    seeds = [1234, 2025]
    train_outcome = run_training(config=SMOKE_CONFIG, seeds=seeds, run_dir=tmp_path)
    assert train_outcome["metrics_rows"], "training produced no metrics"

    snapshots_dir = tmp_path / "snapshots"
    snapshot_files = list(snapshots_dir.glob("*.pt"))
    expected_count = len(seeds) * len(SMOKE_CONFIG["checkpoint_schedule"])
    assert len(snapshot_files) == expected_count

    analysis_outputs = run_analysis(
        config=SMOKE_CONFIG, run_dir=tmp_path, snapshots_dir=snapshots_dir
    )

    expected_csvs = (
        "checkpoint_inventory",
        "checkpoint_exploitability_metrics",
        "head_to_head_pairwise",
        "head_to_head_mean_matrix",
        "head_to_head_seed_win_fraction_matrix",
        "head_to_head_monotonicity_by_seed",
        "head_to_head_strength_by_checkpoint",
        "head_to_head_strength_aggregate",
        "best_checkpoint_summary",
    )
    for key in expected_csvs:
        path = analysis_outputs[key]
        assert Path(path).exists(), f"missing expected output: {path}"

    # Heatmaps are written too.
    for png_name in (
        "head_to_head_mean_matrix.png",
        "head_to_head_later_vs_earlier.png",
        "head_to_head_seed_win_fraction.png",
        "head_to_head_strength_vs_earlier.png",
        "exploitability_by_checkpoint.png",
        "average_policy_value_by_checkpoint.png",
        "strength_vs_average_policy_value.png",
    ):
        assert (tmp_path / png_name).exists(), png_name


@pytest.mark.smoke
def test_analyse_phase_can_run_against_existing_snapshots(tmp_path):
    seeds = [1234, 2025]
    run_training(config=SMOKE_CONFIG, seeds=seeds, run_dir=tmp_path)

    # Re-running analysis only should be idempotent.
    snapshots_dir = tmp_path / "snapshots"
    first = run_analysis(config=SMOKE_CONFIG, run_dir=tmp_path, snapshots_dir=snapshots_dir)
    second = run_analysis(config=SMOKE_CONFIG, run_dir=tmp_path, snapshots_dir=snapshots_dir)
    # Output paths should match across runs.
    for key in first:
        if isinstance(first[key], (str, Path)) and first[key] is not None:
            assert str(first[key]) == str(second[key])
