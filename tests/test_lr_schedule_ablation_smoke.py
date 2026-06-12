"""Tiny smoke test for the learning-rate schedule ablation export path."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_lr_schedule_ablation.config import SCHEDULE_CONFIGS
from experiments.leduc_poker.deep_cfr_lr_schedule_ablation.run import (
    _augment_result,
    _schedule_config,
    export_ablation_results,
)
from deep_cfr_poker.experiment_utils import run_single_seed


SMOKE_CONFIG = {
    "experiment_name": "lr_schedule_smoke",
    "game_name": "leduc_poker",
    "num_iterations": 3,
    "num_traversals": 4,
    "evaluation_interval": 1,
    "policy_network_layers": (8, 8),
    "advantage_network_layers": (8, 8),
    "learning_rate": 0.003,
    "learning_rate_end": 0.0003,
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
    "schedule_configs": tuple(SCHEDULE_CONFIGS),
    "baseline_schedule": "constant_baseline_exp2",
}


@pytest.mark.smoke
def test_lr_schedule_ablation_writes_expected_artifacts(tmp_path):
    results = []
    for schedule in SMOKE_CONFIG["schedule_configs"]:
        schedule_config = _schedule_config(SMOKE_CONFIG, schedule)
        result = run_single_seed(1234, schedule_config, export_dir=tmp_path)
        results.append(
            _augment_result(
                result,
                schedule_config,
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
    assert "by_schedule" in aggregate
    assert "cosine_decay_to_10pct" in aggregate["by_schedule"]

    curve_header = info["curve_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "schedule" in curve_header
    assert "learning_rate_schedule" in curve_header
    assert "learning_rate" in curve_header

    paired = json.loads(info["paired_difference_summary"].read_text(encoding="utf-8"))
    assert "cosine_decay_to_10pct" in paired
    assert (
        "delta_final_exploitability_vs_baseline"
        in paired["cosine_decay_to_10pct"]
    )
