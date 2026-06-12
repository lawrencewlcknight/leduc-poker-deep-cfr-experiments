"""Tiny smoke test for the policy-training-frequency ablation export path."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_policy_training_frequency_ablation.run import (
    _augment_result,
    _variant_config,
    export_ablation_results,
)
from deep_cfr_poker.experiment_utils import run_single_seed


SMOKE_CONFIG = {
    "experiment_name": "policy_frequency_smoke",
    "game_name": "leduc_poker",
    "num_iterations": 3,
    "num_traversals": 4,
    "evaluation_interval": 1,
    "policy_train_every_variants": (1, 2),
    "reference_policy_train_every": 1,
    "policy_network_layers": (8, 8),
    "advantage_network_layers": (8, 8),
    "learning_rate": 0.003,
    "batch_size_advantage": 2,
    "batch_size_strategy": 2,
    "memory_capacity": 256,
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 1,
    "advantage_network_train_steps": 1,
    "compute_exploitability": True,
    "exploitability_threshold": 0.5,
}


@pytest.mark.smoke
def test_policy_training_frequency_ablation_writes_expected_artifacts(tmp_path):
    results = []
    for variant in SMOKE_CONFIG["policy_train_every_variants"]:
        variant_config = _variant_config(SMOKE_CONFIG, variant)
        result = run_single_seed(1234, variant_config, export_dir=tmp_path)
        results.append(
            _augment_result(
                result,
                variant,
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
    ):
        assert path is not None
        assert path.exists()

    aggregate = json.loads(info["aggregate_summary"].read_text(encoding="utf-8"))
    assert "by_policy_network_train_every" in aggregate
    assert "1" in aggregate["by_policy_network_train_every"]

    curve_header = info["curve_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "policy_network_train_every" in curve_header
