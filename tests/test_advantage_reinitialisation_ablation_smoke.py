"""Tiny smoke test for the advantage-reinitialisation ablation export path."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_advantage_reinitialisation_ablation.config import (
    ABLATION_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_advantage_reinitialisation_ablation.run import (
    _augment_result,
    _variant_config,
    export_ablation_results,
)
from deep_cfr_poker.experiment_utils import run_single_seed


SMOKE_CONFIG = {
    "experiment_name": "advantage_reinit_smoke",
    "game_name": "leduc_poker",
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
    "policy_training_mode": "intermittent",
    "final_policy_network_train_steps": None,
    "compute_exploitability": True,
    "exploitability_threshold": 0.5,
    "ablation_variants": tuple(ABLATION_VARIANTS),
    "reference_variant_id": "reinit_false_warm_started_advantage",
    "comparison_variant_id": "reinit_true_reset_advantage",
}


@pytest.mark.smoke
def test_advantage_reinitialisation_ablation_writes_expected_artifacts(tmp_path):
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
    assert "reinit_true_reset_advantage" in aggregate["by_variant_id"]

    curve_header = info["curve_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "reinitialize_advantage_networks" in curve_header
    assert "variant_id" in curve_header

    paired = json.loads(info["paired_difference_summary"].read_text(encoding="utf-8"))
    assert "delta_final_exploitability_true_minus_false" in paired
