"""Smoke test for target processing on the composite architecture."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from deep_cfr_poker.experiment_utils import run_single_seed
from experiments.leduc_poker.deep_cfr_composite_target_processing_ablation.config import (
    TARGET_PROCESSING_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_composite_target_processing_ablation.run import (
    _augment_result,
    _variant_config,
    export_ablation_results,
)


SMOKE_CONFIG = {
    "experiment_name": "composite_target_processing_smoke",
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
    "target_processing": "none",
    "target_clip_value": 1.0,
    "target_standardize_epsilon": 1e-6,
    "ablation_variants": tuple(TARGET_PROCESSING_VARIANTS[:2]),
    "baseline_variant_id": "composite_raw_targets_baseline",
}


@pytest.mark.smoke
def test_composite_target_processing_ablation_writes_expected_artifacts(tmp_path):
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
    assert "composite_standardized_targets" in aggregate["by_variant_id"]

    summary_header = info["summary_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "advantage_network_type" in summary_header

    paired = json.loads(info["paired_difference_summary"].read_text(encoding="utf-8"))
    assert "composite_standardized_targets" in paired
    assert (
        "delta_final_exploitability_vs_baseline"
        in paired["composite_standardized_targets"]
    )
