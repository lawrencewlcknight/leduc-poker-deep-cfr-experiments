"""Tiny smoke test for the final-only policy-training ablation export path."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_final_only_policy_training_ablation.config import (
    POLICY_TRAINING_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_final_only_policy_training_ablation.run import (
    _augment_result,
    _variant_config,
    export_ablation_results,
)
from deep_cfr_poker.experiment_utils import run_single_seed


SMOKE_CONFIG = {
    "experiment_name": "final_only_policy_smoke",
    "game_name": "leduc_poker",
    "num_iterations": 3,
    "num_traversals": 4,
    "evaluation_interval": 1,
    "policy_training_variants": tuple(POLICY_TRAINING_VARIANTS[:2]),
    "reference_variant_id": "intermittent_every_25",
    "policy_network_layers": (8, 8),
    "advantage_network_layers": (8, 8),
    "learning_rate": 0.003,
    "batch_size_advantage": 2,
    "batch_size_strategy": 2,
    "memory_capacity": 256,
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 1,
    "policy_network_train_every": 1,
    "final_policy_network_train_steps": None,
    "advantage_network_train_steps": 1,
    "compute_exploitability": True,
    "exploitability_threshold": 0.5,
}


@pytest.mark.smoke
def test_final_only_policy_training_ablation_writes_expected_artifacts(tmp_path):
    variants = list(SMOKE_CONFIG["policy_training_variants"])
    results = []
    for variant in variants:
        variant = dict(variant)
        variant["policy_network_train_every"] = 1
        variant["policy_network_train_steps"] = 1
        if variant["policy_training_mode"] == "final_only":
            variant["final_policy_network_train_steps"] = 1
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
    ):
        assert path is not None
        assert path.exists()

    aggregate = json.loads(info["aggregate_summary"].read_text(encoding="utf-8"))
    assert "by_variant_id" in aggregate
    assert "final_only_200_steps" in aggregate["by_variant_id"]

    curve_text = info["curve_csv"].read_text(encoding="utf-8")
    assert "policy_network_has_been_trained" in curve_text.splitlines()[0]
    final_only_rows = [
        line for line in curve_text.splitlines()
        if "final_only_200_steps" in line
    ]
    assert final_only_rows
    assert any(",False,False," in row for row in final_only_rows)

    final_only_result = [
        r for r in results if r["variant_id"] == "final_only_200_steps"
    ][0]
    assert np.isnan(final_only_result["exploitability"][0])
    assert np.isfinite(final_only_result["exploitability"][-1])
