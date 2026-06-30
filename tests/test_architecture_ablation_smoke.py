"""Smoke tests for the optional network-architecture ablation exports."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from deep_cfr_poker.experiment_utils import run_single_seed
from deep_cfr_poker.constants import DEFAULT_AVERAGE_POLICY_VALUE_TARGET
from experiments.leduc_poker.architecture_ablation_common import (
    _augment_result,
    _variant_config,
    export_ablation_results,
)
from experiments.leduc_poker.deep_cfr_layer_norm_network_ablation.config import (
    ARCHITECTURE_VARIANTS as LAYER_NORM_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_dropout_ablation.config import (
    ARCHITECTURE_VARIANTS as DROPOUT_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_factorised_advantage_head_ablation.config import (
    ARCHITECTURE_VARIANTS as FACTORISED_HEAD_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_composite_architecture_validation.config import (
    ARCHITECTURE_VARIANTS as COMPOSITE_ARCHITECTURE_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_network_role_ablation.config import (
    ARCHITECTURE_VARIANTS as ROLE_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_residual_network_ablation.config import (
    ARCHITECTURE_VARIANTS as RESIDUAL_VARIANTS,
)
from experiments.leduc_poker.deep_cfr_shared_trunk_head_ablation.config import (
    ARCHITECTURE_VARIANTS as SHARED_TRUNK_VARIANTS,
)


BASE_SMOKE_CONFIG = {
    "game_name": "leduc_poker",
    "num_iterations": 3,
    "num_traversals": 4,
    "evaluation_interval": 1,
    "policy_network_layers": (8, 8),
    "advantage_network_layers": (8, 8),
    "policy_network_type": "mlp",
    "advantage_network_type": "mlp",
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
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
}


@pytest.mark.smoke
@pytest.mark.parametrize(
    "name,variants,baseline_id",
    [
        (
            "residual",
            RESIDUAL_VARIANTS[:2],
            "plain_layers2_width32",
        ),
        (
            "layer_norm",
            LAYER_NORM_VARIANTS[:3],
            "plain_layers2_width32",
        ),
        (
            "role",
            ROLE_VARIANTS[:3],
            "baseline_policy2x32_advantage2x32",
        ),
        (
            "shared_trunk",
            SHARED_TRUNK_VARIANTS[:2],
            "independent_advantage_layers2_width32",
        ),
        (
            "factorised_head",
            FACTORISED_HEAD_VARIANTS[:3],
            "direct_advantage_layers2_width32",
        ),
        (
            "composite_architecture",
            COMPOSITE_ARCHITECTURE_VARIANTS,
            "baseline_direct_2x32",
        ),
        (
            "dropout",
            DROPOUT_VARIANTS[:2],
            "dropout_p00_advantage_layers2_width32",
        ),
    ],
)
def test_architecture_ablation_writes_expected_artifacts(
    tmp_path, name, variants, baseline_id
):
    config = {
        **BASE_SMOKE_CONFIG,
        "experiment_name": f"{name}_smoke",
        "architecture_variants": tuple(variants),
        "baseline_variant_id": baseline_id,
    }
    results = []
    for variant in config["architecture_variants"]:
        variant_config = _variant_config(config, variant)
        result = run_single_seed(1234, variant_config, export_dir=tmp_path)
        results.append(
            _augment_result(
                result,
                variant_config,
                final_window=2,
                exploitability_threshold=config["exploitability_threshold"],
            )
        )

    info = export_ablation_results(results, tmp_path, config, [1234])

    for path in (
        info["summary_csv"],
        info["curve_csv"],
        info["aggregate_summary"],
        info["metadata"],
        info["variant_summary_csv"],
        info["ablation_curves_npz"],
        info["paired_differences_csv"],
        info["paired_difference_summary"],
    ):
        assert path is not None
        assert path.exists()

    aggregate = json.loads(info["aggregate_summary"].read_text(encoding="utf-8"))
    assert "by_variant_id" in aggregate

    summary_header = info["summary_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "policy_network_type" in summary_header
    assert "advantage_network_type" in summary_header

    curve_header = info["curve_csv"].read_text(encoding="utf-8").splitlines()[0]
    assert "policy_network_type" in curve_header
    assert "advantage_network_type" in curve_header
