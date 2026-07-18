"""Configuration tests for Experiment 25's parallel-equivalence ablation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from experiments.leduc_poker.deep_cfr_parallel_equivalence_ablation.config import (
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
    PARALLEL_NUM_WORKERS,
    PARALLEL_VARIANT_ID,
    SEQUENTIAL_VARIANT_ID,
    VARIANTS,
)
from experiments.leduc_poker.deep_cfr_parallel_equivalence_ablation.run import (
    _execution_variant_config,
    _worker_stem,
    build_config,
    build_parser,
)


def _default_args(**overrides):
    values = {
        "variant_ids": None,
        "experiment_name": None,
        "iterations": None,
        "traversals": None,
        "evaluation_interval": None,
        "policy_network_layers": None,
        "advantage_network_layers": None,
        "learning_rate": None,
        "batch_size_advantage": None,
        "batch_size_strategy": None,
        "memory_capacity": None,
        "reinitialize_advantage_networks": None,
        "policy_network_train_steps": None,
        "advantage_network_train_steps": None,
        "policy_network_train_every": None,
        "final_policy_network_train_steps": None,
        "compute_exploitability": None,
        "target_standardize_epsilon": None,
        "priority_alpha": None,
        "priority_epsilon": None,
        "baseline_variant_id": None,
        "parallel_num_workers": None,
        "parallel_ray_address": None,
        "parallel_log_to_driver": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parallel_equivalence_default_config_shape():
    config = build_config(_default_args())
    assert DEFAULT_SEEDS == [1234, 2025, 31415]
    assert config["baseline_variant_id"] == SEQUENTIAL_VARIANT_ID
    assert [v["variant_id"] for v in config["ablation_variants"]] == [
        SEQUENTIAL_VARIANT_ID,
        PARALLEL_VARIANT_ID,
    ]
    assert FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN > 0.0
    assert FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN > 0.0


def test_parallel_arm_changes_only_execution_metadata():
    sequential = _execution_variant_config(DEFAULT_CONFIG, VARIANTS[0])
    parallel = _execution_variant_config(DEFAULT_CONFIG, VARIANTS[1])
    assert sequential["execution_backend"] == "sequential"
    assert sequential["parallel_num_workers"] == 1
    assert parallel["execution_backend"] == "ray_parallel"
    assert parallel["parallel_num_workers"] == PARALLEL_NUM_WORKERS

    allowed_differences = {
        "ablation_variants",
        "baseline_variant_id",
        "description",
        "execution_backend",
        "hp_value",
        "label",
        "parallel_num_workers",
        "variant_id",
    }
    shared_keys = set(sequential) & set(parallel)
    differing = {
        key
        for key in shared_keys
        if sequential[key] != parallel[key]
    }
    assert differing <= allowed_differences


def test_parallel_worker_override_only_affects_parallel_arm():
    config = build_config(_default_args(parallel_num_workers=2))
    sequential = _execution_variant_config(config, config["ablation_variants"][0])
    parallel = _execution_variant_config(config, config["ablation_variants"][1])
    assert sequential["parallel_num_workers"] == 1
    assert parallel["parallel_num_workers"] == 2


def test_subprocess_isolation_cli_defaults_to_enabled():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.disable_subprocess_isolation is False

    disabled = parser.parse_args(["--disable-subprocess-isolation"])
    assert disabled.disable_subprocess_isolation is True


def test_worker_stem_is_safe_for_paths():
    assert _worker_stem("parallel/arm:1", 1234) == "parallel_arm_1_seed_1234"
