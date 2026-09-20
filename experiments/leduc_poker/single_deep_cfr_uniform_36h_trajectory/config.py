"""Frozen scientific contract for Deep CFR Experiment 29."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from experiments.leduc_poker.single_deep_cfr_comparison.config import (
    DEFAULT_CONFIG as EXPERIMENT_28_CONFIG,
    DEFAULT_SEEDS as EXPERIMENT_28_SEEDS,
)


EXPERIMENT_ID = 29
EXPERIMENT_NAME = "single_deep_cfr_uniform_36h_trajectory"
PRODUCTION_SEEDS = tuple(EXPERIMENT_28_SEEDS)
SMOKE_SEEDS = (PRODUCTION_SEEDS[0],)
TRAINING_HOURS = 36
TRAINING_SECONDS = TRAINING_HOURS * 60 * 60
CHECKPOINT_INTERVAL_SECONDS = 30 * 60
TARGET_NODES_TOUCHED = 15_000_000
MAX_ITERATIONS = 1_000_000_000
PRIMARY_WEIGHTING = "uniform"
SECONDARY_WEIGHTING = "linear"


TRAINING_CONFIG = deepcopy(EXPERIMENT_28_CONFIG)
TRAINING_CONFIG.update(
    {
        "experiment_number": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "algorithm": "standalone uniformly weighted SD-CFR",
        "source_configuration": "Experiment 28 selected uniform SD-CFR arm",
        "num_iterations": MAX_ITERATIONS,
        "training_time_budget_seconds": TRAINING_SECONDS,
        "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
        "target_nodes_touched": TARGET_NODES_TOUCHED,
        "seeds": list(PRODUCTION_SEEDS),
        # The average-policy network is not part of SD-CFR. The advantage
        # learner remains exactly the selected Experiment 28 configuration.
        "policy_training_mode": "disabled",
        "collect_strategy_replay": False,
        "compute_exploitability": False,
        "evaluation_interval": MAX_ITERATIONS,
        "sd_cfr_weightings": (PRIMARY_WEIGHTING, SECONDARY_WEIGHTING),
        "primary_sd_cfr_weighting": PRIMARY_WEIGHTING,
        "save_sd_cfr_archives": True,
    }
)


def build_config(*, smoke: bool = False) -> dict:
    config = deepcopy(TRAINING_CONFIG)
    if smoke:
        config.update(
            {
                "num_iterations": 10_000,
                "num_traversals": 2,
                "training_time_budget_seconds": 0.01,
                "checkpoint_interval_seconds": 0.01,
                "target_nodes_touched": 1,
                "policy_network_layers": (8, 8),
                "advantage_network_layers": (8, 8),
                "memory_capacity": 256,
                "batch_size_advantage": 2,
                "batch_size_strategy": 2,
                "advantage_network_train_steps": 1,
                "policy_network_train_steps": 1,
                "evaluation_interval": 10_000,
                "seeds": list(SMOKE_SEEDS),
                "smoke": True,
            }
        )
    else:
        config["smoke"] = False
    validate_contract(config, smoke=smoke)
    return config


def validate_contract(config: Mapping[str, object], *, smoke: bool) -> None:
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if tuple(map(int, config["seeds"])) != expected_seeds:
        raise ValueError(f"Expected seeds {expected_seeds}, got {config['seeds']}")
    if str(config["experiment_name"]) != EXPERIMENT_NAME:
        raise ValueError("Experiment 29 has the wrong experiment_name")
    if str(config["primary_sd_cfr_weighting"]) != PRIMARY_WEIGHTING:
        raise ValueError("Experiment 29 requires uniform SD-CFR as primary")
    if tuple(config["sd_cfr_weightings"]) != (
        PRIMARY_WEIGHTING,
        SECONDARY_WEIGHTING,
    ):
        raise ValueError("Experiment 29 must evaluate uniform and linear SD-CFR")
    if str(config["policy_training_mode"]) != "disabled":
        raise ValueError("Standalone SD-CFR must not fit an average-policy network")
    if bool(config["collect_strategy_replay"]):
        raise ValueError("Standalone SD-CFR must not collect strategy replay")
    if bool(config["compute_exploitability"]):
        raise ValueError("Exploitability must be deferred until after timed training")
    if float(config["training_time_budget_seconds"]) <= 0:
        raise ValueError("training_time_budget_seconds must be positive")
    if float(config["checkpoint_interval_seconds"]) <= 0:
        raise ValueError("checkpoint_interval_seconds must be positive")

    if not smoke:
        required = {
            "training_time_budget_seconds": TRAINING_SECONDS,
            "checkpoint_interval_seconds": CHECKPOINT_INTERVAL_SECONDS,
            "target_nodes_touched": TARGET_NODES_TOUCHED,
            "num_traversals": 320,
            "policy_network_layers": (32, 32),
            "advantage_network_layers": (32,) * 8,
            "policy_network_type": "mlp",
            "advantage_network_type": "residual_layer_norm_centered_advantage_mlp",
            "learning_rate": 0.004,
            "learning_rate_schedule": "constant",
            "batch_size_advantage": 2_048,
            "batch_size_strategy": 1_024,
            "memory_capacity": 5_000_000,
            "reinitialize_advantage_networks": False,
            "policy_network_train_steps": 200,
            "advantage_network_train_steps": 200,
            "policy_network_train_every": 10,
            "target_processing": "standardize",
            "advantage_replay_sampling": "uniform",
            "average_strategy_weighting": "uniform",
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(
                    f"Experiment 29 field {key} changed from Experiment 28: "
                    f"{config.get(key)!r}"
                )


__all__ = [
    "CHECKPOINT_INTERVAL_SECONDS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "MAX_ITERATIONS",
    "PRIMARY_WEIGHTING",
    "PRODUCTION_SEEDS",
    "SECONDARY_WEIGHTING",
    "SMOKE_SEEDS",
    "TARGET_NODES_TOUCHED",
    "TRAINING_CONFIG",
    "TRAINING_HOURS",
    "TRAINING_SECONDS",
    "build_config",
    "validate_contract",
]

