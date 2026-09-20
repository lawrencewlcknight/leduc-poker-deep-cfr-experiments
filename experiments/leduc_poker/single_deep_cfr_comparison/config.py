"""Configuration for Experiment 28, the paired SD-CFR comparison."""

from copy import deepcopy

from experiments.leduc_poker.deep_cfr_multiseed_validation.config import (
    DEFAULT_CONFIG as DEEP_CFR_BASE_CONFIG,
)


DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]

# This is the selected Experiment 26 candidate used as the single training arm
# in Experiment 27. The settings are explicit here so importing the Experiment
# 28 configuration does not also import the full ablation-analysis stack.
DEFAULT_CONFIG = deepcopy(DEEP_CFR_BASE_CONFIG)
DEFAULT_CONFIG.update(
    {
        "experiment_name": "leduc_poker_single_deep_cfr_comparison",
        "num_iterations": 1050,
        "num_traversals": 320,
        "evaluation_interval": 25,
        "policy_network_layers": (32, 32),
        "advantage_network_layers": (32,) * 8,
        "policy_network_type": "mlp",
        "advantage_network_type": "residual_layer_norm_centered_advantage_mlp",
        "learning_rate": 0.004,
        "learning_rate_schedule": "constant",
        "batch_size_advantage": 2048,
        "batch_size_strategy": 1024,
        "memory_capacity": int(5e6),
        "reinitialize_advantage_networks": False,
        "policy_network_train_steps": 200,
        "advantage_network_train_steps": 200,
        "policy_network_train_every": 10,
        "policy_training_mode": "intermittent",
        "final_policy_network_train_steps": None,
        "target_processing": "standardize",
        "target_clip_value": 1.0,
        "target_standardize_epsilon": 1e-6,
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "uniform",
        "priority_alpha": 1.0,
        "priority_epsilon": 1e-6,
        # The uniform arm matches the selected Deep CFR candidate's average-
        # strategy weighting and therefore isolates representation. The linear
        # arm is canonical SD-CFR and is retained as a secondary comparison.
        "sd_cfr_weightings": ("uniform", "linear"),
        "primary_sd_cfr_weighting": "uniform",
        "save_sd_cfr_archives": True,
        "experiment_number": 28,
    }
)


__all__ = ["DEFAULT_CONFIG", "DEFAULT_SEEDS"]
