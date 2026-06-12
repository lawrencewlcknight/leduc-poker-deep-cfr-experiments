"""Configuration for the fair warm-start/checkpoint ablation.

This experiment intentionally varies only whether training is interrupted,
saved as a full checkpoint, reloaded, and resumed. The continuous baseline and
warm-start arm share the same seed, total iteration budget, traversal count,
network architecture, optimiser settings, replay buffers, policy-training
frequency, evaluation interval, and solver implementation.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


ARMS = [
    {
        "arm": "baseline_continuous",
        "label": "Continuous baseline",
        "description": "Train continuously from random initialisation for the full budget.",
    },
    {
        "arm": "warm_start",
        "label": "Checkpoint and resume",
        "description": "Train to the boundary, save a full checkpoint, reload, and continue.",
    },
]

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]
FULL_BASELINE_SEEDS = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_fair_warm_start_ablation",
    "game_name": "leduc_poker",
    "total_iterations": 1500,
    "warm_start_boundary": 100,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "memory_capacity": int(1e7),
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 200,
    "advantage_network_train_steps": 200,
    "policy_network_train_every": 25,
    "policy_training_mode": "intermittent",
    "final_policy_network_train_steps": None,
    "compute_exploitability": True,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
    "arms": tuple(ARMS),
}
