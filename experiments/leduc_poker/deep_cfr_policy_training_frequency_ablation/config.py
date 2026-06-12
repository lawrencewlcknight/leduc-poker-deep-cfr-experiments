"""Configuration for the Deep CFR policy-training-frequency ablation.

This experiment intentionally deviates from the validation experiment along
one axis only: ``policy_network_train_every``. The evaluation interval remains
fixed at 25 CFR iterations for every ablation arm so exploitability and
diagnostic curves are measured at identical checkpoints.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


POLICY_TRAIN_EVERY_VARIANTS = [10, 25, 50, 100]
REFERENCE_POLICY_TRAIN_EVERY = 10

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_policy_training_frequency_ablation",
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_train_every_variants": tuple(POLICY_TRAIN_EVERY_VARIANTS),
    "reference_policy_train_every": REFERENCE_POLICY_TRAIN_EVERY,
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "memory_capacity": int(1e7),
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 200,
    "advantage_network_train_steps": 200,
    "compute_exploitability": True,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
}
