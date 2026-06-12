"""Configuration for the final-only average-policy training ablation.

This experiment intentionally varies only the timing and total supervised
update budget of the average-policy network. CFR data collection, advantage
network training, traversals, replay capacity, architecture, optimiser
settings, and evaluation checkpoints match the validation experiment.
"""

import math

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


BASE_POLICY_NETWORK_TRAIN_EVERY = 25
BASE_POLICY_NETWORK_TRAIN_STEPS = 200
NUM_ITERATIONS = 1500
INTERMITTENT_EVENTS = int(
    math.ceil(NUM_ITERATIONS / BASE_POLICY_NETWORK_TRAIN_EVERY)
)
MATCHED_FINAL_POLICY_STEPS = int(
    INTERMITTENT_EVENTS * BASE_POLICY_NETWORK_TRAIN_STEPS
)

POLICY_TRAINING_VARIANTS = [
    {
        "variant_id": "intermittent_every_25",
        "label": "Intermittent: every 25 iterations",
        "policy_training_mode": "intermittent",
        "policy_network_train_every": BASE_POLICY_NETWORK_TRAIN_EVERY,
        "policy_network_train_steps": BASE_POLICY_NETWORK_TRAIN_STEPS,
        "final_policy_network_train_steps": None,
        "description": "Baseline: train the average-policy network every 25 CFR iterations.",
    },
    {
        "variant_id": "final_only_200_steps",
        "label": "Final only: 200 steps",
        "policy_training_mode": "final_only",
        "policy_network_train_every": BASE_POLICY_NETWORK_TRAIN_EVERY,
        "policy_network_train_steps": BASE_POLICY_NETWORK_TRAIN_STEPS,
        "final_policy_network_train_steps": BASE_POLICY_NETWORK_TRAIN_STEPS,
        "description": "Train the average-policy network once at the end for the usual event size.",
    },
    {
        "variant_id": "final_only_matched_steps",
        "label": f"Final only: {MATCHED_FINAL_POLICY_STEPS} steps",
        "policy_training_mode": "final_only",
        "policy_network_train_every": BASE_POLICY_NETWORK_TRAIN_EVERY,
        "policy_network_train_steps": BASE_POLICY_NETWORK_TRAIN_STEPS,
        "final_policy_network_train_steps": MATCHED_FINAL_POLICY_STEPS,
        "description": "Final-only extraction with the same total policy-gradient-step budget as the intermittent baseline.",
    },
]

REFERENCE_VARIANT_ID = "intermittent_every_25"

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_final_only_policy_training_ablation",
    "game_name": "leduc_poker",
    "num_iterations": NUM_ITERATIONS,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_training_variants": tuple(POLICY_TRAINING_VARIANTS),
    "reference_variant_id": REFERENCE_VARIANT_ID,
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "memory_capacity": int(1e7),
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": BASE_POLICY_NETWORK_TRAIN_STEPS,
    "policy_network_train_every": BASE_POLICY_NETWORK_TRAIN_EVERY,
    "final_policy_network_train_steps": None,
    "advantage_network_train_steps": 200,
    "compute_exploitability": True,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
}
