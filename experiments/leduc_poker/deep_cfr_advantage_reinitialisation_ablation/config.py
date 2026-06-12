"""Configuration for the advantage-network reinitialisation ablation.

This experiment intentionally varies only ``reinitialize_advantage_networks``.
All other solver settings match the multi-seed validation baseline so that
differences can be attributed to warm-starting versus resetting the advantage
networks before each advantage-network training phase.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


ABLATION_VARIANTS = [
    {
        "variant_id": "reinit_false_warm_started_advantage",
        "label": "Warm-started advantage networks",
        "reinitialize_advantage_networks": False,
        "description": "Advantage networks continue training across CFR iterations.",
    },
    {
        "variant_id": "reinit_true_reset_advantage",
        "label": "Reinitialised advantage networks",
        "reinitialize_advantage_networks": True,
        "description": "Advantage networks are reset before each advantage-network training phase.",
    },
]

REFERENCE_VARIANT_ID = "reinit_false_warm_started_advantage"
COMPARISON_VARIANT_ID = "reinit_true_reset_advantage"

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]
SMOKE_TEST_SEEDS = [1234, 2025]
DEFAULT_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_advantage_reinitialisation_ablation",
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "memory_capacity": int(1e7),
    "policy_network_train_steps": 200,
    "advantage_network_train_steps": 200,
    "policy_network_train_every": 25,
    "policy_training_mode": "intermittent",
    "final_policy_network_train_steps": None,
    "compute_exploitability": True,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
    "ablation_variants": tuple(ABLATION_VARIANTS),
    "reference_variant_id": REFERENCE_VARIANT_ID,
    "comparison_variant_id": COMPARISON_VARIANT_ID,
}
