"""Configuration for the Deep CFR target-processing ablation.

The experiment keeps the Experiment 2 Leduc poker Deep CFR setup fixed and
varies only how sampled advantage targets are processed before fitting the
advantage networks. Replay buffers continue to store raw sampled regret
targets for all variants.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


TARGET_PROCESSING_VARIANTS = [
    {
        "variant_id": "raw_targets_exp2_baseline",
        "label": "Raw targets",
        "target_processing": "none",
        "target_clip_value": 1.0,
    },
    {
        "variant_id": "standardized_targets",
        "label": "Standardized targets",
        "target_processing": "standardize",
        "target_clip_value": 1.0,
    },
    {
        "variant_id": "clipped_targets",
        "label": "Clipped targets",
        "target_processing": "clip",
        "target_clip_value": 1.0,
    },
    {
        "variant_id": "standardized_clipped_targets",
        "label": "Standardized + clipped targets",
        "target_processing": "standardize_clip",
        "target_clip_value": 1.0,
    },
]

BASELINE_VARIANT_ID = "raw_targets_exp2_baseline"

DEFAULT_SEEDS = [1234, 2025, 31415]
DEFAULT_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_target_processing_ablation",
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
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 200,
    "advantage_network_train_steps": 200,
    "policy_network_train_every": 25,
    "policy_training_mode": "intermittent",
    "final_policy_network_train_steps": None,
    "compute_exploitability": True,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
    "target_processing": "none",
    "target_clip_value": 1.0,
    "target_standardize_epsilon": 1e-6,
    "ablation_variants": tuple(TARGET_PROCESSING_VARIANTS),
    "baseline_variant_id": BASELINE_VARIANT_ID,
}
