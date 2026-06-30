"""Target-processing ablation on the composite Deep CFR architecture.

This experiment mirrors the original Leduc target-processing ablation, but the
raw-target baseline is the proposed composite architecture rather than the
original 2x32 direct-advantage baseline.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)
from experiments.leduc_poker.deep_cfr_composite_architecture_validation.config import (
    COMPOSITE_VARIANT_ID,
    DEEP_ADVANTAGE_LAYERS,
    POLICY_LAYERS,
)


TARGET_PROCESSING_VARIANTS = [
    {
        "variant_id": "composite_raw_targets_baseline",
        "label": "Composite raw targets",
        "target_processing": "none",
        "target_clip_value": 1.0,
    },
    {
        "variant_id": "composite_standardized_targets",
        "label": "Composite standardized targets",
        "target_processing": "standardize",
        "target_clip_value": 1.0,
    },
    {
        "variant_id": "composite_clipped_targets",
        "label": "Composite clipped targets",
        "target_processing": "clip",
        "target_clip_value": 1.0,
    },
    {
        "variant_id": "composite_standardized_clipped_targets",
        "label": "Composite standardized + clipped targets",
        "target_processing": "standardize_clip",
        "target_clip_value": 1.0,
    },
]

BASELINE_VARIANT_ID = "composite_raw_targets_baseline"

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]
DEFAULT_SEEDS_5 = DEFAULT_SEEDS
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_composite_target_processing_ablation",
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": POLICY_LAYERS,
    "advantage_network_layers": DEEP_ADVANTAGE_LAYERS,
    "policy_network_type": "mlp",
    "advantage_network_type": "residual_layer_norm_centered_advantage_mlp",
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
    "reference_architecture_variant_id": COMPOSITE_VARIANT_ID,
}
