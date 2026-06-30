"""Configuration for validating the composite Deep CFR architecture."""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

WIDTH = 32
POLICY_LAYERS = (32, 32)
BASELINE_ADVANTAGE_LAYERS = (32, 32)
DEEP_ADVANTAGE_LAYERS = (32, 32, 32, 32, 32, 32, 32, 32)

BASELINE_VARIANT_ID = "baseline_direct_2x32"
CENTERED_ADVANTAGE_VARIANT_ID = "centered_advantage_8x32"
COMPOSITE_VARIANT_ID = "composite_res_ln_centered_advantage_8x32"


ARCHITECTURE_VARIANTS = (
    {
        "variant_id": BASELINE_VARIANT_ID,
        "label": "Baseline direct 2x32",
        "network_treatment": "baseline_direct",
        "varied_network": "advantage",
        "architecture_depth": 2,
        "architecture_width": WIDTH,
        "policy_architecture_label": "x".join(map(str, POLICY_LAYERS)),
        "advantage_architecture_label": "x".join(map(str, BASELINE_ADVANTAGE_LAYERS)),
        "policy_network_type": "mlp",
        "advantage_network_type": "mlp",
        "policy_network_layers": POLICY_LAYERS,
        "advantage_network_layers": BASELINE_ADVANTAGE_LAYERS,
    },
    {
        "variant_id": CENTERED_ADVANTAGE_VARIANT_ID,
        "label": "Centred advantage 8x32",
        "network_treatment": "centered_advantage",
        "varied_network": "advantage",
        "architecture_depth": 8,
        "architecture_width": WIDTH,
        "policy_architecture_label": "x".join(map(str, POLICY_LAYERS)),
        "advantage_architecture_label": "x".join(map(str, DEEP_ADVANTAGE_LAYERS)),
        "policy_network_type": "mlp",
        "advantage_network_type": "centered_advantage_mlp",
        "policy_network_layers": POLICY_LAYERS,
        "advantage_network_layers": DEEP_ADVANTAGE_LAYERS,
    },
    {
        "variant_id": COMPOSITE_VARIANT_ID,
        "label": "Composite Res+LN centred advantage 8x32",
        "network_treatment": "composite_residual_layer_norm_centered",
        "varied_network": "advantage",
        "architecture_depth": 8,
        "architecture_width": WIDTH,
        "policy_architecture_label": "x".join(map(str, POLICY_LAYERS)),
        "advantage_architecture_label": "x".join(map(str, DEEP_ADVANTAGE_LAYERS)),
        "policy_network_type": "mlp",
        "advantage_network_type": "residual_layer_norm_centered_advantage_mlp",
        "policy_network_layers": POLICY_LAYERS,
        "advantage_network_layers": DEEP_ADVANTAGE_LAYERS,
    },
)


DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_composite_architecture_validation",
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": POLICY_LAYERS,
    "advantage_network_layers": BASELINE_ADVANTAGE_LAYERS,
    "policy_network_type": "mlp",
    "advantage_network_type": "mlp",
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
    "architecture_variants": ARCHITECTURE_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "composite_variant_id": COMPOSITE_VARIANT_ID,
}
