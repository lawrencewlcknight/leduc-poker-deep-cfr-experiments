"""Configuration for Experiment 13: layer normalisation and residual-LN MLPs."""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
EXTENDED_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]
WIDTH = 32
DEPTHS = (2, 4, 8)

_LABELS = {
    "mlp": "Plain",
    "layer_norm_mlp": "LayerNorm",
    "residual_layer_norm_mlp": "Residual+LayerNorm",
}
_PREFIXES = {
    "mlp": "plain",
    "layer_norm_mlp": "layer_norm",
    "residual_layer_norm_mlp": "residual_layer_norm",
}


def _variant(depth: int, network_type: str) -> dict:
    layers = tuple([WIDTH] * int(depth))
    return {
        "variant_id": f"{_PREFIXES[network_type]}_layers{depth}_width{WIDTH}",
        "label": f"{_LABELS[network_type]} {depth}x{WIDTH}",
        "network_treatment": network_type,
        "architecture_depth": int(depth),
        "architecture_width": WIDTH,
        "policy_network_type": network_type,
        "advantage_network_type": network_type,
        "policy_network_layers": layers,
        "advantage_network_layers": layers,
    }


ARCHITECTURE_VARIANTS = tuple(
    _variant(depth, network_type)
    for depth in DEPTHS
    for network_type in ("mlp", "layer_norm_mlp", "residual_layer_norm_mlp")
)

BASELINE_VARIANT_ID = "plain_layers2_width32"

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_layer_norm_network_ablation",
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
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
}

