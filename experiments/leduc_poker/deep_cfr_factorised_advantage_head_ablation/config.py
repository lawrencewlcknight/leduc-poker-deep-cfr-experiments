"""Configuration for Experiment 16: factorised advantage-output heads."""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
EXTENDED_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

WIDTH = 32
DEPTHS = (2, 4, 8)
POLICY_LAYERS = (32, 32)

HEAD_TYPES = (
    ("mlp", "direct", "Direct"),
    ("centered_advantage_mlp", "centered", "Centred"),
    ("dueling_mlp", "dueling", "Dueling"),
)


def _variant(
    depth: int,
    network_type: str,
    variant_prefix: str,
    label_prefix: str,
) -> dict:
    layers = tuple([WIDTH] * int(depth))
    return {
        "variant_id": f"{variant_prefix}_advantage_layers{depth}_width{WIDTH}",
        "label": f"{label_prefix} advantage head {depth}x{WIDTH}",
        "network_treatment": variant_prefix,
        "varied_network": "advantage",
        "architecture_depth": int(depth),
        "architecture_width": WIDTH,
        "policy_architecture_label": "x".join(map(str, POLICY_LAYERS)),
        "advantage_architecture_label": "x".join(map(str, layers)),
        "policy_network_type": "mlp",
        "advantage_network_type": network_type,
        "policy_network_layers": POLICY_LAYERS,
        "advantage_network_layers": layers,
    }


ARCHITECTURE_VARIANTS = tuple(
    _variant(depth, network_type, variant_prefix, label_prefix)
    for depth in DEPTHS
    for network_type, variant_prefix, label_prefix in HEAD_TYPES
)

BASELINE_VARIANT_ID = "direct_advantage_layers2_width32"

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_factorised_advantage_head_ablation",
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": POLICY_LAYERS,
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
