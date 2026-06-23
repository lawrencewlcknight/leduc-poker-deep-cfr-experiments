"""Configuration for Experiment 15: shared trunk with player/action heads."""

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


def _variant(depth: int, shared_trunk: bool) -> dict:
    layers = tuple([WIDTH] * int(depth))
    treatment = "shared_trunk_player_heads" if shared_trunk else "independent_advantage"
    label_prefix = "Shared trunk" if shared_trunk else "Independent"
    variant_prefix = "shared_trunk_advantage" if shared_trunk else "independent_advantage"
    return {
        "variant_id": f"{variant_prefix}_layers{depth}_width{WIDTH}",
        "label": f"{label_prefix} advantage {depth}x{WIDTH}",
        "network_treatment": treatment,
        "varied_network": "advantage",
        "architecture_depth": int(depth),
        "architecture_width": WIDTH,
        "policy_architecture_label": "x".join(map(str, POLICY_LAYERS)),
        "advantage_architecture_label": "x".join(map(str, layers)),
        "policy_network_type": "mlp",
        "advantage_network_type": (
            "shared_trunk_player_heads" if shared_trunk else "mlp"
        ),
        "policy_network_layers": POLICY_LAYERS,
        "advantage_network_layers": layers,
    }


ARCHITECTURE_VARIANTS = tuple(
    _variant(depth, shared_trunk)
    for depth in DEPTHS
    for shared_trunk in (False, True)
)

BASELINE_VARIANT_ID = "independent_advantage_layers2_width32"

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_shared_trunk_head_ablation",
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
