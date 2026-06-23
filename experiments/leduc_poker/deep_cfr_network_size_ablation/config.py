"""Configuration for the Leduc poker Deep CFR network-size ablation.

This experiment is anchored to Experiment 1. It keeps the data-generation
budget, optimiser, replay capacity, average-policy training schedule, and
seeds fixed while varying only the hidden depth and hidden width of both the
advantage networks and the average-policy network.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


NETWORK_DEPTHS = (2, 4, 8)
NETWORK_WIDTHS = (8, 16, 32)


def make_architecture_variant(depth: int, width: int) -> dict:
    """Returns one architecture variant for a shared policy/advantage MLP size."""
    depth = int(depth)
    width = int(width)
    layers = tuple([width] * depth)
    return {
        "variant_id": f"layers{depth}_width{width}",
        "label": f"{depth}x{width}",
        "architecture_depth": depth,
        "architecture_width": width,
        "policy_network_layers": layers,
        "advantage_network_layers": layers,
    }


NETWORK_SIZE_VARIANTS = tuple(
    make_architecture_variant(depth, width)
    for depth in NETWORK_DEPTHS
    for width in NETWORK_WIDTHS
)

BASELINE_VARIANT_ID = "layers2_width32"

DEFAULT_SEEDS_3 = [1234, 2025, 31415]
DEFAULT_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]
DEFAULT_SEEDS = DEFAULT_SEEDS_3

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_network_size_ablation",
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
    "architecture_variants": NETWORK_SIZE_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
}
