"""Configuration for Experiment 17: dropout regularisation ablation."""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
EXTENDED_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

WIDTH = 32
DEPTHS = (2, 8)
POLICY_LAYERS = (32, 32)

DROPOUT_TREATMENTS = (
    ("mlp", "p00", "No dropout", 0.0),
    ("dropout_mlp_p05", "p05", "Dropout 0.05", 0.05),
    ("dropout_mlp_p10", "p10", "Dropout 0.10", 0.10),
    ("dropout_mlp_p20", "p20", "Dropout 0.20", 0.20),
)


def _variant(
    depth: int,
    network_type: str,
    dropout_id: str,
    label_prefix: str,
    dropout_probability: float,
) -> dict:
    layers = tuple([WIDTH] * int(depth))
    return {
        "variant_id": f"dropout_{dropout_id}_advantage_layers{depth}_width{WIDTH}",
        "label": f"{label_prefix}, advantage {depth}x{WIDTH}",
        "network_treatment": "dropout_regularisation",
        "varied_network": "advantage",
        "architecture_depth": int(depth),
        "architecture_width": WIDTH,
        "dropout_probability": float(dropout_probability),
        "policy_architecture_label": "x".join(map(str, POLICY_LAYERS)),
        "advantage_architecture_label": "x".join(map(str, layers)),
        "policy_network_type": "mlp",
        "advantage_network_type": network_type,
        "policy_network_layers": POLICY_LAYERS,
        "advantage_network_layers": layers,
    }


ARCHITECTURE_VARIANTS = tuple(
    _variant(depth, network_type, dropout_id, label_prefix, dropout_probability)
    for depth in DEPTHS
    for network_type, dropout_id, label_prefix, dropout_probability in DROPOUT_TREATMENTS
)

BASELINE_VARIANT_ID = "dropout_p00_advantage_layers2_width32"

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_dropout_ablation",
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
