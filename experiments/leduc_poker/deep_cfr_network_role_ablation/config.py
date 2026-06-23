"""Configuration for Experiment 14: policy-vs-advantage architecture roles."""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
EXTENDED_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

BASELINE_LAYERS = (32, 32)
SMALL_LAYERS = (8, 8)
DEEP_LAYERS = tuple([32] * 8)


def _variant(
    variant_id: str,
    label: str,
    *,
    policy_layers,
    advantage_layers,
    varied_network: str,
) -> dict:
    return {
        "variant_id": variant_id,
        "label": label,
        "network_treatment": "role_specific_capacity",
        "varied_network": varied_network,
        "policy_architecture_label": "x".join(map(str, policy_layers)),
        "advantage_architecture_label": "x".join(map(str, advantage_layers)),
        "policy_network_type": "mlp",
        "advantage_network_type": "mlp",
        "policy_network_layers": tuple(policy_layers),
        "advantage_network_layers": tuple(advantage_layers),
    }


ARCHITECTURE_VARIANTS = (
    _variant(
        "baseline_policy2x32_advantage2x32",
        "Baseline: both 2x32",
        policy_layers=BASELINE_LAYERS,
        advantage_layers=BASELINE_LAYERS,
        varied_network="none",
    ),
    _variant(
        "small_policy_baseline_advantage",
        "Policy 2x8, advantage 2x32",
        policy_layers=SMALL_LAYERS,
        advantage_layers=BASELINE_LAYERS,
        varied_network="policy",
    ),
    _variant(
        "deep_policy_baseline_advantage",
        "Policy 8x32, advantage 2x32",
        policy_layers=DEEP_LAYERS,
        advantage_layers=BASELINE_LAYERS,
        varied_network="policy",
    ),
    _variant(
        "baseline_policy_small_advantage",
        "Policy 2x32, advantage 2x8",
        policy_layers=BASELINE_LAYERS,
        advantage_layers=SMALL_LAYERS,
        varied_network="advantage",
    ),
    _variant(
        "baseline_policy_deep_advantage",
        "Policy 2x32, advantage 8x32",
        policy_layers=BASELINE_LAYERS,
        advantage_layers=DEEP_LAYERS,
        varied_network="advantage",
    ),
)

BASELINE_VARIANT_ID = "baseline_policy2x32_advantage2x32"

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_network_role_ablation",
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": BASELINE_LAYERS,
    "advantage_network_layers": BASELINE_LAYERS,
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

