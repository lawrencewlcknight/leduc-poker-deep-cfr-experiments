"""Replay and averaging ablation on the standardised composite baseline."""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)
from experiments.leduc_poker.deep_cfr_composite_architecture_validation.config import (
    COMPOSITE_VARIANT_ID,
    DEEP_ADVANTAGE_LAYERS,
    POLICY_LAYERS,
)


REPLAY_AVERAGING_VARIANTS = [
    {
        "variant_id": "composite_std_uniform_replay_linear_avg_baseline",
        "label": "Composite std targets + linear avg",
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "linear",
        "description": (
            "Current composite baseline with standardised advantage targets, "
            "uniform advantage replay, and CFR-style linear average-strategy "
            "weighting."
        ),
    },
    {
        "variant_id": "composite_std_priority_replay_linear_avg",
        "label": "Composite std targets + priority replay + linear avg",
        "advantage_replay_sampling": "priority_abs_adv",
        "average_strategy_weighting": "linear",
        "description": (
            "Exploratory priority advantage replay with the standardised "
            "composite baseline and linear average-strategy weighting."
        ),
    },
    {
        "variant_id": "composite_std_uniform_replay_uniform_avg",
        "label": "Composite std targets + uniform avg",
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "uniform",
        "description": (
            "Standardised composite baseline with uniform average-strategy "
            "weighting."
        ),
    },
    {
        "variant_id": "composite_std_priority_replay_uniform_avg",
        "label": "Composite std targets + priority replay + uniform avg",
        "advantage_replay_sampling": "priority_abs_adv",
        "average_strategy_weighting": "uniform",
        "description": (
            "Exploratory priority advantage replay with the standardised "
            "composite baseline and uniform average-strategy weighting."
        ),
    },
]

BASELINE_VARIANT_ID = "composite_std_uniform_replay_linear_avg_baseline"
DEFAULT_REPLAY_AVERAGING_VARIANTS = (
    REPLAY_AVERAGING_VARIANTS[0],
    REPLAY_AVERAGING_VARIANTS[2],
)

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]
DEFAULT_SEEDS_5 = DEFAULT_SEEDS
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

DEFAULT_CONFIG = {
    "experiment_name": (
        "leduc_poker_deep_cfr_composite_standardized_replay_averaging_ablation"
    ),
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
    "target_processing": "standardize",
    "target_clip_value": 1.0,
    "target_standardize_epsilon": 1e-6,
    "advantage_replay_sampling": "uniform",
    "average_strategy_weighting": "linear",
    "priority_alpha": 1.0,
    "priority_epsilon": 1e-6,
    "ablation_variants": DEFAULT_REPLAY_AVERAGING_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "reference_architecture_variant_id": COMPOSITE_VARIANT_ID,
}
