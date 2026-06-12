"""Configuration for the Deep CFR replay and averaging ablation.

The experiment keeps the Experiment 2 Leduc poker Deep CFR setup fixed and
varies only average-policy target weighting in its default, thesis-facing run.
Priority replay variants remain available for exploratory runs, but are not
included by default because the current priority sampler is too expensive for
the standard experiment suite.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


REPLAY_AVERAGING_VARIANTS = [
    {
        "variant_id": "uniform_replay_linear_avg_exp2_baseline",
        "label": "Uniform replay + linear avg",
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "linear",
        "description": (
            "Experiment 2 baseline: uniform advantage replay and CFR-style "
            "linear average-strategy weighting."
        ),
    },
    {
        "variant_id": "priority_replay_linear_avg",
        "label": "Priority replay + linear avg",
        "advantage_replay_sampling": "priority_abs_adv",
        "average_strategy_weighting": "linear",
        "description": (
            "Priority advantage replay based on absolute regret-target "
            "magnitude with baseline average-strategy weighting."
        ),
    },
    {
        "variant_id": "uniform_replay_uniform_avg",
        "label": "Uniform replay + uniform avg",
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "uniform",
        "description": (
            "Baseline advantage replay with uniform average-strategy weighting."
        ),
    },
    {
        "variant_id": "priority_replay_uniform_avg",
        "label": "Priority replay + uniform avg",
        "advantage_replay_sampling": "priority_abs_adv",
        "average_strategy_weighting": "uniform",
        "description": (
            "Priority advantage replay combined with uniform average-strategy "
            "weighting."
        ),
    },
]

BASELINE_VARIANT_ID = "uniform_replay_linear_avg_exp2_baseline"
DEFAULT_REPLAY_AVERAGING_VARIANTS = (
    REPLAY_AVERAGING_VARIANTS[0],
    REPLAY_AVERAGING_VARIANTS[2],
)

DEFAULT_SEEDS = [1234, 2025, 31415]
DEFAULT_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_replay_and_averaging_ablation",
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
    "advantage_replay_sampling": "uniform",
    "average_strategy_weighting": "linear",
    "priority_alpha": 1.0,
    "priority_epsilon": 1e-6,
    "ablation_variants": DEFAULT_REPLAY_AVERAGING_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
}
