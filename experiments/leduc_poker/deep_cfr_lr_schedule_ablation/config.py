"""Configuration for the learning-rate schedule ablation.

This experiment intentionally varies only the learning-rate schedule. The
default two-arm protocol compares the Experiment 2 constant-learning-rate
baseline with cosine decay to 10% of the initial rate under matched seeds and
the same Deep CFR training budget.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


SCHEDULE_CONFIGS = [
    {
        "schedule": "constant_baseline_exp2",
        "label": "Constant baseline",
        "learning_rate_schedule": "constant",
        "learning_rate_end": 0.003,
        "learning_rate_warmup_iterations": 0,
    },
    {
        "schedule": "cosine_decay_to_10pct",
        "label": "Cosine decay to 10%",
        "learning_rate_schedule": "cosine_decay",
        "learning_rate_end": 0.0003,
        "learning_rate_warmup_iterations": 0,
    },
]

OPTIONAL_EXTRA_SCHEDULES = [
    {
        "schedule": "linear_decay_to_10pct",
        "label": "Linear decay to 10%",
        "learning_rate_schedule": "linear_decay",
        "learning_rate_end": 0.0003,
        "learning_rate_warmup_iterations": 0,
    },
    {
        "schedule": "step_decay_to_10pct",
        "label": "Step decay to 10%",
        "learning_rate_schedule": "step_decay",
        "learning_rate_end": 0.0003,
        "learning_rate_decay_rate": 0.5,
        "learning_rate_decay_steps": 500,
        "learning_rate_warmup_iterations": 0,
    },
]

BASELINE_SCHEDULE = "constant_baseline_exp2"

DEFAULT_SEEDS = [1234, 2025, 31415]
EXTENDED_SEEDS_5 = [1234, 2025, 31415, 27182, 16180]
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_lr_schedule_ablation_aligned",
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
    "learning_rate": 0.003,
    "learning_rate_end": 0.0003,
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
    "schedule_configs": tuple(SCHEDULE_CONFIGS),
    "optional_extra_schedules": tuple(OPTIONAL_EXTRA_SCHEDULES),
    "baseline_schedule": BASELINE_SCHEDULE,
}
