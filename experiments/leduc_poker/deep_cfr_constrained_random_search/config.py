"""Configuration for the constrained Deep CFR random-search experiment.

The experiment keeps the OpenSpiel Leduc poker environment, policy-extraction
procedure, evaluation cadence, and canonical solver implementation aligned
with Experiment 2. It varies only a bounded set of optimisation and capacity
hyperparameters around the Experiment 2 baseline.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


EXPERIMENT2_BASELINE_CONFIG = {
    "config_label": "experiment2_baseline",
    "game_name": "leduc_poker",
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
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
}

CONSTRAINED_SEARCH_SPACE = {
    "learning_rate": (0.001, 0.003, 0.006),
    "num_traversals": (160, 320, 640),
    "policy_network_layers": ((32, 32), (64, 64), (32, 32, 32)),
    "advantage_network_layers": ((32, 32), (64, 64), (32, 32, 32)),
    "batch_size_advantage": (512, 1024, 2048),
    # Keep average-policy minibatch size fixed so random search does not
    # confound policy-extraction memory pressure with the tuned advantage
    # training hyperparameters.
    "batch_size_strategy": (1024,),
    "memory_capacity": (int(1e6), int(5e6), int(1e7)),
    "policy_network_train_steps": (100, 200),
    "advantage_network_train_steps": (100, 200),
    "reinitialize_advantage_networks": (False, True),
}

SCREENING_SEEDS = [1234, 2025]
CONFIRMATION_SEEDS = [1234, 2025, 31415]
EXTENDED_CONFIRMATION_SEEDS = [1234, 2025, 31415, 27182, 16180]

DEFAULT_CONFIG = {
    "experiment_name": "leduc_deep_cfr_aligned_constrained_random_search",
    "baseline_config": dict(EXPERIMENT2_BASELINE_CONFIG),
    "search_space": CONSTRAINED_SEARCH_SPACE,
    "screening_num_iterations": 500,
    "confirmation_num_iterations": 1500,
    "screening_random_configs": 6,
    "confirmation_top_k": 2,
    "screening_seeds": SCREENING_SEEDS,
    "confirmation_seeds": CONFIRMATION_SEEDS,
    "extended_confirmation_seeds": EXTENDED_CONFIRMATION_SEEDS,
    "master_seed": 20260508,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
    "selection_metric": "final_exploitability",
    "secondary_selection_metric": "normalised_exploitability_auc_by_iteration",
}
