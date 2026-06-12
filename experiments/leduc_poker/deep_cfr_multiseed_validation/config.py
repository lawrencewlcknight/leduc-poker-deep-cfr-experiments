"""Default configuration for the Leduc poker Deep CFR validation experiment.

The defaults below are the canonical configuration used to produce the
multi-seed validation curves reported in the thesis. They intentionally
*override* several of the library defaults in :class:`DeepCFRSolver`:

- ``memory_capacity`` is 1e7 here vs 1e6 in the solver default. OpenSpiel's
  default two-player Leduc tree has 936 player information sets, so the larger
  capacity keeps the replay-buffer behaviour aligned with the original
  experiment while leaving room for larger abstractions.
- ``reinitialize_advantage_networks`` is False here vs True in the solver
  default. Brown et al. (2019) reinitialise each iteration; we deliberately
  do not, so the network can warm-start from the previous iteration. This is
  the choice we want to defend in the thesis.
- ``policy_network_train_steps`` and ``advantage_network_train_steps`` are
  set to 200 SGD steps per training session, which is much higher than the
  solver default of 1.

Anyone reading the solver in isolation should be aware that the experiment
deliberately deviates from the library defaults along these axes.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_multiseed_validation",
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
    "compute_exploitability": True,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
}

# Ten fixed seeds make the result reproducible and allow uncertainty reporting.
DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]
