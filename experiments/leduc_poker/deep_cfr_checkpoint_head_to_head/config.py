"""Default configuration for the Leduc poker Deep CFR checkpoint head-to-head experiment.

The experiment trains one Deep CFR run per seed, saving a *policy snapshot* at
each milestone in :data:`CHECKPOINT_SCHEDULE`. The saved snapshots are then
played head-to-head against each other to test whether later checkpoints
consistently beat earlier ones.

The shared ``COMMON_SOLVER_KWARGS`` are intentionally identical to the
``deep_cfr_multiseed_validation`` experiment so that any difference in the
results is attributable to the schedule, not to a different solver
configuration. To compare against alternative solver settings, fork this file
into a separate experiment package rather than mutating this one.
"""

from deep_cfr_poker.constants import (
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)


# Filenames in this experiment's output directory follow the same snake_case
# convention used by the multiseed validation experiment. The run directory
# itself is timestamped, so file names do not need to embed the experiment
# name again.

CHECKPOINT_SCHEDULE = [100, 300, 500, 750, 1000, 1250, 1500]


# Three seeds are the default because each seed runs the full schedule (= the
# full 1,500-iteration training plus 7 saves). Override with --seeds for a
# faster smoke run or a larger final run.
DEFAULT_SEEDS = [1234, 2025, 31415]


# Per-stage policy_network_train_every. Match the notebook: train the policy
# network more frequently in the first stage so the very first checkpoint has
# a fitted average-policy network.
POLICY_TRAIN_EVERY_BY_TARGET = {
    100: 10,
    300: 100,
    500: 100,
    750: 100,
    1000: 100,
    1250: 100,
    1500: 100,
}


# Pairwise EVs whose magnitude is below this threshold are treated as
# practical ties when classifying improvement / regression in the
# monotonicity summary.
DEFAULT_EQUIVALENCE_EPSILON = 1e-3


DEFAULT_CONFIG = {
    "experiment_name": "leduc_poker_deep_cfr_checkpoint_head_to_head",
    "game_name": "leduc_poker",
    "checkpoint_schedule": tuple(CHECKPOINT_SCHEDULE),
    "policy_train_every_by_target": dict(POLICY_TRAIN_EVERY_BY_TARGET),
    # Solver kwargs shared with the validation experiment.
    "policy_network_layers": (32, 32),
    "advantage_network_layers": (32, 32),
    "num_traversals": 320,
    "learning_rate": 0.003,
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "memory_capacity": int(1e7),
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 200,
    "advantage_network_train_steps": 200,
    "compute_exploitability": True,
    # Leduc replay buffers are large enough that serializing and reloading them
    # between every milestone can destabilize cloud VMs. Milestone checkpoints
    # therefore keep the model/optimizer/training state by default, while the
    # in-process solver carries replay buffers continuously through the stages.
    "save_replay_buffers_in_checkpoints": False,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
    # Analysis options.
    "equivalence_epsilon": DEFAULT_EQUIVALENCE_EPSILON,
    "run_monte_carlo_validation": False,
    "num_mc_episodes": 20_000,
    "mc_seed": 12345,
    "mc_alternate_seats": True,
    # By default we evaluate adjacent and milestone pairs only when MC is on,
    # to keep the cost manageable. Allowed values: "adjacent", "milestones",
    # "all_pairs".
    "mc_pair_mode": "adjacent",
    # Heatmap annotation toggle.
    "annotate_heatmap": True,
}
