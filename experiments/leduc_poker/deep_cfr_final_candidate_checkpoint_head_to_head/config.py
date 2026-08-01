"""Configuration for Experiment 27: final-candidate checkpoint head-to-head."""

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    BASE_COMPOSITE_HP_CONFIG,
    DEFAULT_SEEDS_5,
)
from experiments.leduc_poker.deep_cfr_final_candidate_validation.config import (
    FINAL_CANDIDATE_VARIANTS,
    TARGET_NUM_ITERATIONS,
    TARGET_NUM_TRAVERSALS,
)


DEFAULT_SEEDS = DEFAULT_SEEDS_5

# Five policy snapshots at 20%, 40%, 60%, 80%, and 100% of the final
# 1,050-iteration budget. Empirically these correspond to approximately
# 3M, 6M, 9M, 12M, and 15M environment nodes touched.
CHECKPOINT_SCHEDULE = (210, 420, 630, 840, 1050)

_FINAL_CANDIDATE_OVERRIDES = dict(FINAL_CANDIDATE_VARIANTS[1])

DEFAULT_CONFIG = {
    **BASE_COMPOSITE_HP_CONFIG,
    **_FINAL_CANDIDATE_OVERRIDES,
    "experiment_name": "leduc_poker_deep_cfr_final_candidate_checkpoint_head_to_head",
    "num_iterations": TARGET_NUM_ITERATIONS,
    "num_traversals": TARGET_NUM_TRAVERSALS,
    "checkpoint_schedule": CHECKPOINT_SCHEDULE,
    "equivalence_epsilon": 1e-3,
    "temporal_x_axis": "nodes_touched",
    "require_complete_checkpoint_schedule": True,
    "annotate_heatmap": True,
    # Leduc permits exact expected-value evaluation, so Monte Carlo match noise
    # is unnecessary. Independent training seeds are the inferential unit.
    "run_monte_carlo_validation": False,
    "num_mc_episodes": 0,
}


__all__ = ["CHECKPOINT_SCHEDULE", "DEFAULT_CONFIG", "DEFAULT_SEEDS"]
