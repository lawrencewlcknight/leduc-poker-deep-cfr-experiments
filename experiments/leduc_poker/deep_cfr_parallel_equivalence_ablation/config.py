"""Configuration for Experiment 25's Deep CFR parallel-equivalence ablation."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    BASE_COMPOSITE_HP_CONFIG,
)


DEFAULT_SEEDS = [1234, 2025, 31415]
SEQUENTIAL_VARIANT_ID = "composite_best_sequential"
PARALLEL_VARIANT_ID = "composite_best_ray_parallel"
PARALLEL_NUM_WORKERS = 3
PARALLEL_RAY_OBJECT_STORE_MEMORY = 512 * 1024 * 1024

# Pre-declared practical-equivalence margins. These are absolute differences
# in final metrics, not post-hoc tuning objectives.
FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN = 0.05
FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN = 0.02


VARIANTS = (
    {
        "variant_id": SEQUENTIAL_VARIANT_ID,
        "label": "Composite best sequential",
        "hp_family": "execution_backend",
        "hp_value": "sequential",
        "description": (
            "Current best Deep CFR configuration with the existing sequential "
            "external-sampling traversal loop."
        ),
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "uniform",
        "execution_backend": "sequential",
        "parallel_num_workers": 1,
    },
    {
        "variant_id": PARALLEL_VARIANT_ID,
        "label": "Composite best Ray parallel (3 workers)",
        "hp_family": "execution_backend",
        "hp_value": "ray_parallel_3_workers",
        "description": (
            "Current best Deep CFR configuration with traversal collection "
            "partitioned over three Ray actors and one central learner."
        ),
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "uniform",
        "execution_backend": "ray_parallel",
        "parallel_num_workers": PARALLEL_NUM_WORKERS,
    },
)


DEFAULT_CONFIG = deepcopy(BASE_COMPOSITE_HP_CONFIG)
DEFAULT_CONFIG.update({
    "experiment_name": "leduc_poker_deep_cfr_parallel_equivalence_ablation",
    "ablation_variants": VARIANTS,
    "baseline_variant_id": SEQUENTIAL_VARIANT_ID,
    "execution_backend": "sequential",
    "parallel_num_workers": PARALLEL_NUM_WORKERS,
    "parallel_ray_address": None,
    "parallel_log_to_driver": False,
    "parallel_ray_object_store_memory": PARALLEL_RAY_OBJECT_STORE_MEMORY,
    "parallel_worker_memory_capacity": None,
    "replay_buffer_type": "compact",
    "save_final_checkpoints": False,
})
