"""Configuration for replay-memory HP search on the best baseline."""

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_ID,
    BASE_COMPOSITE_HP_CONFIG,
    DEFAULT_SEEDS,
    DEFAULT_SEEDS_5,
    EXTENDED_SEEDS_10,
    make_variant,
)


REPLAY_MEMORY_VARIANTS = (
    dict(BASELINE_VARIANT),
    make_variant(
        "memory_capacity_1m",
        "Replay memory 1M",
        hp_family="replay_memory",
        hp_value="memory_capacity=1000000",
        description=(
            "Reduces reservoir capacity to test whether fresher regret and "
            "strategy samples improve the final average policy."
        ),
        memory_capacity=int(1e6),
    ),
    make_variant(
        "memory_capacity_2m",
        "Replay memory 2M",
        hp_family="replay_memory",
        hp_value="memory_capacity=2000000",
        description=(
            "Intermediate-capacity freshness test between the aggressive 1M "
            "setting and the larger baseline buffer."
        ),
        memory_capacity=int(2e6),
    ),
    make_variant(
        "memory_capacity_5m",
        "Replay memory 5M",
        hp_family="replay_memory",
        hp_value="memory_capacity=5000000",
        description=(
            "Moderate-capacity replay test, preserving more historical coverage "
            "while still increasing sample turnover relative to 10M."
        ),
        memory_capacity=int(5e6),
    ),
)

DEFAULT_CONFIG = {
    **BASE_COMPOSITE_HP_CONFIG,
    "experiment_name": "leduc_poker_deep_cfr_composite_replay_memory_ablation",
    "ablation_variants": REPLAY_MEMORY_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
}
