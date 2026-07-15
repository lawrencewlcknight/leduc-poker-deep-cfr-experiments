"""Configuration for constant learning-rate HP search on the best baseline."""

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_ID,
    BASE_COMPOSITE_HP_CONFIG,
    make_variant,
)


DEFAULT_SEEDS = [1234, 2025, 31415]


LEARNING_RATE_VARIANTS = (
    dict(BASELINE_VARIANT),
    make_variant(
        "learning_rate_0_001",
        "Learning rate 0.001",
        hp_family="learning_rate",
        hp_value="learning_rate=0.001",
        description=(
            "Tests the lower learning-rate value that appeared in the earlier "
            "constrained random search, now isolated on the stronger baseline."
        ),
        learning_rate=0.001,
    ),
    make_variant(
        "learning_rate_0_0015",
        "Learning rate 0.0015",
        hp_family="learning_rate",
        hp_value="learning_rate=0.0015",
        description=(
            "Tests a milder reduction in optimiser step size than 0.001, "
            "aiming to stabilise fitting without slowing adaptation as much."
        ),
        learning_rate=0.0015,
    ),
    make_variant(
        "learning_rate_0_002",
        "Learning rate 0.002",
        hp_family="learning_rate",
        hp_value="learning_rate=0.002",
        description=(
            "Tests a near-baseline reduction in step size that may improve "
            "late-training stability with limited convergence cost."
        ),
        learning_rate=0.002,
    ),
    make_variant(
        "learning_rate_0_004",
        "Learning rate 0.004",
        hp_family="learning_rate",
        hp_value="learning_rate=0.004",
        description=(
            "Tests a small increase in constant learning rate, checking whether "
            "the stronger architecture can exploit faster supervised fitting."
        ),
        learning_rate=0.004,
    ),
)

DEFAULT_CONFIG = {
    **BASE_COMPOSITE_HP_CONFIG,
    "experiment_name": "leduc_poker_deep_cfr_composite_learning_rate_ablation",
    "ablation_variants": LEARNING_RATE_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
}
