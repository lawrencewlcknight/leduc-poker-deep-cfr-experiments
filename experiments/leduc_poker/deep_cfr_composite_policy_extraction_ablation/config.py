"""Configuration for policy-extraction HP search on the best baseline."""

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_ID,
    BASE_COMPOSITE_HP_CONFIG,
    DEFAULT_SEEDS,
    DEFAULT_SEEDS_5,
    EXTENDED_SEEDS_10,
    make_variant,
)


POLICY_EXTRACTION_VARIANTS = (
    dict(BASELINE_VARIANT),
    make_variant(
        "policy_train_every_10",
        "Policy train every 10 iterations",
        hp_family="policy_extraction",
        hp_value="train_every=10",
        description=(
            "More frequent average-policy fitting, testing whether the uniform "
            "average-policy target benefits from lower lag behind the evolving "
            "regret-induced strategy distribution."
        ),
        policy_network_train_every=10,
    ),
    make_variant(
        "policy_train_every_50",
        "Policy train every 50 iterations",
        hp_family="policy_extraction",
        hp_value="train_every=50",
        description=(
            "Less frequent average-policy fitting, testing whether larger "
            "policy-buffer refresh intervals reduce optimisation churn without "
            "hurting final strategy extraction."
        ),
        policy_network_train_every=50,
    ),
    make_variant(
        "policy_steps_400",
        "Policy train steps 400",
        hp_family="policy_extraction",
        hp_value="policy_steps=400",
        description=(
            "Doubles the supervised optimisation effort per average-policy "
            "training event while leaving the regret approximators unchanged."
        ),
        policy_network_train_steps=400,
    ),
    make_variant(
        "strategy_batch_2048",
        "Strategy batch 2048",
        hp_family="policy_extraction",
        hp_value="strategy_batch=2048",
        description=(
            "Uses larger average-policy minibatches to test whether lower "
            "gradient noise improves extraction from the uniform strategy "
            "replay distribution."
        ),
        batch_size_strategy=2048,
    ),
)

DEFAULT_CONFIG = {
    **BASE_COMPOSITE_HP_CONFIG,
    "experiment_name": "leduc_poker_deep_cfr_composite_policy_extraction_ablation",
    "ablation_variants": POLICY_EXTRACTION_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
}
