"""Configuration for advantage-fitting HP search on the best baseline."""

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_ID,
    BASE_COMPOSITE_HP_CONFIG,
    make_variant,
)


DEFAULT_SEEDS = [1234, 2025, 31415]


ADVANTAGE_FITTING_VARIANTS = (
    dict(BASELINE_VARIANT),
    make_variant(
        "advantage_steps_100",
        "Advantage train steps 100",
        hp_family="advantage_fitting",
        hp_value="advantage_steps=100",
        description=(
            "Halves regret-approximator fitting effort, providing a compute "
            "and overfitting control against the 200-step baseline."
        ),
        advantage_network_train_steps=100,
    ),
    make_variant(
        "advantage_steps_400",
        "Advantage train steps 400",
        hp_family="advantage_fitting",
        hp_value="advantage_steps=400",
        description=(
            "Doubles regret-approximator fitting effort to test whether the "
            "deeper residual advantage networks remain under-optimised."
        ),
        advantage_network_train_steps=400,
    ),
    make_variant(
        "advantage_batch_512",
        "Advantage batch 512",
        hp_family="advantage_fitting",
        hp_value="advantage_batch=512",
        description=(
            "Uses smaller advantage minibatches, increasing stochasticity and "
            "testing whether gradient noise acts as useful regularisation."
        ),
        batch_size_advantage=512,
    ),
    make_variant(
        "advantage_batch_2048",
        "Advantage batch 2048",
        hp_family="advantage_fitting",
        hp_value="advantage_batch=2048",
        description=(
            "Uses larger advantage minibatches to test whether lower variance "
            "regret fitting improves exploitability."
        ),
        batch_size_advantage=2048,
    ),
    make_variant(
        "advantage_batch_2048_steps_400",
        "Advantage batch 2048 + steps 400",
        hp_family="advantage_fitting",
        hp_value="advantage_batch=2048;advantage_steps=400",
        description=(
            "Combines larger minibatches with more gradient steps, testing the "
            "targeted hypothesis that the deeper advantage networks benefit "
            "from more settled supervised optimisation."
        ),
        batch_size_advantage=2048,
        advantage_network_train_steps=400,
    ),
)

DEFAULT_CONFIG = {
    **BASE_COMPOSITE_HP_CONFIG,
    "experiment_name": "leduc_poker_deep_cfr_composite_advantage_fitting_ablation",
    "ablation_variants": ADVANTAGE_FITTING_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
}
