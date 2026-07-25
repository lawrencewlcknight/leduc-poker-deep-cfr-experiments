"""Configuration for the final Leduc Deep CFR candidate validation."""

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    BASELINE_VARIANT,
    BASELINE_VARIANT_ID,
    BASE_COMPOSITE_HP_CONFIG,
    DEFAULT_SEEDS_5,
    EXTENDED_SEEDS_10,
    make_variant,
)


DEFAULT_SEEDS = DEFAULT_SEEDS_5

# Previous 1500-iteration composite runs touched approximately 21.3M nodes.
# Scaling the same traversal budget to 1050 iterations targets about 15M nodes.
TARGET_NUM_ITERATIONS = 1050
TARGET_NUM_TRAVERSALS = 320

FINAL_CANDIDATE_VARIANT_ID = "final_candidate_policy10_advbatch2048_memory5m_lr004"


FINAL_CANDIDATE_VARIANTS = (
    dict(BASELINE_VARIANT),
    make_variant(
        FINAL_CANDIDATE_VARIANT_ID,
        "Final candidate configuration",
        hp_family="final_candidate",
        hp_value="policy10_advbatch2048_memory5m_lr004",
        description=(
            "Cumulative candidate selected from the targeted training-parameter "
            "ablations: average-policy fitting every 10 iterations, advantage "
            "minibatch 2048, replay capacity 5M, and constant learning rate "
            "0.004, with the residual LayerNorm centred-advantage architecture, "
            "standardised targets, uniform advantage replay, and uniform "
            "average-strategy weighting retained."
        ),
        policy_network_train_every=10,
        batch_size_advantage=2048,
        memory_capacity=int(5e6),
        learning_rate=0.004,
    ),
)


DEFAULT_CONFIG = {
    **BASE_COMPOSITE_HP_CONFIG,
    "experiment_name": "leduc_poker_deep_cfr_final_candidate_validation",
    "num_iterations": TARGET_NUM_ITERATIONS,
    "num_traversals": TARGET_NUM_TRAVERSALS,
    "ablation_variants": FINAL_CANDIDATE_VARIANTS,
    "baseline_variant_id": BASELINE_VARIANT_ID,
}


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "EXTENDED_SEEDS_10",
    "FINAL_CANDIDATE_VARIANT_ID",
    "FINAL_CANDIDATE_VARIANTS",
    "TARGET_NUM_ITERATIONS",
    "TARGET_NUM_TRAVERSALS",
]
