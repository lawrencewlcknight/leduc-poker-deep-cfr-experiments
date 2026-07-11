"""CLI for constant learning-rate HP search on the best Leduc baseline."""

from __future__ import annotations

import sys

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    build_config_from_args,
    run_experiment_from_cli,
)

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, LEARNING_RATE_VARIANTS


def build_config(args) -> dict:
    return build_config_from_args(
        args,
        default_config=DEFAULT_CONFIG,
        all_variants=LEARNING_RATE_VARIANTS,
    )


def main() -> int:
    return run_experiment_from_cli(
        default_config=DEFAULT_CONFIG,
        all_variants=LEARNING_RATE_VARIANTS,
        default_seeds=DEFAULT_SEEDS,
        logger_name="deep_cfr_poker.experiment.composite_learning_rate",
        description=(
            "Run constant learning-rate magnitude variants on the current best "
            "Leduc Deep CFR baseline."
        ),
        experiment_note=(
            "Controlled HP ablation varying only constant optimiser learning "
            "rate on the composite, standardised, uniform-average baseline. "
            "This is deliberately separate from the previous schedule ablation."
        ),
        plot_title_prefix="Learning-Rate HP Ablation",
    )


if __name__ == "__main__":
    sys.exit(main())
