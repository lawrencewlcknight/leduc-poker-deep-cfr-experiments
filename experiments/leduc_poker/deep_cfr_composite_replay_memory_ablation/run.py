"""CLI for replay-memory HP search on the best Leduc Deep CFR baseline."""

from __future__ import annotations

import sys

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    build_config_from_args,
    run_experiment_from_cli,
)

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, REPLAY_MEMORY_VARIANTS


def build_config(args) -> dict:
    return build_config_from_args(
        args,
        default_config=DEFAULT_CONFIG,
        all_variants=REPLAY_MEMORY_VARIANTS,
    )


def main() -> int:
    return run_experiment_from_cli(
        default_config=DEFAULT_CONFIG,
        all_variants=REPLAY_MEMORY_VARIANTS,
        default_seeds=DEFAULT_SEEDS,
        logger_name="deep_cfr_poker.experiment.composite_replay_memory",
        description=(
            "Run replay-memory hyperparameter variants on the current best "
            "Leduc Deep CFR baseline."
        ),
        experiment_note=(
            "Controlled HP ablation varying only replay reservoir capacity on "
            "the composite, standardised, uniform-average baseline."
        ),
        plot_title_prefix="Replay-Memory HP Ablation",
    )


if __name__ == "__main__":
    sys.exit(main())
