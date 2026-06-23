"""CLI for Experiment 16: factorised advantage-output heads."""

from __future__ import annotations

import sys

from experiments.leduc_poker.architecture_ablation_common import main_from_config

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, EXTENDED_SEEDS_5, EXTENDED_SEEDS_10


def main() -> int:
    return main_from_config(
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        description="Run the Leduc poker Deep CFR factorised advantage-head ablation.",
        logger_name="deep_cfr_poker.experiment.factorised_advantage_head",
        plot_title_prefix="Factorised Advantage-Head Ablation",
        metadata_extra={
            "default_seeds": DEFAULT_SEEDS,
            "extended_seeds_5": EXTENDED_SEEDS_5,
            "extended_seeds_10": EXTENDED_SEEDS_10,
            "experiment_note": (
                "Experiment 16 holds the average-policy network fixed at the "
                "Experiment 1 2x32 MLP architecture and compares direct "
                "advantage outputs, mean-centred action-advantage outputs, "
                "and dueling-style value-plus-centred-action heads at depths "
                "2, 4, and 8."
            ),
        },
    )


if __name__ == "__main__":
    sys.exit(main())
