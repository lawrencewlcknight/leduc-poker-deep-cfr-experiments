"""CLI for Experiment 15: shared advantage trunk with player/action heads."""

from __future__ import annotations

import sys

from experiments.leduc_poker.architecture_ablation_common import main_from_config

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, EXTENDED_SEEDS_5, EXTENDED_SEEDS_10


def main() -> int:
    return main_from_config(
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        description="Run the Leduc poker Deep CFR shared-trunk head ablation.",
        logger_name="deep_cfr_poker.experiment.shared_trunk_head",
        plot_title_prefix="Shared-Trunk Head Ablation",
        metadata_extra={
            "default_seeds": DEFAULT_SEEDS,
            "extended_seeds_5": EXTENDED_SEEDS_5,
            "extended_seeds_10": EXTENDED_SEEDS_10,
            "experiment_note": (
                "Experiment 15 holds the average-policy network fixed at the "
                "Experiment 1 2x32 MLP architecture and compares independent "
                "per-player advantage MLPs with a shared advantage trunk and "
                "separate player/action heads at depths 2, 4, and 8."
            ),
        },
    )


if __name__ == "__main__":
    sys.exit(main())
