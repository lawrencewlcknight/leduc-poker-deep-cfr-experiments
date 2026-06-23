"""CLI for Experiment 13: layer normalisation and residual-LN MLPs."""

from __future__ import annotations

import sys

from experiments.leduc_poker.architecture_ablation_common import main_from_config

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, EXTENDED_SEEDS_5, EXTENDED_SEEDS_10


def main() -> int:
    return main_from_config(
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        description="Run the Leduc poker Deep CFR layer-normalisation ablation.",
        logger_name="deep_cfr_poker.experiment.layer_norm_network",
        plot_title_prefix="Layer-Normalisation Network Ablation",
        metadata_extra={
            "default_seeds": DEFAULT_SEEDS,
            "extended_seeds_5": EXTENDED_SEEDS_5,
            "extended_seeds_10": EXTENDED_SEEDS_10,
            "experiment_note": (
                "Experiment 13 compares plain MLPs, layer-normalised MLPs, "
                "and residual layer-normalised MLPs at fixed width 32 and "
                "depths 2, 4, and 8."
            ),
        },
    )


if __name__ == "__main__":
    sys.exit(main())

