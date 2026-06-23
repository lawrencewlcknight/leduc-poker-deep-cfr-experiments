"""CLI for Experiment 17: dropout regularisation ablation."""

from __future__ import annotations

import sys

from experiments.leduc_poker.architecture_ablation_common import main_from_config

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, EXTENDED_SEEDS_5, EXTENDED_SEEDS_10


def main() -> int:
    return main_from_config(
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        description="Run the Leduc poker Deep CFR dropout ablation.",
        logger_name="deep_cfr_poker.experiment.dropout_ablation",
        plot_title_prefix="Dropout Ablation",
        metadata_extra={
            "default_seeds": DEFAULT_SEEDS,
            "extended_seeds_5": EXTENDED_SEEDS_5,
            "extended_seeds_10": EXTENDED_SEEDS_10,
            "experiment_note": (
                "Experiment 17 is a negative/control-style regularisation "
                "test. It holds the average-policy network fixed at the "
                "Experiment 1 2x32 MLP architecture and applies dropout only "
                "inside the advantage networks during supervised fitting. "
                "Regret-matching queries run with dropout disabled."
            ),
        },
    )


if __name__ == "__main__":
    sys.exit(main())
