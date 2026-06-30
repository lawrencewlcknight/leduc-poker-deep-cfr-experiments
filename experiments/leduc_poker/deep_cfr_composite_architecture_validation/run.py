"""CLI for validating the composite Leduc Deep CFR architecture."""

from __future__ import annotations

import sys

from experiments.leduc_poker.architecture_ablation_common import main_from_config

from .config import (
    CENTERED_ADVANTAGE_VARIANT_ID,
    COMPOSITE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXTENDED_SEEDS_10,
)


def main() -> int:
    return main_from_config(
        default_config=DEFAULT_CONFIG,
        default_seeds=DEFAULT_SEEDS,
        description="Run the Leduc poker Deep CFR composite architecture validation.",
        logger_name="deep_cfr_poker.experiment.composite_architecture_validation",
        plot_title_prefix="Composite Architecture Validation",
        metadata_extra={
            "default_seeds": DEFAULT_SEEDS,
            "extended_seeds_10": EXTENDED_SEEDS_10,
            "centered_advantage_variant_id": CENTERED_ADVANTAGE_VARIANT_ID,
            "composite_variant_id": COMPOSITE_VARIANT_ID,
            "experiment_note": (
                "This validation keeps the average-policy network at the "
                "baseline 2x32 MLP and compares the original direct 2x32 "
                "advantage baseline with the strongest centred-head 8x32 "
                "single intervention and the proposed composite 8x32 "
                "residual LayerNorm centred-advantage architecture."
            ),
        },
    )


if __name__ == "__main__":
    sys.exit(main())
