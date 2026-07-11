"""CLI for policy-extraction HP search on the best Leduc Deep CFR baseline."""

from __future__ import annotations

import sys

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    build_config_from_args,
    run_experiment_from_cli,
)

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, POLICY_EXTRACTION_VARIANTS


def build_config(args) -> dict:
    return build_config_from_args(
        args,
        default_config=DEFAULT_CONFIG,
        all_variants=POLICY_EXTRACTION_VARIANTS,
    )


def main() -> int:
    return run_experiment_from_cli(
        default_config=DEFAULT_CONFIG,
        all_variants=POLICY_EXTRACTION_VARIANTS,
        default_seeds=DEFAULT_SEEDS,
        logger_name="deep_cfr_poker.experiment.composite_policy_extraction",
        description=(
            "Run policy-extraction hyperparameter variants on the current best "
            "Leduc Deep CFR baseline."
        ),
        experiment_note=(
            "Controlled HP ablation varying only average-policy extraction "
            "settings on the composite, standardised, uniform-average baseline."
        ),
        plot_title_prefix="Policy-Extraction HP Ablation",
    )


if __name__ == "__main__":
    sys.exit(main())
