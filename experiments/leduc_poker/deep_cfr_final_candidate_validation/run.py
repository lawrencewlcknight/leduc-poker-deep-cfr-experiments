"""CLI for the final Leduc Deep CFR candidate validation."""

from __future__ import annotations

import sys

from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (
    build_config_from_args,
    run_experiment_from_cli,
)

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, FINAL_CANDIDATE_VARIANTS


def build_config(args) -> dict:
    return build_config_from_args(
        args,
        default_config=DEFAULT_CONFIG,
        all_variants=FINAL_CANDIDATE_VARIANTS,
    )


def main() -> int:
    return run_experiment_from_cli(
        default_config=DEFAULT_CONFIG,
        all_variants=FINAL_CANDIDATE_VARIANTS,
        default_seeds=DEFAULT_SEEDS,
        logger_name="deep_cfr_poker.experiment.final_candidate_validation",
        description=(
            "Run the final candidate Leduc Deep CFR configuration against the "
            "previous best composite baseline."
        ),
        experiment_note=(
            "Final candidate validation over a reduced 1050-iteration budget "
            "chosen to target roughly 15M environment nodes touched. The "
            "candidate combines the independently selected training-parameter "
            "changes: policy extraction every 10 iterations, advantage batch "
            "2048, replay memory 5M, and constant learning rate 0.004."
        ),
        plot_title_prefix="Final Candidate Validation",
    )


if __name__ == "__main__":
    sys.exit(main())
