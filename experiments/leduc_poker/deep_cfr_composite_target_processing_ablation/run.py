"""CLI for target processing on the composite Leduc Deep CFR baseline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/deep_cfr_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/deep_cfr_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from tqdm import tqdm

from deep_cfr_poker.experiment_utils import (
    DEFAULT_FINAL_WINDOW,
    configure_run_logging,
    create_run_dir,
    run_single_seed,
)
from experiments.leduc_poker.deep_cfr_target_processing_ablation.plotting import (
    plot_target_processing_ablation,
)
from experiments.leduc_poker.deep_cfr_target_processing_ablation.run import (
    _augment_result,
    _filter_variants,
    _str2bool,
    _variant_config,
    export_ablation_results,
    parse_int_tuple,
    parse_variant_ids,
)

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, TARGET_PROCESSING_VARIANTS


_LOGGER = logging.getLogger(
    "deep_cfr_poker.experiment.composite_target_processing"
)


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    variants = _filter_variants(
        TARGET_PROCESSING_VARIANTS, parse_variant_ids(args.variant_ids)
    )
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "evaluation_interval": args.evaluation_interval,
        "policy_network_layers": parse_int_tuple(args.policy_network_layers),
        "advantage_network_layers": parse_int_tuple(args.advantage_network_layers),
        "learning_rate": args.learning_rate,
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "memory_capacity": args.memory_capacity,
        "reinitialize_advantage_networks": args.reinitialize_advantage_networks,
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "policy_network_train_every": args.policy_network_train_every,
        "compute_exploitability": args.compute_exploitability,
        "target_clip_value": args.target_clip_value,
        "target_standardize_epsilon": args.target_standardize_epsilon,
        "baseline_variant_id": args.baseline_variant_id,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.target_clip_value is not None:
        for variant in variants:
            variant["target_clip_value"] = float(args.target_clip_value)
    config["ablation_variants"] = tuple(variants)
    variant_ids = {str(v["variant_id"]) for v in variants}
    if str(config["baseline_variant_id"]) not in variant_ids:
        config["baseline_variant_id"] = str(variants[0]["variant_id"])
    return config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run target-processing variants on the composite Leduc Deep CFR "
            "architecture."
        )
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--variant-ids", default=None, help="Comma-separated subset of variant ids.")
    parser.add_argument("--baseline-variant-id", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--advantage-network-layers", default=None)
    parser.add_argument("--reinitialize-advantage-networks", type=_str2bool, default=None)
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--target-clip-value", type=float, default=None)
    parser.add_argument("--target-standardize-epsilon", type=float, default=None)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--save-final-checkpoints", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    config = build_config(args)
    seeds = parse_seeds(args.seeds)

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(Path(args.output_root), config["experiment_name"])

    configure_run_logging(run_dir, verbose=args.verbose)
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Configuration: %s", config)
    _LOGGER.info("Seeds: %s", seeds)

    results = []
    failed = []
    for variant in config["ablation_variants"]:
        variant_config = _variant_config(config, variant)
        _LOGGER.info(
            "Running %s: %s",
            variant_config["variant_id"],
            variant_config.get("label", variant_config["variant_id"]),
        )
        for seed in tqdm(seeds, desc=str(variant_config["variant_id"])):
            try:
                result = run_single_seed(
                    seed,
                    variant_config,
                    export_dir=run_dir,
                    save_final_checkpoint=args.save_final_checkpoints,
                    final_window=args.final_window,
                )
                results.append(
                    _augment_result(
                        result,
                        variant_config,
                        args.final_window,
                        config["exploitability_threshold"],
                    )
                )
            except Exception as exc:  # pragma: no cover
                _LOGGER.exception(
                    "Seed %s failed for variant %s: %s",
                    seed,
                    variant_config["variant_id"],
                    exc,
                )
                failed.append(
                    {
                        "seed": int(seed),
                        "variant_id": str(variant_config["variant_id"]),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    if not results:
        _LOGGER.error("All runs failed; nothing to export.")
        return 1

    export_info = export_ablation_results(
        results, run_dir, config, seeds, failed=failed or None
    )
    plot_target_processing_ablation(
        results,
        run_dir,
        variants=config["ablation_variants"],
        baseline_variant_id=str(config["baseline_variant_id"]),
        exploitability_threshold=float(config["exploitability_threshold"]),
        average_policy_value_target=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
        aggregate_by_variant=export_info["aggregate_by_variant"],
        paired_rows=export_info["paired_rows"],
    )

    _LOGGER.info(
        "Completed %d/%d runs",
        len(results),
        len(seeds) * len(config["ablation_variants"]),
    )
    if failed:
        _LOGGER.warning("%d run(s) failed; see failed_seeds.json", len(failed))
    _LOGGER.info("Per-seed summary: %s", export_info["summary_csv"].resolve())
    _LOGGER.info("Checkpoint curves: %s", export_info["curve_csv"].resolve())
    _LOGGER.info("Aggregate summary: %s", export_info["aggregate_summary"].resolve())
    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
