"""CLI for average weighting on the standardised composite baseline."""

from __future__ import annotations

import argparse
import json
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

from tqdm import tqdm  # noqa: E402

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    DEFAULT_FINAL_WINDOW,
    configure_run_logging,
    create_run_dir,
    export_results,
    json_safe,
    run_single_seed,
    write_dict_rows_csv,
    write_experiment_metadata,
    write_failed_seeds,
)
from experiments.leduc_poker.deep_cfr_replay_averaging_ablation.plotting import (  # noqa: E402
    plot_replay_averaging_ablation,
)
from experiments.leduc_poker.deep_cfr_replay_averaging_ablation.run import (  # noqa: E402
    _augment_result as _augment_replay_result,
    _export_ablation_npz,
    _filter_variants,
    _group_by_variant,
    _paired_rows,
    _paired_summary,
    _str2bool,
    _variant_config,
    parse_int_tuple,
    parse_variant_ids,
)

from .config import (  # noqa: E402
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEFAULT_SEEDS_5,
    EXTENDED_SEEDS_10,
    REPLAY_AVERAGING_VARIANTS,
)


_LOGGER = logging.getLogger(
    "deep_cfr_poker.experiment.composite_standardized_replay_averaging"
)


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    variant_ids = parse_variant_ids(args.variant_ids)
    variants = (
        _filter_variants(config["ablation_variants"], None)
        if variant_ids is None
        else _filter_variants(REPLAY_AVERAGING_VARIANTS, variant_ids)
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
        "target_standardize_epsilon": args.target_standardize_epsilon,
        "priority_alpha": args.priority_alpha,
        "priority_epsilon": args.priority_epsilon,
        "baseline_variant_id": args.baseline_variant_id,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["target_processing"] = "standardize"
    config["ablation_variants"] = tuple(variants)
    variant_ids = {str(v["variant_id"]) for v in variants}
    if str(config["baseline_variant_id"]) not in variant_ids:
        config["baseline_variant_id"] = str(variants[0]["variant_id"])
    return config


def _augment_result(
    result: dict,
    variant_config: dict,
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    result = _augment_replay_result(
        result,
        variant_config,
        final_window=final_window,
        exploitability_threshold=exploitability_threshold,
    )
    result["target_processing"] = str(variant_config.get("target_processing", ""))
    result["target_clip_value"] = float(variant_config.get("target_clip_value", 1.0))
    result["summary"] = {
        "target_processing": result["target_processing"],
        "target_clip_value": result["target_clip_value"],
        "target_standardize_epsilon": float(
            variant_config.get("target_standardize_epsilon", 1e-6)
        ),
        **result["summary"],
    }
    return result


def export_ablation_results(
    results,
    run_dir: Path,
    config: dict,
    seeds,
    failed=None,
) -> dict:
    variants = list(config["ablation_variants"])
    info = export_results(
        results,
        run_dir,
        config,
        seeds,
        failed_seeds=failed,
        write_multiseed_npz=False,
    )
    summary_rows = [result["summary"] for result in results]
    aggregate_by_variant = {"by_variant_id": _group_by_variant(summary_rows, variants)}
    aggregate_path = run_dir / "aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate_by_variant), f, indent=2)

    paired_rows = _paired_rows(results, str(config["baseline_variant_id"]))
    paired_csv = None
    paired_summary_path = None
    if paired_rows:
        paired_csv = write_dict_rows_csv(
            paired_rows, run_dir / "paired_differences_vs_baseline.csv"
        )
        paired_summary = _paired_summary(paired_rows)
        paired_summary_path = run_dir / "paired_difference_summary.json"
        with open(paired_summary_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(paired_summary), f, indent=2)

    npz_path = _export_ablation_npz(results, run_dir, variants)
    write_experiment_metadata(
        run_dir,
        config=config,
        seeds=seeds,
        completed_seeds=sorted({int(r["seed"]) for r in results}),
        extra={
            "default_seeds_5": DEFAULT_SEEDS_5,
            "extended_seeds_10": EXTENDED_SEEDS_10,
            "experiment_note": (
                "Controlled average-strategy weighting ablation on the "
                "composite architecture with standardised advantage targets. "
                "Paired differences are variant minus the uniform-replay, "
                "linear-average standardised composite baseline."
            ),
            "aggregate_summary_json": str(aggregate_path),
            "paired_differences_csv": str(paired_csv) if paired_csv else None,
            "paired_difference_summary_json": (
                str(paired_summary_path) if paired_summary_path else None
            ),
            "ablation_curves_npz": str(npz_path),
        },
    )
    if failed:
        write_failed_seeds(run_dir, failed)

    return {
        **info,
        "aggregate_summary": aggregate_path,
        "paired_differences_csv": paired_csv,
        "paired_difference_summary": paired_summary_path,
        "ablation_curves_npz": npz_path,
        "aggregate_by_variant": aggregate_by_variant,
        "paired_rows": paired_rows,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run average-strategy weighting variants on the composite Leduc "
            "Deep CFR baseline with standardised advantage targets."
        )
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--variant-ids", default=None, help="Comma-separated variant ids.")
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
    parser.add_argument("--target-standardize-epsilon", type=float, default=None)
    parser.add_argument("--priority-alpha", type=float, default=None)
    parser.add_argument("--priority-epsilon", type=float, default=None)
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
    plot_replay_averaging_ablation(
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
