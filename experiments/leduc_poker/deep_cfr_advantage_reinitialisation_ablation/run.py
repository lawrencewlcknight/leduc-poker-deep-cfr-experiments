"""CLI for the Leduc poker Deep CFR advantage-reinitialisation ablation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats

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
    final_window_std,
    json_safe,
    normalised_auc,
    run_single_seed,
    summarise_numeric_fields,
    write_dict_rows_csv,
    write_experiment_metadata,
    write_failed_seeds,
)

from .config import (  # noqa: E402
    ABLATION_VARIANTS,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEFAULT_SEEDS_5,
    SMOKE_TEST_SEEDS,
)
from .plotting import plot_advantage_reinitialisation_ablation  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.advantage_reinitialisation")


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_variant_ids(value: Optional[str]):
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _filter_variants(variants: Sequence[Mapping[str, object]], ids: Optional[Sequence[str]]):
    if ids is None:
        return [dict(v) for v in variants]
    by_id = {str(v["variant_id"]): dict(v) for v in variants}
    missing = [variant_id for variant_id in ids if variant_id not in by_id]
    if missing:
        raise ValueError(f"Unknown variant id(s): {missing}")
    return [by_id[variant_id] for variant_id in ids]


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    variants = _filter_variants(ABLATION_VARIANTS, parse_variant_ids(args.variant_ids))
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
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "policy_network_train_every": args.policy_network_train_every,
        "compute_exploitability": args.compute_exploitability,
        "reference_variant_id": args.reference_variant_id,
        "comparison_variant_id": args.comparison_variant_id,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["ablation_variants"] = tuple(variants)
    variant_ids = {str(v["variant_id"]) for v in variants}
    if str(config["reference_variant_id"]) not in variant_ids:
        config["reference_variant_id"] = str(variants[0]["variant_id"])
    if str(config["comparison_variant_id"]) not in variant_ids:
        config["comparison_variant_id"] = str(variants[-1]["variant_id"])
    return config


def _variant_config(base_config: dict, variant: Mapping[str, object]) -> dict:
    config = deepcopy(base_config)
    config.update(dict(variant))
    config["reinitialize_advantage_networks"] = bool(
        variant["reinitialize_advantage_networks"]
    )
    return config


def _best_exploitability_iteration(iterations, exploitability) -> float:
    values = np.asarray(exploitability, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    finite_indices = np.where(finite)[0]
    local = int(np.nanargmin(values[finite]))
    return int(np.asarray(iterations)[finite_indices[local]])


def _augment_result(
    result: dict,
    variant_config: Mapping[str, object],
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    variant_id = str(variant_config["variant_id"])
    result["variant_id"] = variant_id
    result["variant_label"] = str(variant_config["label"])
    result["policy_training_mode"] = str(variant_config.get("policy_training_mode", "intermittent"))
    result["reinitialize_advantage_networks"] = bool(
        variant_config["reinitialize_advantage_networks"]
    )
    result["policy_network_train_every"] = int(variant_config["policy_network_train_every"])

    exploitability_curve = np.asarray(result["exploitability"], dtype=np.float64)
    finite_exploitability = exploitability_curve[np.isfinite(exploitability_curve)]
    fraction_below_threshold = (
        float(np.mean(finite_exploitability <= float(exploitability_threshold)))
        if finite_exploitability.size
        else float("nan")
    )

    variant_fields = {
        "variant_id": variant_id,
        "variant_label": str(variant_config["label"]),
        "reinitialize_advantage_networks": bool(
            variant_config["reinitialize_advantage_networks"]
        ),
    }
    metric_fields = {
        "best_exploitability_iteration": _best_exploitability_iteration(
            result["iterations"], result["exploitability"]
        ),
        "final_window_std_exploitability": final_window_std(
            result["exploitability"], window=final_window
        ),
        "exploitability_auc_by_iteration": normalised_auc(
            result["iterations"], result["exploitability"]
        ),
        "exploitability_auc_by_nodes": normalised_auc(
            result["nodes_touched"], result["exploitability"]
        ),
        "fraction_checkpoints_below_threshold": fraction_below_threshold,
        "nodes_to_threshold": result["summary"]["nodes_to_exploitability_threshold"],
        "seconds_to_threshold": result["summary"]["seconds_to_exploitability_threshold"],
        "final_average_policy_value": result["summary"]["final_policy_value"],
        "final_advantage_target_variance": result["summary"]["final_advantage_target_variance"],
        "final_policy_entropy": result["summary"]["final_policy_normalized_entropy_mean"],
    }
    result["summary"] = {
        **variant_fields,
        **dict(result["summary"]),
        **metric_fields,
    }
    return result


def _group_by_variant(rows: Sequence[dict], variants: Sequence[Mapping[str, object]]):
    grouped = {}
    for variant in variants:
        variant_id = str(variant["variant_id"])
        variant_rows = [row for row in rows if row["variant_id"] == variant_id]
        grouped[variant_id] = summarise_numeric_fields(variant_rows)
    return grouped


def _paired_rows(
    results: Sequence[dict],
    *,
    reference_variant_id: str,
    comparison_variant_id: str,
) -> List[dict]:
    by_variant_seed = {
        (str(r["variant_id"]), int(r["seed"])): r["summary"]
        for r in results
    }
    seeds = sorted({int(r["seed"]) for r in results})
    rows = []
    for seed in seeds:
        reference = by_variant_seed.get((str(reference_variant_id), seed))
        comparison = by_variant_seed.get((str(comparison_variant_id), seed))
        if reference is None or comparison is None:
            continue
        rows.append(
            {
                "seed": int(seed),
                "reference_variant_id": str(reference_variant_id),
                "comparison_variant_id": str(comparison_variant_id),
                "delta_final_exploitability_true_minus_false": float(
                    comparison["final_exploitability"]
                    - reference["final_exploitability"]
                ),
                "delta_best_exploitability_true_minus_false": float(
                    comparison["best_exploitability"]
                    - reference["best_exploitability"]
                ),
                "delta_final_window_mean_true_minus_false": float(
                    comparison["final_window_mean_exploitability"]
                    - reference["final_window_mean_exploitability"]
                ),
                "delta_policy_value_error_true_minus_false": float(
                    comparison["final_policy_value_error"]
                    - reference["final_policy_value_error"]
                ),
                "delta_auc_by_iteration_true_minus_false": float(
                    comparison["exploitability_auc_by_iteration"]
                    - reference["exploitability_auc_by_iteration"]
                ),
            }
        )
    return rows


def _paired_summary(rows: Sequence[dict]) -> dict:
    output = {}
    if not rows:
        return output
    for field in rows[0].keys():
        if field in {"seed", "reference_variant_id", "comparison_variant_id"}:
            continue
        vals = np.asarray([row[field] for row in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            output[field] = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
                "n": int(finite.size),
                "fraction_comparison_better": float(np.mean(finite < 0.0)),
            }
    return output


def _export_ablation_npz(
    results: Sequence[dict],
    run_dir: Path,
    variants: Sequence[Mapping[str, object]],
):
    payload = {
        "variant_ids": np.asarray([str(v["variant_id"]) for v in variants]),
        "seeds": np.asarray(sorted({int(r["seed"]) for r in results}), dtype=np.int64),
        "iterations": np.asarray(results[0]["iterations"], dtype=np.int64),
    }
    for variant in variants:
        variant_id = str(variant["variant_id"])
        variant_results = [r for r in results if str(r["variant_id"]) == variant_id]
        for key in (
            "exploitability",
            "policy_value_error",
            "nodes_touched",
            "wall_clock_seconds",
        ):
            payload[f"{variant_id}_{key}"] = np.vstack(
                [np.asarray(r[key], dtype=np.float64) for r in variant_results]
            )
    path = run_dir / "ablation_curves.npz"
    np.savez_compressed(path, **payload)
    return path


def export_ablation_results(
    results: Sequence[dict],
    run_dir: Path,
    config: dict,
    seeds: Sequence[int],
    failed: Optional[Sequence[dict]] = None,
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

    paired_rows = _paired_rows(
        results,
        reference_variant_id=str(config["reference_variant_id"]),
        comparison_variant_id=str(config["comparison_variant_id"]),
    )
    paired_csv = None
    paired_summary_path = None
    if paired_rows:
        paired_csv = write_dict_rows_csv(
            paired_rows, run_dir / "paired_differences_true_minus_false.csv"
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
            "smoke_test_seeds": SMOKE_TEST_SEEDS,
            "default_seeds_5": DEFAULT_SEEDS_5,
            "experiment_note": (
                "Controlled ablation varying only reinitialize_advantage_networks. "
                "Paired differences are comparison minus reference; with the "
                "default variants this is True minus False."
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
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Leduc poker Deep CFR advantage-reinitialisation ablation."
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--variant-ids", default=None, help="Comma-separated subset of variant ids.")
    parser.add_argument("--reference-variant-id", default=None)
    parser.add_argument("--comparison-variant-id", default=None)
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
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
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
            variant_config["description"],
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
    plot_advantage_reinitialisation_ablation(
        results,
        run_dir,
        variants=config["ablation_variants"],
        reference_variant_id=config["reference_variant_id"],
        comparison_variant_id=config["comparison_variant_id"],
        exploitability_threshold=config["exploitability_threshold"],
        average_policy_value_target=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
        aggregate_by_variant=export_info["aggregate_by_variant"],
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
