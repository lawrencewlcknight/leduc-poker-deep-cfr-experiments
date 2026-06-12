"""CLI for the Leduc poker Deep CFR replay and averaging ablation."""

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
    BASELINE_VARIANT_ID,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    DEFAULT_SEEDS_5,
    EXTENDED_SEEDS_10,
    REPLAY_AVERAGING_VARIANTS,
)
from .plotting import plot_replay_averaging_ablation  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.replay_averaging")


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


def _filter_variants(
    variants: Sequence[Mapping[str, object]], ids: Optional[Sequence[str]]
):
    if ids is None:
        return [dict(v) for v in variants]
    by_id = {str(v["variant_id"]): dict(v) for v in variants}
    missing = [variant_id for variant_id in ids if variant_id not in by_id]
    if missing:
        raise ValueError(f"Unknown variant id(s): {missing}")
    return [by_id[variant_id] for variant_id in ids]


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
        "priority_alpha": args.priority_alpha,
        "priority_epsilon": args.priority_epsilon,
        "baseline_variant_id": args.baseline_variant_id,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["ablation_variants"] = tuple(variants)
    variant_ids = {str(v["variant_id"]) for v in variants}
    if str(config["baseline_variant_id"]) not in variant_ids:
        config["baseline_variant_id"] = str(variants[0]["variant_id"])
    return config


def _variant_config(base_config: dict, variant: Mapping[str, object]) -> dict:
    config = deepcopy(base_config)
    config.update(dict(variant))
    config["advantage_replay_sampling"] = str(variant["advantage_replay_sampling"])
    config["average_strategy_weighting"] = str(variant["average_strategy_weighting"])
    config["priority_alpha"] = float(base_config.get("priority_alpha", 1.0))
    config["priority_epsilon"] = float(base_config.get("priority_epsilon", 1e-6))
    return config


def _best_exploitability_iteration(iterations, exploitability) -> float:
    values = np.asarray(exploitability, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    finite_indices = np.where(finite)[0]
    local = int(np.nanargmin(values[finite]))
    return int(np.asarray(iterations)[finite_indices[local]])


def _final_diag(result: Mapping[str, object], key: str) -> float:
    values = np.asarray(result["diagnostics"].get(key, []), dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(values[-1])


def _augment_result(
    result: dict,
    variant_config: Mapping[str, object],
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    variant_id = str(variant_config["variant_id"])
    result["variant_id"] = variant_id
    result["variant_label"] = str(variant_config.get("label", variant_id))
    result["advantage_replay_sampling"] = str(
        variant_config["advantage_replay_sampling"]
    )
    result["average_strategy_weighting"] = str(
        variant_config["average_strategy_weighting"]
    )
    result["priority_alpha"] = float(variant_config.get("priority_alpha", 1.0))
    result["priority_epsilon"] = float(variant_config.get("priority_epsilon", 1e-6))
    result["policy_training_mode"] = str(
        variant_config.get("policy_training_mode", "intermittent")
    )
    result["reinitialize_advantage_networks"] = bool(
        variant_config["reinitialize_advantage_networks"]
    )
    result["policy_network_train_every"] = int(
        variant_config["policy_network_train_every"]
    )

    exploitability_curve = np.asarray(result["exploitability"], dtype=np.float64)
    finite_exploitability = exploitability_curve[np.isfinite(exploitability_curve)]
    fraction_below_threshold = (
        float(np.mean(finite_exploitability <= float(exploitability_threshold)))
        if finite_exploitability.size
        else float("nan")
    )

    variant_fields = {
        "variant_id": variant_id,
        "variant_label": str(variant_config.get("label", variant_id)),
        "advantage_replay_sampling": str(variant_config["advantage_replay_sampling"]),
        "average_strategy_weighting": str(
            variant_config["average_strategy_weighting"]
        ),
        "priority_alpha": float(variant_config.get("priority_alpha", 1.0)),
        "priority_epsilon": float(variant_config.get("priority_epsilon", 1e-6)),
    }
    metric_fields = {
        "best_exploitability_iteration": _best_exploitability_iteration(
            result["iterations"], result["exploitability"]
        ),
        "final_window_std_exploitability": final_window_std(
            result["exploitability"], window=final_window
        ),
        "normalised_exploitability_auc_by_iteration": normalised_auc(
            result["iterations"], result["exploitability"]
        ),
        "normalised_exploitability_auc_by_nodes": normalised_auc(
            result["nodes_touched"], result["exploitability"]
        ),
        "fraction_checkpoints_below_threshold": fraction_below_threshold,
        "final_priority_effective_sample_size": _final_diag(
            result, "advantage_priority_effective_sample_size"
        ),
        "policy_training_events": int(
            result["summary"].get("final_policy_training_events", 0)
        ),
        "policy_network_gradient_steps": int(
            result["summary"].get("final_policy_gradient_steps", 0)
        ),
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


def _paired_rows(results: Sequence[dict], baseline_variant_id: str) -> List[dict]:
    by_variant_seed = {
        (str(result["variant_id"]), int(result["seed"])): result["summary"]
        for result in results
    }
    seeds = sorted({int(result["seed"]) for result in results})
    variants = sorted({str(result["variant_id"]) for result in results})
    rows = []
    for seed in seeds:
        baseline = by_variant_seed.get((str(baseline_variant_id), seed))
        if baseline is None:
            continue
        for variant_id in variants:
            if variant_id == str(baseline_variant_id):
                continue
            comparison = by_variant_seed.get((variant_id, seed))
            if comparison is None:
                continue
            rows.append(
                {
                    "seed": int(seed),
                    "baseline_variant_id": str(baseline_variant_id),
                    "variant_id": variant_id,
                    "delta_final_exploitability_vs_baseline": float(
                        comparison["final_exploitability"]
                        - baseline["final_exploitability"]
                    ),
                    "delta_best_exploitability_vs_baseline": float(
                        comparison["best_exploitability"]
                        - baseline["best_exploitability"]
                    ),
                    "delta_final_window_mean_exploitability_vs_baseline": float(
                        comparison["final_window_mean_exploitability"]
                        - baseline["final_window_mean_exploitability"]
                    ),
                    "delta_auc_by_iteration_vs_baseline": float(
                        comparison["normalised_exploitability_auc_by_iteration"]
                        - baseline["normalised_exploitability_auc_by_iteration"]
                    ),
                    "delta_auc_by_nodes_vs_baseline": float(
                        comparison["normalised_exploitability_auc_by_nodes"]
                        - baseline["normalised_exploitability_auc_by_nodes"]
                    ),
                    "delta_policy_value_error_vs_baseline": float(
                        comparison["final_policy_value_error"]
                        - baseline["final_policy_value_error"]
                    ),
                    "delta_wall_clock_seconds_vs_baseline": float(
                        comparison["final_wall_clock_seconds"]
                        - baseline["final_wall_clock_seconds"]
                    ),
                }
            )
    return rows


def _paired_summary(rows: Sequence[dict]) -> dict:
    output = {}
    for variant_id in sorted({str(row["variant_id"]) for row in rows}):
        variant_rows = [row for row in rows if str(row["variant_id"]) == variant_id]
        output[variant_id] = {}
        for field in variant_rows[0].keys():
            if field in {"seed", "baseline_variant_id", "variant_id"}:
                continue
            vals = np.asarray([row[field] for row in variant_rows], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                output[variant_id][field] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
                    "n": int(finite.size),
                    "fraction_variant_better": float(np.mean(finite < 0.0)),
                }
    return output


def _stack_padded(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty((0, 0))
    arrays = [np.asarray(array, dtype=np.float64) for array in arrays]
    max_len = max(array.shape[0] for array in arrays)
    padded = []
    for array in arrays:
        if array.shape[0] == max_len:
            padded.append(array)
        else:
            pad = np.full(max_len - array.shape[0], np.nan, dtype=np.float64)
            padded.append(np.concatenate([array, pad]))
    return np.vstack(padded)


def _diag_curve(result: Mapping[str, object], key: str) -> np.ndarray:
    return np.asarray(result["diagnostics"].get(key, []), dtype=np.float64)


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
        subset = [r for r in results if str(r["variant_id"]) == variant_id]
        for key in (
            "exploitability",
            "policy_value_error",
            "nodes_touched",
            "wall_clock_seconds",
        ):
            payload[f"{variant_id}_{key}"] = _stack_padded(
                [np.asarray(r[key], dtype=np.float64) for r in subset]
            )
        payload[f"{variant_id}_advantage_priority_effective_sample_size"] = (
            _stack_padded(
                [
                    _diag_curve(r, "advantage_priority_effective_sample_size")
                    for r in subset
                ]
            )
        )
        payload[f"{variant_id}_advantage_target_variance"] = _stack_padded(
            [_diag_curve(r, "advantage_target_variance") for r in subset]
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
                "Controlled replay and average-strategy weighting ablation. "
                "Paired differences are variant minus the uniform-replay, "
                "linear-average Experiment 2 baseline."
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
        description="Run the Leduc poker Deep CFR replay and averaging ablation."
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
