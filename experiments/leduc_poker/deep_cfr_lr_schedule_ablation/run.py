"""CLI for the Leduc poker Deep CFR learning-rate schedule ablation."""

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
    BASELINE_SCHEDULE,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXTENDED_SEEDS_5,
    EXTENDED_SEEDS_10,
    OPTIONAL_EXTRA_SCHEDULES,
    SCHEDULE_CONFIGS,
)
from .plotting import plot_lr_schedule_ablation  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.lr_schedule")


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_schedule_ids(value: Optional[str]):
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


def _filter_schedules(
    schedules: Sequence[Mapping[str, object]], ids: Optional[Sequence[str]]
):
    if ids is None:
        return [dict(schedule) for schedule in schedules]
    by_id = {str(schedule["schedule"]): dict(schedule) for schedule in schedules}
    missing = [schedule_id for schedule_id in ids if schedule_id not in by_id]
    if missing:
        raise ValueError(f"Unknown schedule id(s): {missing}")
    return [by_id[schedule_id] for schedule_id in ids]


def _default_schedule_pool(include_optional: bool) -> List[Mapping[str, object]]:
    schedules: List[Mapping[str, object]] = list(SCHEDULE_CONFIGS)
    if include_optional:
        schedules.extend(OPTIONAL_EXTRA_SCHEDULES)
    return schedules


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    schedules = _filter_schedules(
        _default_schedule_pool(args.include_optional_schedules),
        parse_schedule_ids(args.schedule_ids),
    )
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "evaluation_interval": args.evaluation_interval,
        "policy_network_layers": parse_int_tuple(args.policy_network_layers),
        "advantage_network_layers": parse_int_tuple(args.advantage_network_layers),
        "learning_rate": args.learning_rate,
        "learning_rate_end": args.learning_rate_end,
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "memory_capacity": args.memory_capacity,
        "reinitialize_advantage_networks": args.reinitialize_advantage_networks,
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "policy_network_train_every": args.policy_network_train_every,
        "compute_exploitability": args.compute_exploitability,
        "baseline_schedule": args.baseline_schedule,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    for schedule in schedules:
        schedule_kind = str(schedule.get("learning_rate_schedule", "constant"))
        schedule.setdefault("learning_rate_warmup_iterations", 0)
        if schedule_kind == "constant":
            schedule["learning_rate_end"] = float(config["learning_rate"])
        elif args.learning_rate_end is not None:
            schedule["learning_rate_end"] = float(args.learning_rate_end)
        else:
            schedule["learning_rate_end"] = float(
                schedule.get("learning_rate_end", config["learning_rate_end"])
            )
        if args.learning_rate_decay_rate is not None:
            schedule["learning_rate_decay_rate"] = float(args.learning_rate_decay_rate)
        if args.learning_rate_decay_steps is not None:
            schedule["learning_rate_decay_steps"] = int(args.learning_rate_decay_steps)
        if args.learning_rate_warmup_iterations is not None:
            schedule["learning_rate_warmup_iterations"] = int(
                args.learning_rate_warmup_iterations
            )

    schedule_ids = {str(schedule["schedule"]) for schedule in schedules}
    if str(config["baseline_schedule"]) not in schedule_ids:
        config["baseline_schedule"] = str(schedules[0]["schedule"])
    config["schedule_configs"] = tuple(schedules)
    return config


def _schedule_config(base_config: dict, schedule: Mapping[str, object]) -> dict:
    config = deepcopy(base_config)
    config.update(dict(schedule))
    config["learning_rate_schedule"] = str(schedule["learning_rate_schedule"])
    config["learning_rate_end"] = float(schedule["learning_rate_end"])
    config["learning_rate_warmup_iterations"] = int(
        schedule.get("learning_rate_warmup_iterations", 0)
    )
    if schedule.get("learning_rate_decay_rate") is not None:
        config["learning_rate_decay_rate"] = float(schedule["learning_rate_decay_rate"])
    if schedule.get("learning_rate_decay_steps") is not None:
        config["learning_rate_decay_steps"] = int(schedule["learning_rate_decay_steps"])
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
    schedule_config: Mapping[str, object],
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    schedule_id = str(schedule_config["schedule"])
    label = str(schedule_config.get("label", schedule_id))
    result["schedule"] = schedule_id
    result["schedule_label"] = label
    result["learning_rate_schedule"] = str(schedule_config["learning_rate_schedule"])
    result["learning_rate"] = np.asarray(
        result["diagnostics"].get("learning_rate", []), dtype=np.float64
    )
    result["policy_training_mode"] = str(
        schedule_config.get("policy_training_mode", "intermittent")
    )
    result["reinitialize_advantage_networks"] = bool(
        schedule_config["reinitialize_advantage_networks"]
    )
    result["policy_network_train_every"] = int(
        schedule_config["policy_network_train_every"]
    )

    exploitability_curve = np.asarray(result["exploitability"], dtype=np.float64)
    finite_exploitability = exploitability_curve[np.isfinite(exploitability_curve)]
    fraction_below_threshold = (
        float(np.mean(finite_exploitability <= float(exploitability_threshold)))
        if finite_exploitability.size
        else float("nan")
    )
    final_learning_rate = (
        float(result["learning_rate"][-1])
        if result["learning_rate"].size
        else float(schedule_config["learning_rate"])
    )

    schedule_fields = {
        "schedule": schedule_id,
        "schedule_label": label,
        "learning_rate_schedule": str(schedule_config["learning_rate_schedule"]),
        "initial_learning_rate": float(schedule_config["learning_rate"]),
        "terminal_learning_rate": final_learning_rate,
        "configured_learning_rate_end": float(schedule_config["learning_rate_end"]),
        "learning_rate_decay_rate": float(
            schedule_config.get("learning_rate_decay_rate", 0.5)
        ),
        "learning_rate_decay_steps": int(
            schedule_config.get(
                "learning_rate_decay_steps",
                max(1, int(schedule_config["num_iterations"]) // 3),
            )
        ),
        "learning_rate_warmup_iterations": int(
            schedule_config.get("learning_rate_warmup_iterations", 0)
        ),
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
        "policy_training_events": int(
            result["summary"].get("final_policy_training_events", 0)
        ),
        "policy_network_gradient_steps": int(
            result["summary"].get("final_policy_gradient_steps", 0)
        ),
        "final_learning_rate": final_learning_rate,
        "final_average_policy_value": result["summary"]["final_policy_value"],
        "final_advantage_target_variance": result["summary"][
            "final_advantage_target_variance"
        ],
        "final_policy_entropy": result["summary"][
            "final_policy_normalized_entropy_mean"
        ],
    }
    result["summary"] = {
        **schedule_fields,
        **dict(result["summary"]),
        **metric_fields,
    }
    return result


def _group_by_schedule(
    rows: Sequence[dict], schedules: Sequence[Mapping[str, object]]
) -> dict:
    grouped = {}
    for schedule in schedules:
        schedule_id = str(schedule["schedule"])
        schedule_rows = [row for row in rows if row["schedule"] == schedule_id]
        grouped[schedule_id] = summarise_numeric_fields(schedule_rows)
    return grouped


def _paired_rows(results: Sequence[dict], baseline_schedule: str) -> List[dict]:
    by_schedule_seed = {
        (str(result["schedule"]), int(result["seed"])): result["summary"]
        for result in results
    }
    seeds = sorted({int(result["seed"]) for result in results})
    schedules = sorted({str(result["schedule"]) for result in results})
    rows = []
    for seed in seeds:
        baseline = by_schedule_seed.get((str(baseline_schedule), seed))
        if baseline is None:
            continue
        for schedule in schedules:
            if schedule == str(baseline_schedule):
                continue
            comparison = by_schedule_seed.get((schedule, seed))
            if comparison is None:
                continue
            rows.append(
                {
                    "seed": int(seed),
                    "baseline_schedule": str(baseline_schedule),
                    "schedule": str(schedule),
                    "delta_final_exploitability_vs_baseline": float(
                        comparison["final_exploitability"]
                        - baseline["final_exploitability"]
                    ),
                    "delta_best_exploitability_vs_baseline": float(
                        comparison["best_exploitability"]
                        - baseline["best_exploitability"]
                    ),
                    "delta_auc_by_nodes_vs_baseline": float(
                        comparison["normalised_exploitability_auc_by_nodes"]
                        - baseline["normalised_exploitability_auc_by_nodes"]
                    ),
                    "delta_auc_by_iteration_vs_baseline": float(
                        comparison["normalised_exploitability_auc_by_iteration"]
                        - baseline["normalised_exploitability_auc_by_iteration"]
                    ),
                    "delta_final_policy_value_error_vs_baseline": float(
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
    if not rows:
        return output
    schedules = sorted({str(row["schedule"]) for row in rows})
    for schedule in schedules:
        schedule_rows = [row for row in rows if str(row["schedule"]) == schedule]
        output[schedule] = {}
        for field in schedule_rows[0].keys():
            if field in {"seed", "baseline_schedule", "schedule"}:
                continue
            vals = np.asarray([row[field] for row in schedule_rows], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                output[schedule][field] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
                    "n": int(finite.size),
                    "fraction_schedule_better": float(np.mean(finite < 0.0)),
                }
    return output


def _stack_padded(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty((0, 0))
    arrays = [np.asarray(array) for array in arrays]
    max_len = max(array.shape[0] for array in arrays)
    padded = []
    for array in arrays:
        if array.shape[0] == max_len:
            padded.append(array)
            continue
        pad = np.full(max_len - array.shape[0], np.nan, dtype=np.float64)
        padded.append(np.concatenate([array.astype(np.float64), pad]))
    return np.vstack(padded)


def _export_ablation_npz(
    results: Sequence[dict],
    run_dir: Path,
    schedules: Sequence[Mapping[str, object]],
):
    payload = {
        "schedules": np.asarray([str(schedule["schedule"]) for schedule in schedules]),
        "seeds": np.asarray(sorted({int(r["seed"]) for r in results}), dtype=np.int64),
        "iterations": np.asarray(results[0]["iterations"], dtype=np.int64),
    }
    for schedule in schedules:
        schedule_id = str(schedule["schedule"])
        subset = [r for r in results if str(r["schedule"]) == schedule_id]
        for key in (
            "exploitability",
            "policy_value_error",
            "nodes_touched",
            "wall_clock_seconds",
            "learning_rate",
        ):
            payload[f"{schedule_id}_{key}"] = _stack_padded(
                [np.asarray(r[key], dtype=np.float64) for r in subset]
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
    schedules = list(config["schedule_configs"])
    info = export_results(
        results,
        run_dir,
        config,
        seeds,
        failed_seeds=failed,
        write_multiseed_npz=False,
    )
    summary_rows = [result["summary"] for result in results]
    aggregate_by_schedule = {
        "by_schedule": _group_by_schedule(summary_rows, schedules)
    }
    aggregate_path = run_dir / "aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate_by_schedule), f, indent=2)

    paired_rows = _paired_rows(results, str(config["baseline_schedule"]))
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

    npz_path = _export_ablation_npz(results, run_dir, schedules)
    write_experiment_metadata(
        run_dir,
        config=config,
        seeds=seeds,
        completed_seeds=sorted({int(r["seed"]) for r in results}),
        extra={
            "extended_seeds_5": EXTENDED_SEEDS_5,
            "extended_seeds_10": EXTENDED_SEEDS_10,
            "experiment_note": (
                "Controlled ablation varying only the learning-rate schedule. "
                "Paired differences are schedule minus baseline, where lower "
                "exploitability and policy-value error are better."
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
        "aggregate_by_schedule": aggregate_by_schedule,
        "paired_rows": paired_rows,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Leduc poker Deep CFR learning-rate schedule ablation."
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--schedule-ids", default=None, help="Comma-separated subset of schedule ids.")
    parser.add_argument("--include-optional-schedules", action="store_true")
    parser.add_argument("--baseline-schedule", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-end", type=float, default=None)
    parser.add_argument("--learning-rate-decay-rate", type=float, default=None)
    parser.add_argument("--learning-rate-decay-steps", type=int, default=None)
    parser.add_argument("--learning-rate-warmup-iterations", type=int, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--advantage-network-layers", default=None)
    parser.add_argument("--reinitialize-advantage-networks", type=_str2bool, default=None)
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
    for schedule in config["schedule_configs"]:
        schedule_config = _schedule_config(config, schedule)
        _LOGGER.info(
            "Running %s: %s",
            schedule_config["schedule"],
            schedule_config.get("label", schedule_config["schedule"]),
        )
        for seed in tqdm(seeds, desc=str(schedule_config["schedule"])):
            try:
                result = run_single_seed(
                    seed,
                    schedule_config,
                    export_dir=run_dir,
                    save_final_checkpoint=args.save_final_checkpoints,
                    final_window=args.final_window,
                )
                results.append(
                    _augment_result(
                        result,
                        schedule_config,
                        args.final_window,
                        config["exploitability_threshold"],
                    )
                )
            except Exception as exc:  # pragma: no cover
                _LOGGER.exception(
                    "Seed %s failed for schedule %s: %s",
                    seed,
                    schedule_config["schedule"],
                    exc,
                )
                failed.append(
                    {
                        "seed": int(seed),
                        "schedule": str(schedule_config["schedule"]),
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
    plot_lr_schedule_ablation(
        results,
        run_dir,
        schedules=config["schedule_configs"],
        baseline_schedule=str(config["baseline_schedule"]),
        exploitability_threshold=float(config["exploitability_threshold"]),
        average_policy_value_target=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
        aggregate_by_schedule=export_info["aggregate_by_schedule"],
        paired_rows=export_info["paired_rows"],
    )

    _LOGGER.info(
        "Completed %d/%d runs",
        len(results),
        len(seeds) * len(config["schedule_configs"]),
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
