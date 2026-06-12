"""CLI for the Leduc poker Deep CFR constrained random search."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

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
    resolve_solver_batch_sizes,
    run_single_seed,
    summarise_numeric_fields,
    write_dict_rows_csv,
    write_experiment_metadata,
    write_failed_seeds,
)

from .config import (  # noqa: E402
    DEFAULT_CONFIG,
    EXPERIMENT2_BASELINE_CONFIG,
)
from .plotting import plot_random_search  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.constrained_random_search")


def parse_seeds(seed_string: Optional[str]) -> Optional[List[int]]:
    if seed_string is None:
        return None
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _normalise_value(value):
    if isinstance(value, list):
        return tuple(value)
    return value


def _config_signature(config: Mapping[str, object]) -> tuple:
    return tuple(
        (key, _normalise_value(value))
        for key, value in sorted(config.items())
        if key not in {"config_label"}
    )


def _compact_search_space() -> dict:
    return {
        "learning_rate": (0.001, 0.003),
        "num_traversals": (4, 6),
        "policy_network_layers": ((8, 8), (8,)),
        "advantage_network_layers": ((8, 8), (8,)),
        "batch_size_advantage": (2,),
        "batch_size_strategy": (2,),
        "memory_capacity": (256,),
        "policy_network_train_steps": (1,),
        "advantage_network_train_steps": (1,),
        "reinitialize_advantage_networks": (False, True),
    }


def _apply_quick_test_defaults(config: dict) -> None:
    baseline = dict(config["baseline_config"])
    baseline.update(
        {
            "policy_network_layers": (8, 8),
            "advantage_network_layers": (8, 8),
            "num_traversals": 4,
            "evaluation_interval": 1,
            "learning_rate": 0.003,
            "batch_size_advantage": 2,
            "batch_size_strategy": 2,
            "memory_capacity": 256,
            "policy_network_train_steps": 1,
            "advantage_network_train_steps": 1,
            "policy_network_train_every": 1,
        }
    )
    config["baseline_config"] = baseline
    config["search_space"] = _compact_search_space()
    config["screening_num_iterations"] = 3
    config["confirmation_num_iterations"] = 3
    config["screening_random_configs"] = 1
    config["confirmation_top_k"] = 1
    config["screening_seeds"] = [1234]
    config["confirmation_seeds"] = [1234]


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if args.quick_test:
        _apply_quick_test_defaults(config)

    overrides = {
        "experiment_name": args.experiment_name,
        "screening_num_iterations": args.screening_iterations,
        "confirmation_num_iterations": args.confirmation_iterations,
        "screening_random_configs": args.screening_random_configs,
        "confirmation_top_k": args.confirmation_top_k,
        "master_seed": args.master_seed,
        "exploitability_threshold": args.exploitability_threshold,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    screening_seeds = parse_seeds(args.screening_seeds)
    confirmation_seeds = parse_seeds(args.confirmation_seeds)
    if screening_seeds is not None:
        config["screening_seeds"] = screening_seeds
    if args.use_extended_confirmation_seeds:
        config["confirmation_seeds"] = list(config["extended_confirmation_seeds"])
    if confirmation_seeds is not None:
        config["confirmation_seeds"] = confirmation_seeds

    baseline = dict(config["baseline_config"])
    baseline_overrides = {
        "num_traversals": args.traversals,
        "evaluation_interval": args.evaluation_interval,
        "policy_network_train_every": args.policy_network_train_every,
        "policy_network_layers": parse_int_tuple(args.policy_network_layers),
        "advantage_network_layers": parse_int_tuple(args.advantage_network_layers),
        "learning_rate": args.learning_rate,
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "memory_capacity": args.memory_capacity,
        "reinitialize_advantage_networks": args.reinitialize_advantage_networks,
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "compute_exploitability": args.compute_exploitability,
    }
    for key, value in baseline_overrides.items():
        if value is not None:
            baseline[key] = value
    config["baseline_config"] = baseline

    if int(config["screening_random_configs"]) < 0:
        raise ValueError("screening_random_configs must be >= 0")
    if int(config["confirmation_top_k"]) < 0:
        raise ValueError("confirmation_top_k must be >= 0")
    return config


def sample_candidate_configs(config: Mapping[str, object]) -> List[dict]:
    """Samples unique random-search candidates from the constrained space."""
    n_configs = int(config["screening_random_configs"])
    rng = random.Random(int(config["master_seed"]))
    baseline = dict(config["baseline_config"])
    search_space = dict(config["search_space"])
    seen = {_config_signature(baseline)}
    candidates: List[dict] = []
    max_attempts = max(100, n_configs * 100)
    attempts = 0

    while len(candidates) < n_configs and attempts < max_attempts:
        attempts += 1
        candidate = dict(baseline)
        for key, values in search_space.items():
            candidate[key] = rng.choice(list(values))
        candidate["policy_network_train_every"] = int(
            baseline["policy_network_train_every"]
        )
        candidate["evaluation_interval"] = int(baseline["evaluation_interval"])
        candidate["game_name"] = str(baseline["game_name"])
        signature = _config_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        candidate["config_label"] = f"random_candidate_{len(candidates) + 1:02d}"
        candidates.append(candidate)

    if len(candidates) < n_configs:
        raise ValueError(
            f"Could only sample {len(candidates)} unique candidates from the "
            f"constrained search space; requested {n_configs}."
        )
    return candidates


def _run_config_for_stage(config: Mapping[str, object], stage: str, iterations: int) -> dict:
    run_config = dict(config)
    run_config["num_iterations"] = int(iterations)
    run_config["game_name"] = str(run_config.get("game_name", "leduc_poker"))
    run_config.setdefault("exploitability_threshold", DEFAULT_CONFIG["exploitability_threshold"])
    run_config.setdefault("policy_training_mode", "intermittent")
    run_config.setdefault("final_policy_network_train_steps", None)
    batch_size_advantage, batch_size_strategy = resolve_solver_batch_sizes(run_config)
    run_config["batch_size_advantage"] = batch_size_advantage
    run_config["batch_size_strategy"] = batch_size_strategy
    return run_config


def _curve_rows(result: Mapping[str, object]) -> List[dict]:
    diag = result["diagnostics"]
    rows = []
    for i, iteration in enumerate(result["iterations"]):
        row = {
            "stage": result["stage"],
            "config_label": result["config_label"],
            "seed": int(result["seed"]),
            "iteration": int(iteration),
            "nodes_touched": float(result["nodes_touched"][i]),
            "wall_clock_seconds": float(result["wall_clock_seconds"][i]),
            "exploitability": float(result["exploitability"][i]),
            "average_policy_value": float(result["average_policy_value"][i]),
            "policy_value_signed_error": float(result["policy_value_signed_error"][i]),
            "policy_value_error": float(result["policy_value_error"][i]),
        }
        for key in (
            "policy_loss",
            "learning_rate",
            "strategy_buffer_size",
            "advantage_buffer_size_player_0",
            "advantage_buffer_size_player_1",
            "legal_action_mass_mean",
            "legal_action_mass_min",
            "policy_normalized_entropy_mean",
            "advantage_target_variance",
            "policy_grad_norm",
            "advantage_grad_norm_player_0",
            "advantage_grad_norm_player_1",
            "policy_training_events",
            "policy_gradient_steps",
        ):
            if key in diag and len(diag[key]) > i:
                value = diag[key][i]
                row[key] = float(value) if np.asarray(value).dtype.kind == "f" else int(value)
        rows.append(row)
    return rows


def _augment_result(
    result: dict,
    run_config: Mapping[str, object],
    stage: str,
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    label = str(run_config["config_label"])
    result["stage"] = str(stage)
    result["config_label"] = label
    result["policy_training_mode"] = str(run_config.get("policy_training_mode", "intermittent"))
    result["reinitialize_advantage_networks"] = bool(run_config["reinitialize_advantage_networks"])
    result["policy_network_train_every"] = int(run_config["policy_network_train_every"])
    if run_config.get("final_policy_network_train_steps") is not None:
        result["final_policy_network_train_steps"] = int(
            run_config["final_policy_network_train_steps"]
        )

    exploitability = np.asarray(result["exploitability"], dtype=np.float64)
    finite_exploitability = exploitability[np.isfinite(exploitability)]
    fraction_below_threshold = (
        float(np.mean(finite_exploitability <= float(exploitability_threshold)))
        if finite_exploitability.size
        else float("nan")
    )

    config_fields = {
        "stage": str(stage),
        "config_label": label,
        "status": "ok",
        "config_kind": (
            "baseline"
            if label == EXPERIMENT2_BASELINE_CONFIG["config_label"]
            else "candidate"
        ),
        "num_iterations": int(run_config["num_iterations"]),
        "num_traversals": int(run_config["num_traversals"]),
        "evaluation_interval": int(run_config["evaluation_interval"]),
        "learning_rate": float(run_config["learning_rate"]),
        "policy_network_layers": str(tuple(run_config["policy_network_layers"])),
        "advantage_network_layers": str(tuple(run_config["advantage_network_layers"])),
        "batch_size_advantage": int(run_config["batch_size_advantage"]),
        "batch_size_strategy": int(run_config["batch_size_strategy"]),
        "memory_capacity": int(run_config["memory_capacity"]),
        "reinitialize_advantage_networks": bool(run_config["reinitialize_advantage_networks"]),
        "policy_network_train_steps": int(run_config["policy_network_train_steps"]),
        "advantage_network_train_steps": int(run_config["advantage_network_train_steps"]),
        "policy_network_train_every": int(run_config["policy_network_train_every"]),
    }
    metric_fields = {
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
    }
    result["summary"] = {
        **config_fields,
        **dict(result["summary"]),
        **metric_fields,
    }
    return result


def _write_trace(run_dir: Path, result: Mapping[str, object], run_config: Mapping[str, object]) -> Path:
    trace_dir = Path(run_dir) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / (
        f"{result['stage']}_{result['config_label']}_seed_{int(result['seed'])}.json"
    )
    payload = {
        "stage": result["stage"],
        "config_label": result["config_label"],
        "seed": int(result["seed"]),
        "config": json_safe(dict(run_config)),
        "summary": json_safe(result["summary"]),
        "curves": json_safe(_curve_rows(result)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def run_stage(
    *,
    configs: Sequence[Mapping[str, object]],
    seeds: Sequence[int],
    stage: str,
    num_iterations: int,
    run_dir: Path,
    final_window: int,
    exploitability_threshold: float,
    save_final_checkpoints: bool = False,
) -> tuple[List[dict], List[dict]]:
    results = []
    failed = []
    total = len(configs) * len(seeds)
    progress = tqdm(total=total, desc=stage)
    for base_config in configs:
        run_config = _run_config_for_stage(base_config, stage, num_iterations)
        _LOGGER.info(
            "[%s] Running config=%s iterations=%s",
            stage,
            run_config["config_label"],
            num_iterations,
        )
        for seed in seeds:
            try:
                result = run_single_seed(
                    int(seed),
                    {**run_config, "exploitability_threshold": exploitability_threshold},
                    export_dir=run_dir,
                    save_final_checkpoint=save_final_checkpoints,
                    final_window=final_window,
                )
                result = _augment_result(
                    result,
                    run_config,
                    stage,
                    final_window,
                    exploitability_threshold,
                )
                trace_path = _write_trace(run_dir, result, run_config)
                result["summary"]["trace_path"] = str(trace_path)
                results.append(result)
            except Exception as exc:  # pragma: no cover
                _LOGGER.exception(
                    "[%s] Failed config=%s seed=%s: %s",
                    stage,
                    run_config["config_label"],
                    seed,
                    exc,
                )
                failed.append(
                    {
                        "stage": str(stage),
                        "config_label": str(run_config["config_label"]),
                        "seed": int(seed),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                progress.update(1)
    progress.close()
    return results, failed


def _rows_for_stage(summary_rows: Sequence[dict], stage: str) -> List[dict]:
    return [
        row for row in summary_rows
        if str(row.get("stage")) == str(stage) and row.get("status") == "ok"
    ]


def aggregate_config_rows(summary_rows: Sequence[dict], stage: str) -> List[dict]:
    rows = _rows_for_stage(summary_rows, stage)
    labels = sorted({str(row["config_label"]) for row in rows})
    output = []
    for label in labels:
        group = [row for row in rows if str(row["config_label"]) == label]
        flat = {
            "stage": str(stage),
            "config_label": label,
            "n_seeds": len({int(row["seed"]) for row in group}),
            "config_kind": group[0].get("config_kind", ""),
        }
        summary = summarise_numeric_fields(group)
        for metric, stats_dict in summary.items():
            flat[f"{metric}_mean"] = stats_dict["mean"]
            flat[f"{metric}_std"] = stats_dict["std"]
            flat[f"{metric}_se"] = stats_dict["se"]
        output.append(flat)
    output.sort(
        key=lambda row: (
            float(row.get("final_exploitability_mean", np.inf)),
            float(row.get("normalised_exploitability_auc_by_iteration_mean", np.inf)),
            row["config_label"],
        )
    )
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def choose_confirmation_configs(
    *,
    screening_summary_rows: Sequence[dict],
    baseline_config: Mapping[str, object],
    candidate_configs: Sequence[Mapping[str, object]],
    top_k: int,
) -> tuple[List[dict], List[dict]]:
    screening_agg = aggregate_config_rows(screening_summary_rows, "screening")
    ranked_candidates = [
        row for row in screening_agg
        if str(row["config_label"]) != str(baseline_config["config_label"])
    ]
    top_labels = [row["config_label"] for row in ranked_candidates[: int(top_k)]]
    by_label = {str(config["config_label"]): dict(config) for config in candidate_configs}
    confirmation = [dict(baseline_config)]
    confirmation.extend(by_label[label] for label in top_labels if label in by_label)
    return confirmation, screening_agg


def paired_differences(summary_rows: Sequence[dict], baseline_label: str) -> List[dict]:
    rows = _rows_for_stage(summary_rows, "confirmation")
    by_label_seed = {
        (str(row["config_label"]), int(row["seed"])): row
        for row in rows
    }
    labels = sorted({str(row["config_label"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    output = []
    for label in labels:
        if label == str(baseline_label):
            continue
        for seed in seeds:
            baseline = by_label_seed.get((str(baseline_label), seed))
            candidate = by_label_seed.get((label, seed))
            if baseline is None or candidate is None:
                continue
            output.append(
                {
                    "seed": int(seed),
                    "baseline_config_label": str(baseline_label),
                    "config_label": label,
                    "delta_final_exploitability_vs_baseline": float(
                        candidate["final_exploitability"] - baseline["final_exploitability"]
                    ),
                    "delta_best_exploitability_vs_baseline": float(
                        candidate["best_exploitability"] - baseline["best_exploitability"]
                    ),
                    "delta_auc_by_iteration_vs_baseline": float(
                        candidate["normalised_exploitability_auc_by_iteration"]
                        - baseline["normalised_exploitability_auc_by_iteration"]
                    ),
                    "delta_auc_by_nodes_vs_baseline": float(
                        candidate["normalised_exploitability_auc_by_nodes"]
                        - baseline["normalised_exploitability_auc_by_nodes"]
                    ),
                    "delta_final_policy_value_error_vs_baseline": float(
                        candidate["final_policy_value_error"]
                        - baseline["final_policy_value_error"]
                    ),
                    "delta_wall_clock_seconds_vs_baseline": float(
                        candidate["final_wall_clock_seconds"]
                        - baseline["final_wall_clock_seconds"]
                    ),
                    "delta_nodes_touched_vs_baseline": float(
                        candidate["final_nodes_touched"]
                        - baseline["final_nodes_touched"]
                    ),
                }
            )
    return output


def paired_difference_summary(rows: Sequence[dict]) -> dict:
    output = {}
    for label in sorted({str(row["config_label"]) for row in rows}):
        label_rows = [row for row in rows if str(row["config_label"]) == label]
        output[label] = {}
        for field in label_rows[0].keys():
            if field in {"seed", "baseline_config_label", "config_label"}:
                continue
            values = np.asarray([row[field] for row in label_rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                output[label][field] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
                    "n": int(finite.size),
                    "fraction_candidate_better": float(np.mean(finite < 0.0)),
                }
    return output


def _safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _stack_padded(arrays: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(array, dtype=np.float64) for array in arrays]
    if not arrays:
        return np.empty((0, 0))
    max_len = max(array.shape[0] for array in arrays)
    padded = []
    for array in arrays:
        if array.shape[0] == max_len:
            padded.append(array)
        else:
            pad = np.full(max_len - array.shape[0], np.nan, dtype=np.float64)
            padded.append(np.concatenate([array, pad]))
    return np.vstack(padded)


def export_search_npz(results: Sequence[dict], run_dir: Path) -> Path:
    payload = {
        "stages": np.asarray(sorted({str(result["stage"]) for result in results})),
        "config_labels": np.asarray(sorted({str(result["config_label"]) for result in results})),
        "seeds": np.asarray(sorted({int(result["seed"]) for result in results}), dtype=np.int64),
    }
    for stage in sorted({str(result["stage"]) for result in results}):
        for label in sorted({str(result["config_label"]) for result in results if str(result["stage"]) == stage}):
            subset = [
                result for result in results
                if str(result["stage"]) == stage and str(result["config_label"]) == label
            ]
            prefix = f"{_safe_key(stage)}_{_safe_key(label)}"
            for key in (
                "iterations",
                "exploitability",
                "average_policy_value",
                "policy_value_error",
                "nodes_touched",
                "wall_clock_seconds",
            ):
                payload[f"{prefix}_{key}"] = _stack_padded(
                    [np.asarray(result[key], dtype=np.float64) for result in subset]
                )
    path = Path(run_dir) / "search_curves.npz"
    np.savez_compressed(path, **payload)
    return path


def configuration_rows(
    baseline_config: Mapping[str, object],
    candidate_configs: Sequence[Mapping[str, object]],
    confirmation_configs: Sequence[Mapping[str, object]],
) -> List[dict]:
    selected = {str(config["config_label"]) for config in confirmation_configs}
    rows = []
    for config in [baseline_config, *candidate_configs]:
        row = {
            "config_label": str(config["config_label"]),
            "config_kind": (
                "baseline"
                if str(config["config_label"]) == str(baseline_config["config_label"])
                else "candidate"
            ),
            "selected_for_confirmation": str(config["config_label"]) in selected,
        }
        for key in (
            "learning_rate",
            "num_traversals",
            "policy_network_layers",
            "advantage_network_layers",
            "batch_size_advantage",
            "batch_size_strategy",
            "memory_capacity",
            "policy_network_train_steps",
            "advantage_network_train_steps",
            "reinitialize_advantage_networks",
            "policy_network_train_every",
            "evaluation_interval",
        ):
            value = config[key]
            row[key] = str(tuple(value)) if isinstance(value, tuple) else value
        rows.append(row)
    return rows


def export_search_results(
    *,
    results: Sequence[dict],
    run_dir: Path,
    config: dict,
    baseline_config: Mapping[str, object],
    candidate_configs: Sequence[Mapping[str, object]],
    confirmation_configs: Sequence[Mapping[str, object]],
    screening_agg: Sequence[dict],
    confirmation_agg: Sequence[dict],
    failed: Optional[Sequence[dict]] = None,
) -> dict:
    seeds = sorted(
        set(map(int, config["screening_seeds"])) | set(map(int, config["confirmation_seeds"]))
    )
    info = export_results(
        results,
        run_dir,
        config,
        seeds,
        failed_seeds=failed,
        write_multiseed_npz=False,
    )
    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in _curve_rows(result)]

    screening_summary_path = write_dict_rows_csv(
        [row for row in summary_rows if row["stage"] == "screening"],
        Path(run_dir) / "screening_seed_summary.csv",
    )
    confirmation_summary_path = write_dict_rows_csv(
        [row for row in summary_rows if row["stage"] == "confirmation"],
        Path(run_dir) / "confirmation_seed_summary.csv",
    )
    screening_agg_path = write_dict_rows_csv(
        screening_agg, Path(run_dir) / "screening_config_summary.csv"
    )
    confirmation_agg_path = write_dict_rows_csv(
        confirmation_agg, Path(run_dir) / "confirmation_config_summary.csv"
    )
    config_table_path = write_dict_rows_csv(
        configuration_rows(baseline_config, candidate_configs, confirmation_configs),
        Path(run_dir) / "search_configurations.csv",
    )

    aggregate_summary = {
        "screening": {
            "by_config_label": {
                label: summarise_numeric_fields(
                    [row for row in _rows_for_stage(summary_rows, "screening") if row["config_label"] == label]
                )
                for label in sorted({row["config_label"] for row in _rows_for_stage(summary_rows, "screening")})
            }
        },
        "confirmation": {
            "by_config_label": {
                label: summarise_numeric_fields(
                    [row for row in _rows_for_stage(summary_rows, "confirmation") if row["config_label"] == label]
                )
                for label in sorted({row["config_label"] for row in _rows_for_stage(summary_rows, "confirmation")})
            }
        },
    }
    aggregate_path = Path(run_dir) / "aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate_summary), f, indent=2)

    paired_rows = paired_differences(summary_rows, str(baseline_config["config_label"]))
    paired_csv = None
    paired_summary_path = None
    paired_summary = {}
    if paired_rows:
        paired_csv = write_dict_rows_csv(
            paired_rows, Path(run_dir) / "paired_differences_vs_baseline.csv"
        )
        paired_summary = paired_difference_summary(paired_rows)
        paired_summary_path = Path(run_dir) / "paired_difference_summary.json"
        with open(paired_summary_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(paired_summary), f, indent=2)

    npz_path = export_search_npz(results, Path(run_dir))
    plot_paths = plot_random_search(
        run_dir=run_dir,
        screening_config_summary=screening_agg,
        confirmation_config_summary=confirmation_agg,
        curve_rows=curve_rows,
        paired_rows=paired_rows,
        exploitability_threshold=float(config["exploitability_threshold"]),
        average_policy_value_target=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
    )

    metadata_path = write_experiment_metadata(
        Path(run_dir),
        config=config,
        seeds=seeds,
        completed_seeds=sorted({int(result["seed"]) for result in results}),
        extra={
            "experiment_note": (
                "Two-stage constrained random search. Screening ranks sampled "
                "configurations under a shortened budget; confirmation reruns "
                "the Experiment 2 baseline plus the strongest screened "
                "candidates under the full configured budget."
            ),
            "screening_seed_summary_csv": str(screening_summary_path),
            "confirmation_seed_summary_csv": str(confirmation_summary_path),
            "screening_config_summary_csv": str(screening_agg_path),
            "confirmation_config_summary_csv": str(confirmation_agg_path),
            "search_configurations_csv": str(config_table_path),
            "paired_differences_csv": str(paired_csv) if paired_csv else None,
            "paired_difference_summary_json": (
                str(paired_summary_path) if paired_summary_path else None
            ),
            "search_curves_npz": str(npz_path),
            "plot_paths": [str(path) for path in plot_paths],
        },
    )
    if failed:
        write_failed_seeds(Path(run_dir), failed)

    return {
        **info,
        "metadata": metadata_path,
        "aggregate_summary": aggregate_path,
        "screening_seed_summary": screening_summary_path,
        "confirmation_seed_summary": confirmation_summary_path,
        "screening_config_summary": screening_agg_path,
        "confirmation_config_summary": confirmation_agg_path,
        "search_configurations": config_table_path,
        "paired_differences_csv": paired_csv,
        "paired_difference_summary": paired_summary_path,
        "search_curves_npz": npz_path,
        "plot_paths": plot_paths,
        "paired_rows": paired_rows,
        "paired_summary": paired_summary,
    }


def run_staged_random_search(
    *,
    config: dict,
    run_dir: Path,
    final_window: int = DEFAULT_FINAL_WINDOW,
    save_final_checkpoints: bool = False,
) -> dict:
    baseline_config = dict(config["baseline_config"])
    candidate_configs = sample_candidate_configs(config)
    screening_configs = [dict(baseline_config), *candidate_configs]

    _LOGGER.info("Sampled candidate configurations:")
    for candidate in candidate_configs:
        _LOGGER.info("  %s: %s", candidate["config_label"], candidate)

    screening_results, screening_failed = run_stage(
        configs=screening_configs,
        seeds=config["screening_seeds"],
        stage="screening",
        num_iterations=int(config["screening_num_iterations"]),
        run_dir=run_dir,
        final_window=final_window,
        exploitability_threshold=float(config["exploitability_threshold"]),
        save_final_checkpoints=save_final_checkpoints,
    )
    screening_rows = [result["summary"] for result in screening_results]
    confirmation_configs, screening_agg = choose_confirmation_configs(
        screening_summary_rows=screening_rows,
        baseline_config=baseline_config,
        candidate_configs=candidate_configs,
        top_k=int(config["confirmation_top_k"]),
    )

    _LOGGER.info("Selected confirmation configurations: %s", [
        cfg["config_label"] for cfg in confirmation_configs
    ])
    confirmation_results, confirmation_failed = run_stage(
        configs=confirmation_configs,
        seeds=config["confirmation_seeds"],
        stage="confirmation",
        num_iterations=int(config["confirmation_num_iterations"]),
        run_dir=run_dir,
        final_window=final_window,
        exploitability_threshold=float(config["exploitability_threshold"]),
        save_final_checkpoints=save_final_checkpoints,
    )
    all_results = [*screening_results, *confirmation_results]
    all_failed = [*screening_failed, *confirmation_failed]
    if not all_results:
        raise RuntimeError("All random-search runs failed; nothing to export.")

    confirmation_agg = aggregate_config_rows(
        [result["summary"] for result in confirmation_results], "confirmation"
    )
    export_info = export_search_results(
        results=all_results,
        run_dir=run_dir,
        config={
            **config,
            "sampled_candidate_configs": candidate_configs,
            "confirmation_configs": confirmation_configs,
        },
        baseline_config=baseline_config,
        candidate_configs=candidate_configs,
        confirmation_configs=confirmation_configs,
        screening_agg=screening_agg,
        confirmation_agg=confirmation_agg,
        failed=all_failed or None,
    )
    export_info["results"] = all_results
    export_info["failed"] = all_failed
    return export_info


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Leduc poker Deep CFR constrained random search."
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--screening-seeds", default=None)
    parser.add_argument("--confirmation-seeds", default=None)
    parser.add_argument("--use-extended-confirmation-seeds", action="store_true")
    parser.add_argument("--screening-iterations", type=int, default=None)
    parser.add_argument("--confirmation-iterations", type=int, default=None)
    parser.add_argument("--screening-random-configs", type=int, default=None)
    parser.add_argument("--confirmation-top-k", type=int, default=None)
    parser.add_argument("--master-seed", type=int, default=None)
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
    parser.add_argument("--exploitability-threshold", type=float, default=None)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--save-final-checkpoints", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    config = build_config(args)
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(Path(args.output_root), str(config["experiment_name"]))

    configure_run_logging(run_dir, verbose=args.verbose)
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Configuration: %s", config)

    try:
        export_info = run_staged_random_search(
            config=config,
            run_dir=run_dir,
            final_window=args.final_window,
            save_final_checkpoints=args.save_final_checkpoints,
        )
    except Exception as exc:  # pragma: no cover
        _LOGGER.exception("Random search failed: %s", exc)
        return 1

    _LOGGER.info("Per-run summary: %s", export_info["summary_csv"].resolve())
    _LOGGER.info("Checkpoint curves: %s", export_info["curve_csv"].resolve())
    _LOGGER.info("Confirmation summary: %s", export_info["confirmation_config_summary"].resolve())
    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
