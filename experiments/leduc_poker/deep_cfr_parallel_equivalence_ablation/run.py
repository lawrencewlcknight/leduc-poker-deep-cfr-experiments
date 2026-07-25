"""CLI for Experiment 25's sequential/parallel Deep CFR comparison."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import traceback
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/deep_cfr_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/deep_cfr_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from tqdm import tqdm  # noqa: E402

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    configure_run_logging,
    create_run_dir,
    game_value_player_0,
    json_safe,
    run_single_seed,
    summarise_numeric_fields,
    write_dict_rows_csv,
    write_failed_seeds,
)
from deep_cfr_poker.parallel_utils import equivalence_summary  # noqa: E402
from deep_cfr_poker.plotting import set_chart_title  # noqa: E402
from experiments.leduc_poker.deep_cfr_composite_hp_ablation_common import (  # noqa: E402
    _augment_result,
    _variant_config,
    build_arg_parser,
    export_ablation_results,
    parse_seeds,
)
from experiments.leduc_poker.deep_cfr_replay_averaging_ablation.run import (  # noqa: E402
    _filter_variants,
    _str2bool,
    parse_int_tuple,
    parse_variant_ids,
)

from .config import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
    PARALLEL_NUM_WORKERS,
    VARIANTS,
)


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.parallel_equivalence")

RUNTIME_FIELDS = (
    "solver_initialization_seconds",
    "elapsed_seconds",
    "end_to_end_seconds",
    "traversal_collection_seconds",
)

EQUIVALENCE_MARGINS = {
    "final_exploitability": FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    "final_policy_value": FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
}


def build_parser():
    parser = build_arg_parser(
        "Compare current-best sequential and Ray-parallel Leduc Deep CFR."
    )
    parser.add_argument("--parallel-num-workers", type=int, default=None)
    parser.add_argument("--parallel-ray-address", default=None)
    parser.add_argument("--parallel-log-to-driver", type=_str2bool, default=None)
    parser.add_argument("--parallel-worker-memory-capacity", type=int, default=None)
    parser.add_argument("--parallel-ray-object-store-memory", type=int, default=None)
    parser.add_argument(
        "--replay-buffer-type",
        choices=("python", "compact"),
        default=None,
        help=(
            "Replay storage backend. Experiment 25 defaults to compact array "
            "storage to keep the parallel comparison memory-feasible."
        ),
    )
    parser.add_argument(
        "--disable-subprocess-isolation",
        action="store_true",
        help=(
            "Run all variant/seed trainings in the parent process. By default "
            "each independent training runs in a fresh Python worker so replay "
            "buffers, PyTorch allocator state, and Ray actors are fully "
            "released between runs."
        ),
    )
    parser.add_argument("--worker-input-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-pickle", default=None, help=argparse.SUPPRESS)
    return parser


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    variant_ids = parse_variant_ids(args.variant_ids)
    variants = (
        list(VARIANTS)
        if variant_ids is None
        else _filter_variants(VARIANTS, variant_ids)
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
        "final_policy_network_train_steps": args.final_policy_network_train_steps,
        "compute_exploitability": args.compute_exploitability,
        "target_standardize_epsilon": args.target_standardize_epsilon,
        "priority_alpha": args.priority_alpha,
        "priority_epsilon": args.priority_epsilon,
        "baseline_variant_id": args.baseline_variant_id,
        "parallel_num_workers": args.parallel_num_workers,
        "parallel_ray_address": args.parallel_ray_address,
        "parallel_log_to_driver": args.parallel_log_to_driver,
        "parallel_worker_memory_capacity": args.parallel_worker_memory_capacity,
        "parallel_ray_object_store_memory": args.parallel_ray_object_store_memory,
        "replay_buffer_type": args.replay_buffer_type,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["ablation_variants"] = tuple(variants)
    variant_ids = {str(v["variant_id"]) for v in variants}
    if str(config["baseline_variant_id"]) not in variant_ids:
        config["baseline_variant_id"] = str(variants[0]["variant_id"])
    return config


def _execution_variant_config(base_config: Mapping[str, object], variant: Mapping[str, object]) -> dict:
    config = _variant_config(dict(base_config), variant)
    backend = str(config.get("execution_backend", "sequential"))
    if backend == "ray_parallel":
        config["parallel_num_workers"] = int(
            base_config.get("parallel_num_workers", PARALLEL_NUM_WORKERS)
        )
    else:
        config["parallel_num_workers"] = 1
    config["parallel_ray_address"] = base_config.get("parallel_ray_address")
    config["parallel_log_to_driver"] = bool(
        base_config.get("parallel_log_to_driver", False)
    )
    return config


def _augment_parallel_result(
    result: dict,
    variant_config: Mapping[str, object],
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    result = _augment_result(
        result,
        variant_config,
        final_window=final_window,
        exploitability_threshold=exploitability_threshold,
    )
    result["execution_backend"] = str(variant_config.get("execution_backend", ""))
    result["parallel_num_workers"] = int(variant_config.get("parallel_num_workers", 1))
    result["replay_buffer_type"] = str(variant_config.get("replay_buffer_type", ""))
    result["summary"]["execution_backend"] = result["execution_backend"]
    result["summary"]["parallel_num_workers"] = result["parallel_num_workers"]
    result["summary"]["replay_buffer_type"] = result["replay_buffer_type"]
    result["summary"]["parallel_worker_memory_capacity"] = (
        int(variant_config["parallel_worker_memory_capacity"])
        if variant_config.get("parallel_worker_memory_capacity") is not None
        else float("nan")
    )
    result["summary"]["parallel_ray_object_store_memory"] = (
        int(variant_config["parallel_ray_object_store_memory"])
        if variant_config.get("parallel_ray_object_store_memory") is not None
        else float("nan")
    )
    return result


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _worker_stem(variant_id: str, seed: int) -> str:
    return f"{_safe_stem(str(variant_id))}_seed_{int(seed)}"


def _write_worker_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(payload), f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_worker_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _tail_text(path: Path, *, max_chars: int = 12000) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _run_worker(worker_input_json: str, worker_output_pickle: str) -> int:
    output_path = Path(worker_output_pickle)
    try:
        with open(worker_input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        variant_config = dict(payload["config"])
        final_window = int(payload["final_window"])
        result = run_single_seed(
            int(payload["seed"]),
            variant_config,
            export_dir=Path(payload["export_dir"]),
            save_final_checkpoint=bool(payload.get("save_final_checkpoint", False)),
            final_window=final_window,
        )
        result = _augment_parallel_result(
            result,
            variant_config,
            final_window,
            float(payload["exploitability_threshold"]),
        )
        _write_worker_pickle(output_path, {"ok": True, "result": result})
        return 0
    except Exception as exc:  # pragma: no cover - operational safety net
        _write_worker_pickle(
            output_path,
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


def _run_seed_variant_subprocess(
    seed: int,
    variant_config: Mapping[str, object],
    run_dir: Path,
    *,
    final_window: int,
    exploitability_threshold: float,
    save_final_checkpoint: bool,
) -> dict:
    run_dir = Path(run_dir)
    stem = _worker_stem(str(variant_config["variant_id"]), int(seed))
    worker_input = run_dir / "worker_inputs" / f"{stem}.json"
    worker_output = run_dir / "worker_results" / f"{stem}.pickle"
    worker_log = run_dir / "worker_logs" / f"{stem}.log"
    worker_export_dir = run_dir / "worker_exports" / stem
    worker_input.parent.mkdir(parents=True, exist_ok=True)
    worker_log.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": int(seed),
        "config": json_safe(dict(variant_config)),
        "export_dir": str(worker_export_dir if save_final_checkpoint else run_dir),
        "save_final_checkpoint": bool(save_final_checkpoint),
        "final_window": int(final_window),
        "exploitability_threshold": float(exploitability_threshold),
    }
    with open(worker_input, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.deep_cfr_parallel_equivalence_ablation.run",
        "--worker-input-json",
        str(worker_input),
        "--worker-output-pickle",
        str(worker_output),
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    env.setdefault("MPLCONFIGDIR", "/private/tmp/deep_cfr_poker_matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp/deep_cfr_poker_cache")
    env.setdefault("PYTHONUNBUFFERED", "1")
    with open(worker_log, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    try:
        worker_payload = (
            _read_worker_pickle(worker_output) if worker_output.exists() else None
        )
    except Exception as exc:
        raise RuntimeError(
            f"Worker wrote an unreadable result at {worker_output}: {exc}. "
            f"See {worker_log}.\n{_tail_text(worker_log)}"
        ) from exc
    if (
        completed.returncode == 0
        and worker_payload is not None
        and bool(worker_payload.get("ok", False))
    ):
        return worker_payload["result"]

    log_tail = _tail_text(worker_log)
    if worker_payload is None:
        raise RuntimeError(
            f"Worker failed with exit code {completed.returncode} and did not "
            f"write {worker_output}. See {worker_log}.\n{log_tail}"
        )
    raise RuntimeError(
        f"Worker failed with exit code {completed.returncode}: "
        f"{worker_payload.get('error', 'unknown error')}. See {worker_log}.\n"
        f"{worker_payload.get('traceback', '')}\n{log_tail}"
    )


def _run_seed_variant(
    seed: int,
    variant_config: Mapping[str, object],
    run_dir: Path,
    *,
    subprocess_isolation_enabled: bool,
    final_window: int,
    exploitability_threshold: float,
    save_final_checkpoint: bool,
) -> dict:
    if subprocess_isolation_enabled:
        return _run_seed_variant_subprocess(
            seed,
            variant_config,
            run_dir,
            final_window=final_window,
            exploitability_threshold=exploitability_threshold,
            save_final_checkpoint=save_final_checkpoint,
        )
    result = run_single_seed(
        seed,
        dict(variant_config),
        export_dir=run_dir,
        save_final_checkpoint=save_final_checkpoint,
        final_window=final_window,
    )
    return _augment_parallel_result(
        result,
        variant_config,
        final_window,
        exploitability_threshold,
    )


def _results_for_variant(results: Sequence[dict], variant_id: str) -> List[dict]:
    return [r for r in results if str(r["variant_id"]) == str(variant_id)]


def _stack(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [np.asarray(result[key], dtype=np.float64) for result in results]
    return np.vstack(arrays) if arrays else np.empty((0, 0))


def _stack_diag(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [
        np.asarray(result["diagnostics"].get(key, []), dtype=np.float64)
        for result in results
    ]
    arrays = [array for array in arrays if array.size]
    return np.vstack(arrays) if arrays else np.empty((0, 0))


def _mean_and_se(matrix: np.ndarray):
    if matrix.size == 0:
        return np.array([]), np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
    se = (
        stats.sem(matrix, axis=0, nan_policy="omit")
        if matrix.shape[0] > 1
        else np.zeros(matrix.shape[1])
    )
    return mean, se


def _plot_curve_by_nodes(
    results: Sequence[dict],
    run_dir: Path,
    *,
    variants: Sequence[Mapping[str, object]],
    key: str,
    ylabel: str,
    title: str,
    filename: str,
    target_line: Optional[float] = None,
    diagnostic_key: Optional[str] = None,
) -> None:
    labels = {str(v["variant_id"]): str(v.get("label", v["variant_id"])) for v in variants}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = False
    for variant in variants:
        variant_id = str(variant["variant_id"])
        subset = _results_for_variant(results, variant_id)
        if diagnostic_key is None:
            y_mean, y_se = _mean_and_se(_stack(subset, key))
        else:
            y_mean, y_se = _mean_and_se(_stack_diag(subset, diagnostic_key))
        x_mean, _ = _mean_and_se(_stack(subset, "nodes_touched"))
        if y_mean.size == 0 or x_mean.size == 0:
            continue
        ax.plot(x_mean, y_mean, linewidth=2, label=labels[variant_id])
        ax.fill_between(x_mean, y_mean - y_se, y_mean + y_se, alpha=0.15)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    if target_line is not None:
        ax.axhline(target_line, linestyle="--", label="Reference")
    ax.set_xlabel("Nodes Touched")
    ax.set_ylabel(ylabel)
    set_chart_title(ax, title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_parallel_equivalence(
    results: Sequence[dict],
    run_dir: Path,
    *,
    variants: Sequence[Mapping[str, object]],
    exploitability_threshold: float,
    average_policy_value_target: float,
) -> None:
    run_dir = Path(run_dir)
    _plot_curve_by_nodes(
        results,
        run_dir,
        variants=variants,
        key="exploitability",
        ylabel="Exploitability (NashConv/2)",
        title="Deep CFR sequential versus Ray parallel: exploitability",
        filename="exploitability_by_nodes.png",
        target_line=exploitability_threshold,
    )
    _plot_curve_by_nodes(
        results,
        run_dir,
        variants=variants,
        key="average_policy_value",
        ylabel="Average policy value for player 0",
        title="Deep CFR sequential versus Ray parallel: average policy value",
        filename="average_policy_value_by_nodes.png",
        target_line=average_policy_value_target,
    )
    _plot_curve_by_nodes(
        results,
        run_dir,
        variants=variants,
        key="policy_value_error",
        ylabel=r"$|v(\sigma)-v^*|$",
        title="Deep CFR sequential versus Ray parallel: policy-value error",
        filename="policy_value_error_by_nodes.png",
    )
    _plot_curve_by_nodes(
        results,
        run_dir,
        variants=variants,
        key="",
        diagnostic_key="cumulative_traversal_collection_seconds",
        ylabel="Cumulative traversal-collection time (seconds)",
        title="Deep CFR sequential versus Ray parallel: collection time",
        filename="traversal_collection_seconds_by_nodes.png",
    )

    variant_ids = [str(v["variant_id"]) for v in variants]
    labels = {str(v["variant_id"]): str(v.get("label", v["variant_id"])) for v in variants}
    x_pos = np.arange(len(variant_ids))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, field, title in zip(
        axes,
        ("elapsed_seconds", "end_to_end_seconds", "traversal_collection_seconds"),
        ("Training loop", "End to end", "Traversal collection"),
    ):
        means = []
        ses = []
        for variant_id in variant_ids:
            vals = np.asarray(
                [
                    row["summary"][field]
                    for row in _results_for_variant(results, variant_id)
                ],
                dtype=np.float64,
            )
            finite = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite)) if finite.size else float("nan"))
            ses.append(float(stats.sem(finite)) if finite.size > 1 else 0.0)
        ax.bar(x_pos, means, yerr=ses, capsize=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels[v] for v in variant_ids], rotation=25, ha="right")
        ax.set_ylabel("Seconds")
        set_chart_title(ax, title)
        ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(run_dir / "runtime_by_variant.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _paired_parallel_rows(results: Sequence[dict], baseline_variant_id: str) -> List[dict]:
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
            row = {
                "seed": int(seed),
                "baseline_variant_id": str(baseline_variant_id),
                "variant_id": variant_id,
            }
            for field in (
                "final_exploitability",
                "best_exploitability",
                "final_policy_value",
                "final_policy_value_error",
                "final_nodes_touched",
                *RUNTIME_FIELDS,
            ):
                row[f"baseline_{field}"] = baseline.get(field, float("nan"))
                row[f"variant_{field}"] = comparison.get(field, float("nan"))
                row[f"delta_{field}_vs_baseline"] = float(
                    comparison.get(field, float("nan")) - baseline.get(field, float("nan"))
                )
            for field in RUNTIME_FIELDS:
                denom = float(comparison.get(field, float("nan")))
                numer = float(baseline.get(field, float("nan")))
                row[f"{field}_speedup"] = (
                    numer / denom if np.isfinite(numer) and np.isfinite(denom) and denom > 0 else float("nan")
                )
            rows.append(row)
    return rows


def write_parallel_equivalence_outputs(
    results: Sequence[dict],
    run_dir: Path,
    *,
    baseline_variant_id: str,
) -> dict:
    paired_rows = _paired_parallel_rows(results, baseline_variant_id)
    paired_csv = write_dict_rows_csv(
        paired_rows,
        Path(run_dir) / "paired_parallel_equivalence_and_timing.csv",
    )
    timing_summary = summarise_numeric_fields(paired_rows)
    equivalence = {}
    for field, margin in EQUIVALENCE_MARGINS.items():
        equivalence[field] = equivalence_summary(
            [row[f"delta_{field}_vs_baseline"] for row in paired_rows],
            margin,
        )
    summary = {
        "paired_difference_convention": (
            "variant minus sequential baseline; speedup is sequential seconds "
            "divided by parallel seconds"
        ),
        "equivalence_margins": EQUIVALENCE_MARGINS,
        "equivalence_summary": equivalence,
        "timing_summary": timing_summary,
    }
    summary_path = Path(run_dir) / "parallel_equivalence_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2)
    return {
        "paired_parallel_csv": paired_csv,
        "parallel_equivalence_summary": summary_path,
        "paired_parallel_rows": paired_rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_input_json or args.worker_output_pickle:
        if not args.worker_input_json or not args.worker_output_pickle:
            parser.error(
                "--worker-input-json and --worker-output-pickle must be used together"
            )
        return _run_worker(args.worker_input_json, args.worker_output_pickle)

    config = build_config(args)
    seeds = parse_seeds(args.seeds, DEFAULT_SEEDS)
    subprocess_isolation_enabled = not bool(args.disable_subprocess_isolation)
    config["subprocess_isolation_enabled"] = bool(subprocess_isolation_enabled)
    config["worker_results_dir"] = "worker_results"
    config["worker_logs_dir"] = "worker_logs"

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(Path(args.output_root), str(config["experiment_name"]))

    configure_run_logging(run_dir, verbose=args.verbose)
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Configuration: %s", config)
    _LOGGER.info("Seeds: %s", seeds)
    _LOGGER.info("Subprocess isolation enabled: %s", subprocess_isolation_enabled)

    results = []
    failed = []
    for variant in config["ablation_variants"]:
        variant_config = _execution_variant_config(config, variant)
        _LOGGER.info(
            "Running %s: %s",
            variant_config["variant_id"],
            variant_config.get("label", variant_config["variant_id"]),
        )
        for seed in tqdm(seeds, desc=str(variant_config["variant_id"])):
            try:
                result = _run_seed_variant(
                    seed,
                    variant_config,
                    run_dir,
                    subprocess_isolation_enabled=subprocess_isolation_enabled,
                    final_window=args.final_window,
                    exploitability_threshold=float(config["exploitability_threshold"]),
                    save_final_checkpoint=args.save_final_checkpoints,
                )
                results.append(result)
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
        results,
        run_dir,
        config,
        seeds,
        failed=failed or None,
        experiment_note=(
            "Experiment 25 compares the current-best sequential Deep CFR "
            "configuration with the same learner using Ray-parallel traversal "
            "collection. Learning-quality charts should be read by nodes "
            "touched; timing outputs report solver initialisation, training "
            "loop, end-to-end, and traversal-collection seconds. Variant/seed "
            "runs are isolated in subprocesses by default so replay memory, "
            "PyTorch allocator state, and Ray actors are released after each "
            "independent run."
        ),
    )
    parallel_info = write_parallel_equivalence_outputs(
        results,
        run_dir,
        baseline_variant_id=str(config["baseline_variant_id"]),
    )
    plot_parallel_equivalence(
        results,
        run_dir,
        variants=config["ablation_variants"],
        exploitability_threshold=float(config["exploitability_threshold"]),
        average_policy_value_target=game_value_player_0(config),
    )
    if failed:
        write_failed_seeds(run_dir, failed)

    _LOGGER.info(
        "Completed %d/%d runs",
        len(results),
        len(seeds) * len(config["ablation_variants"]),
    )
    if failed:
        _LOGGER.warning("%d run(s) failed; see failed_seeds.json", len(failed))
    _LOGGER.info("Per-seed summary: %s", export_info["summary_csv"].resolve())
    _LOGGER.info("Checkpoint curves: %s", export_info["curve_csv"].resolve())
    _LOGGER.info(
        "Parallel equivalence summary: %s",
        parallel_info["parallel_equivalence_summary"].resolve(),
    )
    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())
    if failed:
        _LOGGER.error(
            "Experiment completed with %d failed variant/seed run(s). "
            "Returning non-zero status so Batch marks the comparison failed.",
            len(failed),
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
