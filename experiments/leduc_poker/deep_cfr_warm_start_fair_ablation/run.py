"""CLI for the fair Deep CFR warm-start/checkpoint ablation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pyspiel
from scipy import stats

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/deep_cfr_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/deep_cfr_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from deep_cfr_poker.constants import LEDUC_GAME_VALUE_PLAYER_0  # noqa: E402
from deep_cfr_poker.experiment_utils import (  # noqa: E402
    DEFAULT_FINAL_WINDOW,
    cleanup_training_memory,
    configure_run_logging,
    create_run_dir,
    final_window_mean,
    first_nodes_to_threshold,
    first_time_to_threshold,
    json_safe,
    normalised_auc,
    resolve_solver_batch_sizes,
    write_dict_rows_csv,
    write_experiment_metadata,
    write_failed_seeds,
)
from deep_cfr_poker.seeding import set_seed  # noqa: E402
from deep_cfr_poker.snapshots import package_full_checkpoint_filename  # noqa: E402
from deep_cfr_poker.solver import DeepCFRSolver, SolveResult  # noqa: E402

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, FULL_BASELINE_SEEDS  # noqa: E402
from .plotting import plot_warm_start_fair_ablation  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.warm_start_fair")


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
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


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "experiment_name": args.experiment_name,
        "total_iterations": args.iterations,
        "warm_start_boundary": args.warm_start_boundary,
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
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if int(config["warm_start_boundary"]) <= 0:
        raise ValueError("warm_start_boundary must be > 0")
    if int(config["warm_start_boundary"]) >= int(config["total_iterations"]):
        raise ValueError("warm_start_boundary must be less than total_iterations")
    return config


def _solver_kwargs(config: Mapping[str, object], *, num_iterations: int) -> dict:
    batch_size_advantage, batch_size_strategy = resolve_solver_batch_sizes(config)
    if isinstance(config, dict):
        config["batch_size_advantage"] = batch_size_advantage
        config["batch_size_strategy"] = batch_size_strategy
    return dict(
        policy_network_layers=tuple(config["policy_network_layers"]),
        advantage_network_layers=tuple(config["advantage_network_layers"]),
        num_iterations=int(num_iterations),
        num_traversals=int(config["num_traversals"]),
        learning_rate=float(config["learning_rate"]),
        batch_size_advantage=batch_size_advantage,
        batch_size_strategy=batch_size_strategy,
        memory_capacity=int(config["memory_capacity"]),
        reinitialize_advantage_networks=bool(config["reinitialize_advantage_networks"]),
        policy_network_train_steps=int(config["policy_network_train_steps"]),
        advantage_network_train_steps=int(config["advantage_network_train_steps"]),
        compute_exploitability=bool(config["compute_exploitability"]),
        policy_network_train_every=int(config["policy_network_train_every"]),
        evaluation_interval=int(config["evaluation_interval"]),
        policy_training_mode=str(config.get("policy_training_mode", "intermittent")),
        final_policy_network_train_steps=(
            int(config["final_policy_network_train_steps"])
            if config.get("final_policy_network_train_steps") is not None
            else None
        ),
    )


def _make_solver(game, config: Mapping[str, object], *, num_iterations: int) -> DeepCFRSolver:
    return DeepCFRSolver(game, **_solver_kwargs(config, num_iterations=num_iterations))


def _result_from_solve(
    *,
    seed: int,
    arm: str,
    solve_result: SolveResult,
    iteration_offset: int = 0,
    wall_clock_offset: float = 0.0,
) -> dict:
    diagnostics = {k: np.asarray(v) for k, v in solve_result.diagnostics.items()}
    iterations = diagnostics["iteration"].astype(int) + int(iteration_offset)
    wall_clock = diagnostics["wall_clock_seconds"].astype(float) + float(wall_clock_offset)
    values = np.asarray(solve_result.average_policy_value, dtype=np.float64)
    return {
        "seed": int(seed),
        "arm": str(arm),
        "iterations": iterations,
        "nodes_touched": np.asarray(solve_result.nodes_touched, dtype=np.float64),
        "wall_clock_seconds": wall_clock,
        "exploitability": np.asarray(solve_result.nash_conv, dtype=np.float64) / 2.0,
        "average_policy_value": values,
        "policy_value_error": np.abs(values - LEDUC_GAME_VALUE_PLAYER_0),
        "diagnostics": diagnostics,
    }


def _concat_results(seed: int, arm: str, first: dict, second: dict) -> dict:
    diagnostics = {}
    for key in sorted(set(first["diagnostics"]) | set(second["diagnostics"])):
        if key in first["diagnostics"] and key in second["diagnostics"]:
            diagnostics[key] = np.concatenate([first["diagnostics"][key], second["diagnostics"][key]])
        elif key in first["diagnostics"]:
            diagnostics[key] = first["diagnostics"][key]
        else:
            diagnostics[key] = second["diagnostics"][key]
    return {
        "seed": int(seed),
        "arm": str(arm),
        "iterations": np.concatenate([first["iterations"], second["iterations"]]),
        "nodes_touched": np.concatenate([first["nodes_touched"], second["nodes_touched"]]),
        "wall_clock_seconds": np.concatenate([first["wall_clock_seconds"], second["wall_clock_seconds"]]),
        "exploitability": np.concatenate([first["exploitability"], second["exploitability"]]),
        "average_policy_value": np.concatenate([first["average_policy_value"], second["average_policy_value"]]),
        "policy_value_error": np.concatenate([first["policy_value_error"], second["policy_value_error"]]),
        "diagnostics": diagnostics,
    }


def _summarise_result(result: dict, config: Mapping[str, object]) -> dict:
    exploitability = np.asarray(result["exploitability"], dtype=np.float64)
    policy_value_error = np.asarray(result["policy_value_error"], dtype=np.float64)
    return {
        "seed": int(result["seed"]),
        "arm": str(result["arm"]),
        "final_exploitability": float(exploitability[-1]),
        "best_exploitability": float(np.nanmin(exploitability)),
        "final_window_mean_exploitability": final_window_mean(exploitability),
        "normalised_exploitability_auc_by_nodes": normalised_auc(
            result["nodes_touched"], exploitability
        ),
        "normalised_exploitability_auc_by_iteration": normalised_auc(
            result["iterations"], exploitability
        ),
        "final_policy_value": float(result["average_policy_value"][-1]),
        "final_policy_value_error": float(policy_value_error[-1]),
        "best_policy_value_error": float(np.nanmin(policy_value_error)),
        "final_nodes_touched": float(result["nodes_touched"][-1]),
        "final_wall_clock_seconds": float(result["wall_clock_seconds"][-1]),
        "nodes_to_exploitability_threshold": first_nodes_to_threshold(
            result["nodes_touched"],
            exploitability,
            config["exploitability_threshold"],
        ),
        "seconds_to_exploitability_threshold": first_time_to_threshold(
            result["wall_clock_seconds"],
            exploitability,
            config["exploitability_threshold"],
        ),
        "policy_training_events": int(result["diagnostics"]["policy_training_events"][-1]),
        "policy_network_gradient_steps": int(result["diagnostics"]["policy_gradient_steps"][-1]),
        "advantage_network_gradient_steps_per_player": int(
            int(config["total_iterations"]) * int(config["advantage_network_train_steps"])
        ),
        "total_iterations": int(config["total_iterations"]),
        "evaluation_interval": int(config["evaluation_interval"]),
    }


def run_continuous_baseline(seed: int, config: Mapping[str, object]) -> dict:
    set_seed(seed)
    game = pyspiel.load_game(str(config["game_name"]))
    solver = _make_solver(game, config, num_iterations=int(config["total_iterations"]))
    start = time.perf_counter()
    solve_result = solver.solve()
    result = _result_from_solve(
        seed=seed,
        arm="baseline_continuous",
        solve_result=solve_result,
    )
    result["summary"] = _summarise_result(result, config)
    result["summary"]["total_outer_wall_clock_seconds"] = float(time.perf_counter() - start)
    del solver, solve_result, game
    cleanup_training_memory()
    return result


def run_warm_start_arm(seed: int, config: Mapping[str, object], run_dir: Path) -> dict:
    boundary = int(config["warm_start_boundary"])
    total = int(config["total_iterations"])
    remaining = total - boundary

    set_seed(seed)
    game = pyspiel.load_game(str(config["game_name"]))
    start = time.perf_counter()

    first_solver = _make_solver(game, config, num_iterations=boundary)
    first_solve = first_solver.solve()
    first_result = _result_from_solve(
        seed=seed,
        arm="warm_start",
        solve_result=first_solve,
    )

    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / package_full_checkpoint_filename(seed, boundary)
    first_solver.save_full_model(
        checkpoint_path,
        include_buffers=True,
        include_rng_state=True,
    )
    del first_solver, first_solve
    cleanup_training_memory()

    second_solver = _make_solver(game, config, num_iterations=remaining)
    second_solver.load_full_model(checkpoint_path, map_location="cpu")
    second_solve = second_solver.solve()
    wall_clock_offset = (
        float(first_result["wall_clock_seconds"][-1])
        if len(first_result["wall_clock_seconds"])
        else 0.0
    )
    second_result = _result_from_solve(
        seed=seed,
        arm="warm_start",
        solve_result=second_solve,
        iteration_offset=boundary,
        wall_clock_offset=wall_clock_offset,
    )

    result = _concat_results(seed, "warm_start", first_result, second_result)
    result["summary"] = _summarise_result(result, config)
    result["summary"]["total_outer_wall_clock_seconds"] = float(time.perf_counter() - start)
    result["summary"]["warm_start_boundary"] = boundary
    result["summary"]["checkpoint_path"] = str(checkpoint_path)
    del second_solver, second_solve, game, first_result, second_result
    cleanup_training_memory()
    return result


def _aggregate_by_arm(summary_rows: Sequence[dict]) -> dict:
    output = {}
    arms = sorted({row["arm"] for row in summary_rows})
    for arm in arms:
        rows = [row for row in summary_rows if row["arm"] == arm]
        output[arm] = {}
        for field in rows[0].keys():
            if field in {"seed", "arm", "checkpoint_path"}:
                continue
            try:
                vals = np.asarray([row[field] for row in rows], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            finite = vals[np.isfinite(vals)]
            if finite.size:
                output[arm][field] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
                    "n": int(finite.size),
                }
    return output


def _paired_rows(results: Sequence[dict], seeds: Sequence[int]) -> List[dict]:
    summaries = {
        (int(result["seed"]), str(result["arm"])): result["summary"]
        for result in results
    }
    rows = []
    for seed in seeds:
        baseline = summaries.get((int(seed), "baseline_continuous"))
        warm = summaries.get((int(seed), "warm_start"))
        if baseline is None or warm is None:
            continue
        rows.append(
            {
                "seed": int(seed),
                "delta_final_exploitability_warm_minus_baseline": float(
                    warm["final_exploitability"] - baseline["final_exploitability"]
                ),
                "delta_best_exploitability_warm_minus_baseline": float(
                    warm["best_exploitability"] - baseline["best_exploitability"]
                ),
                "delta_final_window_exploitability_warm_minus_baseline": float(
                    warm["final_window_mean_exploitability"]
                    - baseline["final_window_mean_exploitability"]
                ),
                "delta_auc_nodes_warm_minus_baseline": float(
                    warm["normalised_exploitability_auc_by_nodes"]
                    - baseline["normalised_exploitability_auc_by_nodes"]
                ),
                "delta_final_policy_value_error_warm_minus_baseline": float(
                    warm["final_policy_value_error"] - baseline["final_policy_value_error"]
                ),
                "baseline_final_exploitability": float(baseline["final_exploitability"]),
                "warm_start_final_exploitability": float(warm["final_exploitability"]),
                "baseline_final_nodes_touched": float(baseline["final_nodes_touched"]),
                "warm_start_final_nodes_touched": float(warm["final_nodes_touched"]),
                "baseline_policy_training_events": int(baseline["policy_training_events"]),
                "warm_start_policy_training_events": int(warm["policy_training_events"]),
            }
        )
    return rows


def _paired_aggregate(rows: Sequence[dict]) -> dict:
    output = {}
    if not rows:
        return output
    for field in rows[0].keys():
        if not field.startswith("delta_"):
            continue
        vals = np.asarray([row[field] for row in rows], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            output[field] = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
                "n": int(finite.size),
                "fraction_warm_start_better": float(np.mean(finite < 0.0)),
            }
    return output


def _paired_curve_rows(results: Sequence[dict], seeds: Sequence[int]) -> List[dict]:
    by_seed_arm = {
        (int(result["seed"]), str(result["arm"])): result
        for result in results
    }
    rows = []
    for seed in seeds:
        baseline = by_seed_arm.get((int(seed), "baseline_continuous"))
        warm = by_seed_arm.get((int(seed), "warm_start"))
        if baseline is None or warm is None:
            continue
        baseline_by_iteration = {
            int(iteration): i for i, iteration in enumerate(baseline["iterations"])
        }
        for warm_i, iteration in enumerate(warm["iterations"]):
            iteration = int(iteration)
            base_i = baseline_by_iteration.get(iteration)
            if base_i is None:
                continue
            rows.append(
                {
                    "seed": int(seed),
                    "iteration": iteration,
                    "delta_exploitability_warm_minus_baseline": float(
                        warm["exploitability"][warm_i] - baseline["exploitability"][base_i]
                    ),
                    "delta_policy_value_error_warm_minus_baseline": float(
                        warm["policy_value_error"][warm_i]
                        - baseline["policy_value_error"][base_i]
                    ),
                }
            )
    return rows


def _curve_rows(results: Sequence[dict]) -> List[dict]:
    rows = []
    for result in results:
        for i, iteration in enumerate(result["iterations"]):
            rows.append(
                {
                    "seed": int(result["seed"]),
                    "arm": str(result["arm"]),
                    "iteration": int(iteration),
                    "nodes_touched": float(result["nodes_touched"][i]),
                    "wall_clock_seconds": float(result["wall_clock_seconds"][i]),
                    "exploitability": float(result["exploitability"][i]),
                    "average_policy_value": float(result["average_policy_value"][i]),
                    "policy_value_error": float(result["policy_value_error"][i]),
                    "policy_training_events": int(result["diagnostics"]["policy_training_events"][i]),
                    "policy_gradient_steps": int(result["diagnostics"]["policy_gradient_steps"][i]),
                    "strategy_buffer_size": int(result["diagnostics"]["strategy_buffer_size"][i]),
                    "advantage_buffer_size_player_0": int(result["diagnostics"]["advantage_buffer_size_player_0"][i]),
                    "advantage_buffer_size_player_1": int(result["diagnostics"]["advantage_buffer_size_player_1"][i]),
                    "policy_loss": float(result["diagnostics"]["policy_loss"][i]),
                    "policy_grad_norm": float(result["diagnostics"]["policy_grad_norm"][i]),
                    "advantage_grad_norm_player_0": float(result["diagnostics"]["advantage_grad_norm_player_0"][i]),
                    "advantage_grad_norm_player_1": float(result["diagnostics"]["advantage_grad_norm_player_1"][i]),
                }
            )
    return rows


def _export_npz(results: Sequence[dict], run_dir: Path, seeds: Sequence[int]) -> Path:
    payload = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "iterations": np.asarray(results[0]["iterations"], dtype=np.int64),
    }
    for arm in ("baseline_continuous", "warm_start"):
        arm_results = [r for r in results if r["arm"] == arm]
        for key in (
            "exploitability",
            "policy_value_error",
            "nodes_touched",
            "wall_clock_seconds",
        ):
            payload[f"{arm}_{key}"] = np.vstack(
                [np.asarray(r[key], dtype=np.float64) for r in arm_results]
            )
    path = run_dir / "ablation_curves.npz"
    np.savez_compressed(path, **payload)
    return path


def export_results(
    results: Sequence[dict],
    run_dir: Path,
    config: dict,
    seeds: Sequence[int],
    failed: Optional[Sequence[dict]] = None,
) -> dict:
    summary_rows = [result["summary"] for result in results]
    curve_rows = _curve_rows(results)
    paired_rows = _paired_rows(results, seeds)
    paired_curve_rows = _paired_curve_rows(results, seeds)
    aggregate = {"by_arm": _aggregate_by_arm(summary_rows)}
    paired_aggregate = _paired_aggregate(paired_rows)

    summary_csv = write_dict_rows_csv(summary_rows, run_dir / "seed_summary.csv")
    curve_csv = write_dict_rows_csv(curve_rows, run_dir / "checkpoint_curves.csv")
    paired_csv = write_dict_rows_csv(paired_rows, run_dir / "paired_summary.csv")
    paired_curve_csv = write_dict_rows_csv(
        paired_curve_rows, run_dir / "paired_checkpoint_differences.csv"
    )
    aggregate_path = run_dir / "aggregate_summary.json"
    paired_aggregate_path = run_dir / "paired_aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate), f, indent=2)
    with open(paired_aggregate_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(paired_aggregate), f, indent=2)

    npz_path = _export_npz(results, run_dir, seeds)
    write_experiment_metadata(
        run_dir,
        config=config,
        seeds=seeds,
        completed_seeds=sorted({int(r["seed"]) for r in results}),
        extra={
            "full_baseline_seeds": FULL_BASELINE_SEEDS,
            "interpretation": (
                "Paired fair warm-start ablation; warm-start minus baseline > 0 "
                "means warm-start is worse for that metric."
            ),
            "summary_csv": str(summary_csv),
            "checkpoint_curves_csv": str(curve_csv),
            "paired_summary_csv": str(paired_csv),
            "paired_checkpoint_differences_csv": str(paired_curve_csv),
            "aggregate_summary_json": str(aggregate_path),
            "paired_aggregate_summary_json": str(paired_aggregate_path),
            "ablation_curves_npz": str(npz_path),
        },
    )
    if failed:
        write_failed_seeds(run_dir, failed)

    return {
        "summary_csv": summary_csv,
        "curve_csv": curve_csv,
        "paired_summary_csv": paired_csv,
        "paired_checkpoint_differences_csv": paired_curve_csv,
        "aggregate_summary": aggregate_path,
        "paired_aggregate_summary": paired_aggregate_path,
        "ablation_curves_npz": npz_path,
        "paired_rows": paired_rows,
        "paired_curve_rows": paired_curve_rows,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Leduc poker Deep CFR fair warm-start ablation."
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warm-start-boundary", type=int, default=None)
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
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    configure_run_logging(run_dir, verbose=args.verbose)
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Configuration: %s", config)
    _LOGGER.info("Seeds: %s", seeds)

    results = []
    failed = []
    for seed in seeds:
        _LOGGER.info("Running seed %s baseline continuous", seed)
        try:
            results.append(run_continuous_baseline(seed, config))
        except Exception as exc:  # pragma: no cover
            _LOGGER.exception("Seed %s baseline failed: %s", seed, exc)
            failed.append(
                {
                    "seed": int(seed),
                    "arm": "baseline_continuous",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        _LOGGER.info("Running seed %s warm-start", seed)
        try:
            results.append(run_warm_start_arm(seed, config, run_dir))
        except Exception as exc:  # pragma: no cover
            _LOGGER.exception("Seed %s warm-start failed: %s", seed, exc)
            failed.append(
                {
                    "seed": int(seed),
                    "arm": "warm_start",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    if not results:
        _LOGGER.error("All runs failed; nothing to export.")
        return 1

    export_info = export_results(results, run_dir, config, seeds, failed=failed or None)
    plot_warm_start_fair_ablation(
        results,
        export_info["paired_curve_rows"],
        export_info["paired_rows"],
        run_dir,
        warm_start_boundary=int(config["warm_start_boundary"]),
        average_policy_value_target=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
    )

    _LOGGER.info("Completed %d/%d arm runs", len(results), len(seeds) * 2)
    if failed:
        _LOGGER.warning("%d run(s) failed; see failed_seeds.json", len(failed))
    _LOGGER.info("Per-seed summary: %s", export_info["summary_csv"].resolve())
    _LOGGER.info("Checkpoint curves: %s", export_info["curve_csv"].resolve())
    _LOGGER.info("Paired summary: %s", export_info["paired_summary_csv"].resolve())
    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
