"""Paired conventional Deep CFR and Single Deep CFR evaluation on Leduc."""

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
from typing import List, Optional, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/deep_cfr_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/deep_cfr_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python import policy as osp_policy  # noqa: E402
from open_spiel.python.algorithms import expected_game_score, exploitability  # noqa: E402

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    cleanup_training_memory,
    configure_run_logging,
    create_run_dir,
    game_value_player_0,
    json_safe,
    make_solver,
    normalised_auc,
    summarise_numeric_fields,
    write_dict_rows_csv,
    write_experiment_metadata,
    write_failed_seeds,
)
from deep_cfr_poker.plotting import set_chart_title  # noqa: E402
from deep_cfr_poker.sd_cfr import (  # noqa: E402
    SDCFRArchive,
    exact_average_policies_at_checkpoints,
)
from deep_cfr_poker.seeding import set_seed  # noqa: E402

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.single_deep_cfr")


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_seeds(value: Optional[str]) -> List[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_layers(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_weightings(value: Optional[str]):
    if value is None:
        return None
    weightings = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    invalid = sorted(set(weightings) - {"linear", "uniform"})
    if invalid:
        raise argparse.ArgumentTypeError(
            "SD-CFR weightings must be a comma-separated subset of "
            f"'uniform,linear'; invalid values: {invalid}"
        )
    if not weightings:
        raise argparse.ArgumentTypeError("At least one SD-CFR weighting is required")
    if len(set(weightings)) != len(weightings):
        raise argparse.ArgumentTypeError("SD-CFR weightings must be unique")
    return weightings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 28: train one Deep CFR trajectory per seed and compare "
            "its fitted average-policy network with the exact SD-CFR historical "
            "advantage-network mixture."
        )
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Write directly to this directory instead of creating a timestamped run.",
    )
    parser.add_argument(
        "--aggregate-workers-root",
        default=None,
        help=(
            "Do not train; aggregate durable seed result JSON files found below "
            "this directory. Used by the parallel cloud launcher."
        ),
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--advantage-network-layers", default=None)
    parser.add_argument(
        "--reinitialize-advantage-networks", type=_str2bool, default=None
    )
    parser.add_argument(
        "--sd-cfr-weightings",
        type=_parse_weightings,
        default=None,
        help=(
            "Comma-separated SD-CFR averaging rules. Defaults to uniform,linear: "
            "uniform is matched to the selected Deep CFR candidate and linear is "
            "canonical SD-CFR."
        ),
    )
    parser.add_argument(
        "--save-sd-cfr-archives", type=_str2bool, default=None
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_config(args) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "evaluation_interval": args.evaluation_interval,
        "learning_rate": args.learning_rate,
        "memory_capacity": args.memory_capacity,
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "policy_network_train_steps": args.policy_network_train_steps,
        "policy_network_train_every": args.policy_network_train_every,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "policy_network_layers": _parse_layers(args.policy_network_layers),
        "advantage_network_layers": _parse_layers(args.advantage_network_layers),
        "reinitialize_advantage_networks": args.reinitialize_advantage_networks,
        "sd_cfr_weightings": args.sd_cfr_weightings,
        "save_sd_cfr_archives": args.save_sd_cfr_archives,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["compute_exploitability"] = True
    config["policy_network_train_every"] = int(
        config.get("policy_network_train_every", config["evaluation_interval"])
    )
    config["sd_cfr_weightings"] = tuple(config["sd_cfr_weightings"])
    primary = str(config["primary_sd_cfr_weighting"])
    if primary not in config["sd_cfr_weightings"]:
        raise ValueError(
            "primary_sd_cfr_weighting must be included in sd_cfr_weightings"
        )
    return config


def _evaluate_policy(game, policy, value_target: float) -> dict:
    tabular = osp_policy.tabular_policy_from_callable(
        game, policy.action_probabilities
    )
    nash_conv = float(exploitability.nash_conv(game, tabular))
    value = float(
        expected_game_score.policy_value(
            game.new_initial_state(), [tabular] * game.num_players()
        )[0]
    )
    return {
        "nash_conv": nash_conv,
        "exploitability": nash_conv / 2.0,
        "policy_value": value,
        "policy_value_error": abs(value - value_target),
    }


def _write_json_atomic(path: Path, payload) -> None:
    """Write JSON through a temporary file so periodic uploads see whole files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    temporary.replace(path)


def _persist_seed_curves(seed: int, curves: Sequence[dict], run_dir: Path) -> Path:
    """Persist expensive checkpoint measurements before summary calculations."""
    path = run_dir / "seed_results" / f"seed_{seed}_checkpoint_curves.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_dict_rows_csv(curves, path)
    return path


def _persist_seed_result(result: dict, run_dir: Path) -> Path:
    path = run_dir / "seed_results" / f"seed_{result['seed']}_result.json"
    _write_json_atomic(path, result)
    return path


def _load_worker_results(workers_root: Path, expected_seeds: Sequence[int]) -> list[dict]:
    """Load one durable result per seed from independently executed workers."""
    results_by_seed = {}
    for path in sorted(workers_root.glob("**/seed_results/seed_*_result.json")):
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        seed = int(result["seed"])
        if seed in results_by_seed:
            raise ValueError(
                f"Duplicate durable result for seed {seed}: "
                f"{results_by_seed[seed][0]} and {path}"
            )
        results_by_seed[seed] = (path, result)

    missing = sorted(set(int(seed) for seed in expected_seeds) - set(results_by_seed))
    if missing:
        raise FileNotFoundError(
            f"Missing durable Experiment 28 result(s) for seed(s): {missing}"
        )
    unexpected = sorted(set(results_by_seed) - set(int(seed) for seed in expected_seeds))
    if unexpected:
        raise ValueError(f"Unexpected Experiment 28 seed result(s): {unexpected}")
    return [results_by_seed[int(seed)][1] for seed in expected_seeds]


def _run_seed(seed: int, config: dict, run_dir: Path) -> dict:
    set_seed(seed)
    game = pyspiel.load_game(str(config["game_name"]))
    solver = make_solver(game, config)
    archive = SDCFRArchive.from_solver(
        solver,
        game_name=str(config["game_name"]),
        metadata={
            "seed": int(seed),
            "experiment_name": str(config["experiment_name"]),
            "supported_averaging_weightings": tuple(config["sd_cfr_weightings"]),
            "capture_phase": "immediately_after_each_player_advantage_update",
        },
    )

    try:
        training_start = time.perf_counter()
        solve_result = solver.solve(
            post_player_update_callback=archive.capture_from_solver
        )
        training_seconds = time.perf_counter() - training_start
        archive.validate(require_aligned_players=True)

        checkpoints = [int(value) for value in solve_result.diagnostics["iteration"]]
        sd_evaluation_start = time.perf_counter()
        value_target = game_value_player_0(config)
        sd_metrics_by_weighting = {
            weighting: {
                checkpoint: _evaluate_policy(game, policy, value_target)
                for checkpoint, policy in exact_average_policies_at_checkpoints(
                    game,
                    archive,
                    checkpoints,
                    weighting=weighting,
                ).items()
            }
            for weighting in config["sd_cfr_weightings"]
        }
        sd_evaluation_seconds = time.perf_counter() - sd_evaluation_start

        conventional_exploitability = (
            np.asarray(solve_result.nash_conv, dtype=np.float64) / 2.0
        )
        nodes = np.asarray(solve_result.nodes_touched, dtype=np.float64)
        wall_clock = np.asarray(
            solve_result.diagnostics["wall_clock_seconds"], dtype=np.float64
        )
        conventional_values = np.asarray(
            solve_result.average_policy_value, dtype=np.float64
        )
        sd_exploitability = {
            weighting: np.asarray(
                [
                    sd_metrics_by_weighting[weighting][checkpoint]["exploitability"]
                    for checkpoint in checkpoints
                ],
                dtype=np.float64,
            )
            for weighting in config["sd_cfr_weightings"]
        }
        sd_values = {
            weighting: np.asarray(
                [
                    sd_metrics_by_weighting[weighting][checkpoint]["policy_value"]
                    for checkpoint in checkpoints
                ],
                dtype=np.float64,
            )
            for weighting in config["sd_cfr_weightings"]
        }

        archive_path = None
        if bool(config["save_sd_cfr_archives"]):
            archive_path = archive.save(
                run_dir / "sd_cfr_archives" / f"seed_{seed}_sd_cfr_archive.pt"
            )
        policy_snapshot = solver.save_policy_snapshot(
            run_dir / "policy_snapshots" / f"seed_{seed}_deep_cfr_policy.pt",
            seed=seed,
            target_iteration=checkpoints[-1],
            stage_label="Experiment 28 final conventional Deep CFR policy",
            experiment_name=str(config["experiment_name"]),
            game_name=str(config["game_name"]),
            solver_config=dict(config),
        )

        curve_rows = []
        for index, checkpoint in enumerate(checkpoints):
            row = {
                "seed": int(seed),
                "iteration": int(checkpoint),
                "nodes_touched": float(nodes[index]),
                "training_wall_clock_seconds": float(wall_clock[index]),
                "deep_cfr_exploitability": float(conventional_exploitability[index]),
                "deep_cfr_policy_value": float(conventional_values[index]),
                "deep_cfr_policy_value_error": float(
                    abs(conventional_values[index] - value_target)
                ),
            }
            for weighting in config["sd_cfr_weightings"]:
                row.update(
                    {
                        f"sd_cfr_{weighting}_exploitability": float(
                            sd_exploitability[weighting][index]
                        ),
                        f"sd_cfr_{weighting}_minus_deep_cfr_exploitability": float(
                            sd_exploitability[weighting][index]
                            - conventional_exploitability[index]
                        ),
                        f"sd_cfr_{weighting}_policy_value": float(
                            sd_values[weighting][index]
                        ),
                        f"sd_cfr_{weighting}_policy_value_error": float(
                            abs(sd_values[weighting][index] - value_target)
                        ),
                    }
                )
            curve_rows.append(row)

        # The checkpoint arrays are the expensive, irrecoverable part of a
        # conventional Deep CFR trajectory. Save them before AUC/statistical
        # summarisation so a reporting-layer exception cannot discard a run.
        _persist_seed_curves(seed, curve_rows, run_dir)

        summary = {
            "seed": int(seed),
            "num_iterations": int(config["num_iterations"]),
            "num_archived_networks_per_player": int(
                len(archive.entries_by_player[0])
            ),
            "training_seconds": float(training_seconds),
            "sd_cfr_exact_evaluation_seconds": float(sd_evaluation_seconds),
            "final_nodes_touched": float(nodes[-1]),
            "deep_cfr_final_exploitability": float(conventional_exploitability[-1]),
            "deep_cfr_best_exploitability": float(
                np.nanmin(conventional_exploitability)
            ),
            "deep_cfr_exploitability_auc_by_nodes": normalised_auc(
                nodes, conventional_exploitability
            ),
            "deep_cfr_final_policy_value_error": float(
                abs(conventional_values[-1] - value_target)
            ),
            "sd_cfr_archive": str(archive_path) if archive_path else "",
            "deep_cfr_policy_snapshot": str(policy_snapshot),
        }
        deep_auc = float(summary["deep_cfr_exploitability_auc_by_nodes"])
        for weighting in config["sd_cfr_weightings"]:
            sd_auc = normalised_auc(nodes, sd_exploitability[weighting])
            summary.update(
                {
                    f"sd_cfr_{weighting}_final_exploitability": float(
                        sd_exploitability[weighting][-1]
                    ),
                    f"sd_cfr_{weighting}_minus_deep_cfr_final_exploitability": float(
                        sd_exploitability[weighting][-1]
                        - conventional_exploitability[-1]
                    ),
                    f"sd_cfr_{weighting}_best_exploitability": float(
                        np.nanmin(sd_exploitability[weighting])
                    ),
                    f"sd_cfr_{weighting}_exploitability_auc_by_nodes": sd_auc,
                    f"sd_cfr_{weighting}_minus_deep_cfr_exploitability_auc_by_nodes": float(
                        sd_auc - deep_auc
                    ),
                    f"sd_cfr_{weighting}_final_policy_value_error": float(
                        abs(sd_values[weighting][-1] - value_target)
                    ),
                }
            )
        result = {"seed": int(seed), "summary": summary, "curves": curve_rows}
        _persist_seed_result(result, run_dir)
        return result
    finally:
        close = getattr(solver, "close", None)
        if callable(close):
            close()
        cleanup_training_memory()


def _mean_and_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    counts = np.sum(np.isfinite(values), axis=0)
    if values.shape[0] > 1:
        std = np.nanstd(values, axis=0, ddof=1)
        se = np.divide(std, np.sqrt(counts), out=np.zeros_like(std), where=counts > 1)
    else:
        se = np.zeros_like(mean)
    return mean, se


def _plot_results(results: Sequence[dict], config: dict, run_dir: Path) -> None:
    iterations = np.asarray(
        [
            [row["iteration"] for row in result["curves"]]
            for result in results
        ],
        dtype=np.float64,
    )
    nodes = np.asarray(
        [
            [row["nodes_touched"] for row in result["curves"]]
            for result in results
        ],
        dtype=np.float64,
    )
    training_hours = np.asarray(
        [
            [row["training_wall_clock_seconds"] / 3600.0 for row in result["curves"]]
            for result in results
        ],
        dtype=np.float64,
    )
    deep = np.asarray(
        [
            [row["deep_cfr_exploitability"] for row in result["curves"]]
            for result in results
        ],
        dtype=np.float64,
    )
    deep_mean, deep_se = _mean_and_se(deep)
    sd_arrays = {
        weighting: np.asarray(
            [
                [
                    row[f"sd_cfr_{weighting}_exploitability"]
                    for row in result["curves"]
                ]
                for result in results
            ],
            dtype=np.float64,
        )
        for weighting in config["sd_cfr_weightings"]
    }
    sd_colours = {"uniform": "#F58518", "linear": "#54A24B"}
    sd_labels = {
        "uniform": "SD-CFR (uniform, weighting-matched)",
        "linear": "SD-CFR (linear, canonical)",
    }

    for x_by_seed, xlabel, filename in (
        (iterations, "CFR iteration", "exploitability_by_iteration.png"),
        (nodes, "Nodes touched", "exploitability_by_nodes.png"),
        (
            training_hours,
            "Training wall-clock time (hours)",
            "exploitability_by_training_time.png",
        ),
    ):
        mean_x = np.nanmean(x_by_seed, axis=0)
        fig, ax = plt.subplots(figsize=(9.0, 5.6))
        for seed_index, row in enumerate(deep):
            ax.plot(
                x_by_seed[seed_index],
                row,
                color="#4C78A8",
                alpha=0.18,
                linewidth=0.8,
            )
        ax.plot(
            mean_x,
            deep_mean,
            color="#4C78A8",
            linewidth=2.2,
            label="Deep CFR",
        )
        ax.fill_between(
            mean_x,
            deep_mean - deep_se,
            deep_mean + deep_se,
            color="#4C78A8",
            alpha=0.2,
        )
        for weighting, values in sd_arrays.items():
            colour = sd_colours[weighting]
            mean, se = _mean_and_se(values)
            for seed_index, row in enumerate(values):
                ax.plot(
                    x_by_seed[seed_index],
                    row,
                    color=colour,
                    alpha=0.14,
                    linewidth=0.8,
                )
            ax.plot(
                mean_x,
                mean,
                color=colour,
                linewidth=2.2,
                label=sd_labels[weighting],
            )
            ax.fill_between(mean_x, mean - se, mean + se, color=colour, alpha=0.18)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Exploitability (NashConv / 2)")
        set_chart_title(
            ax,
            "Paired output-policy exploitability",
            algorithm_variant="Deep CFR / weighting-matched and canonical SD-CFR",
            game_name="leduc_poker",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=180)
        plt.close(fig)

    final_deep = deep[:, -1]
    final_series = {"deep_cfr": final_deep}
    final_series.update(
        {f"sd_cfr_{key}": values[:, -1] for key, values in sd_arrays.items()}
    )
    labels = ["Deep CFR"] + [sd_labels[key] for key in sd_arrays]
    colours = ["#4C78A8"] + [sd_colours[key] for key in sd_arrays]
    values = list(final_series.values())
    means = [float(np.mean(series)) for series in values]
    ses = [
        float(np.std(series, ddof=1) / np.sqrt(len(series)))
        if len(series) > 1
        else 0.0
        for series in values
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.bar(labels, means, yerr=ses, capsize=5, color=colours)
    for series_index in range(1, len(values)):
        for seed_index in range(len(results)):
            ax.plot(
                [0, series_index],
                [final_deep[seed_index], values[series_index][seed_index]],
                color="black",
                alpha=0.2,
                linewidth=0.7,
                marker="o",
                markersize=2.5,
            )
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(
        ax,
        "Paired final-policy comparison",
        algorithm_variant="Deep CFR / weighting-matched and canonical SD-CFR",
        game_name="leduc_poker",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_paired.png", dpi=180)
    plt.close(fig)


def _export(results: Sequence[dict], failed: Sequence[dict], config: dict, seeds, run_dir):
    summaries = [result["summary"] for result in results]
    curves = [row for result in results for row in result["curves"]]
    write_dict_rows_csv(summaries, run_dir / "seed_summary.csv")
    write_dict_rows_csv(curves, run_dir / "checkpoint_curves.csv")
    aggregate = summarise_numeric_fields(summaries)
    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(aggregate), handle, indent=2)
    paired_payload = {
        "difference_convention": "SD-CFR minus Deep CFR; negative favours SD-CFR",
        "primary_comparison": str(config["primary_sd_cfr_weighting"]),
        "comparisons": {},
    }
    for weighting in config["sd_cfr_weightings"]:
        field = f"sd_cfr_{weighting}_minus_deep_cfr_final_exploitability"
        paired = np.asarray([row[field] for row in summaries], dtype=np.float64)
        paired_payload["comparisons"][weighting] = {
            "n": int(len(paired)),
            "mean": float(np.mean(paired)),
            "std": float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0,
            "se": (
                float(np.std(paired, ddof=1) / np.sqrt(len(paired)))
                if len(paired) > 1
                else 0.0
            ),
            "per_seed": {
                str(row["seed"]): float(row[field]) for row in summaries
            },
        }
    with open(
        run_dir / "paired_difference_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(paired_payload, handle, indent=2)
    write_failed_seeds(run_dir, failed)
    write_experiment_metadata(
        run_dir,
        config=config,
        seeds=seeds,
        completed_seeds=[result["seed"] for result in results],
        extra={
            "comparison_design": (
                "Paired output-policy comparison using identical traversals and "
                "identical fitted advantage networks within each seed"
            ),
            "primary_comparison": (
                "Uniformly weighted SD-CFR versus uniformly weighted fitted Deep "
                "CFR average policy; this isolates output representation"
            ),
            "secondary_comparison": (
                "Canonical linearly iteration-weighted SD-CFR versus the selected "
                "uniformly weighted Deep CFR candidate"
            ),
            "sd_cfr_exact_averages": (
                "Uniform or iteration-weighted, with player-own-reach-weighted "
                "behavioural averaging in both cases"
            ),
            "archive_capture_phase": (
                "Immediately after each player's advantage-network update"
            ),
        },
    )
    _plot_results(results, config, run_dir)


def main() -> int:
    args = _build_parser().parse_args()
    config = build_config(args)
    seeds = _parse_seeds(args.seeds)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(Path(args.output_root), str(config["experiment_name"]))
    configure_run_logging(run_dir, verbose=args.verbose)
    _LOGGER.info("Output directory: %s", run_dir.resolve())
    _LOGGER.info("Configuration: %s", config)
    _LOGGER.info("Seeds: %s", seeds)

    if args.aggregate_workers_root:
        workers_root = Path(args.aggregate_workers_root)
        _LOGGER.info("Aggregating durable worker results from %s", workers_root)
        results = _load_worker_results(workers_root, seeds)
        _export(results, [], config, seeds, run_dir)
        _LOGGER.info("Aggregated %d seed results into %s", len(results), run_dir)
        return 0

    results = []
    failed = []
    for seed in seeds:
        _LOGGER.info("Starting paired Deep CFR / SD-CFR seed %s", seed)
        try:
            result = _run_seed(int(seed), deepcopy(config), run_dir)
            results.append(result)
            _export(results, failed, config, seeds, run_dir)
        except Exception as exc:  # pragma: no cover - cloud/runtime guard
            _LOGGER.exception("Seed %s failed: %s", seed, exc)
            failed.append(
                {
                    "seed": int(seed),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            write_failed_seeds(run_dir, failed)

    if not results:
        _LOGGER.error("All seeds failed; see failed_seeds.json")
        return 1
    _export(results, failed, config, seeds, run_dir)
    _LOGGER.info("Completed %d/%d seeds", len(results), len(seeds))
    _LOGGER.info("Results: %s", run_dir.resolve())
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
