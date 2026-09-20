"""Training, deferred exact evaluation, and analysis for Experiment 29."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/deep_cfr_exp29_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/deep_cfr_exp29_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python import policy as osp_policy  # noqa: E402
from open_spiel.python.algorithms import expected_game_score, exploitability  # noqa: E402
from scipy import stats  # noqa: E402

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    cleanup_training_memory,
    game_value_player_0,
    json_safe,
    make_solver,
    normalised_auc,
    write_dict_rows_csv,
)
from deep_cfr_poker.plotting import set_chart_title  # noqa: E402
from deep_cfr_poker.sd_cfr import (  # noqa: E402
    SDCFRArchive,
    exact_average_policies_at_checkpoints,
)
from deep_cfr_poker.seeding import set_seed  # noqa: E402

from .config import (  # noqa: E402
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    PRIMARY_WEIGHTING,
    PRODUCTION_SEEDS,
    SECONDARY_WEIGHTING,
    build_config,
)


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.uniform_sd_cfr_36h")


def write_json(path: Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    temporary.replace(path)
    return path


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate_policy(game, candidate, value_target: float) -> dict:
    tabular = osp_policy.tabular_policy_from_callable(
        game, candidate.action_probabilities
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


def _checkpoint_row(
    *,
    checkpoint_id: str,
    checkpoint_kind: str,
    seed: int,
    iteration: int,
    nodes_touched: int,
    active_seconds: float,
    time_checkpoint_index: int | None = None,
    scheduled_seconds: float | None = None,
    is_node_15m_endpoint: bool = False,
    is_final_endpoint: bool = False,
) -> dict:
    scheduled_hours = (
        float(scheduled_seconds) / 3600.0
        if scheduled_seconds is not None
        else float("nan")
    )
    return {
        "checkpoint_id": str(checkpoint_id),
        "checkpoint_kind": str(checkpoint_kind),
        "seed": int(seed),
        "iteration": int(iteration),
        "nodes_touched": int(nodes_touched),
        "active_training_seconds": float(active_seconds),
        "training_hours": float(active_seconds) / 3600.0,
        "time_checkpoint_index": (
            "" if time_checkpoint_index is None else int(time_checkpoint_index)
        ),
        "scheduled_active_seconds": (
            "" if scheduled_seconds is None else float(scheduled_seconds)
        ),
        "scheduled_training_hours": scheduled_hours,
        "is_12h_endpoint": bool(
            scheduled_seconds is not None
            and math.isclose(float(scheduled_seconds), 12.0 * 3600.0)
        ),
        "is_24h_endpoint": bool(
            scheduled_seconds is not None
            and math.isclose(float(scheduled_seconds), 24.0 * 3600.0)
        ),
        "is_36h_endpoint": bool(
            scheduled_seconds is not None
            and math.isclose(float(scheduled_seconds), 36.0 * 3600.0)
        ),
        "is_node_15m_endpoint": bool(is_node_15m_endpoint),
        "is_final_endpoint": bool(is_final_endpoint),
    }


def run_training_seed(*, seed: int, output_dir: Path, smoke: bool = False) -> dict:
    """Train one standalone SD-CFR seed and retain its complete model archive."""
    config = build_config(smoke=smoke)
    seed = int(seed)
    if seed not in tuple(map(int, config["seeds"])):
        raise ValueError(f"Seed {seed} is outside the Experiment 29 schedule")
    output_dir = Path(output_dir)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    game = pyspiel.load_game(str(config["game_name"]))
    solver = make_solver(game, config)
    archive = SDCFRArchive.from_solver(
        solver,
        game_name=str(config["game_name"]),
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "seed": seed,
            "primary_weighting": PRIMARY_WEIGHTING,
            "capture_phase": "immediately_after_each_player_advantage_update",
            "policy_training_mode": "disabled",
            "collect_strategy_replay": False,
        },
    )
    rows: list[dict] = []
    interval = float(config["checkpoint_interval_seconds"])
    requested_seconds = float(config["training_time_budget_seconds"])
    target_nodes = int(config["target_nodes_touched"])
    next_scheduled = interval
    next_time_index = 1
    node_endpoint_recorded = False
    latest_progress: dict | None = None
    started = time.perf_counter()

    def record_progress(active_solver, completed_iteration: int) -> None:
        nonlocal next_scheduled, next_time_index, node_endpoint_recorded, latest_progress
        active_seconds = time.perf_counter() - started
        latest_progress = {
            "iteration": int(completed_iteration),
            "nodes_touched": int(active_solver._nodes_touched),
            "active_training_seconds": float(active_seconds),
        }
        while active_seconds >= next_scheduled and next_scheduled <= requested_seconds:
            rows.append(
                _checkpoint_row(
                    checkpoint_id=f"time_{next_time_index:03d}",
                    checkpoint_kind="scheduled_time",
                    seed=seed,
                    iteration=completed_iteration,
                    nodes_touched=active_solver._nodes_touched,
                    active_seconds=active_seconds,
                    time_checkpoint_index=next_time_index,
                    scheduled_seconds=next_scheduled,
                )
            )
            _LOGGER.info(
                "Seed %s crossed scheduled checkpoint %.2fh at iteration %s, %s nodes",
                seed,
                next_scheduled / 3600.0,
                completed_iteration,
                active_solver._nodes_touched,
            )
            next_time_index += 1
            next_scheduled += interval
        if not node_endpoint_recorded and int(active_solver._nodes_touched) >= target_nodes:
            rows.append(
                _checkpoint_row(
                    checkpoint_id="node_15m",
                    checkpoint_kind="node_endpoint",
                    seed=seed,
                    iteration=completed_iteration,
                    nodes_touched=active_solver._nodes_touched,
                    active_seconds=active_seconds,
                    is_node_15m_endpoint=True,
                )
            )
            node_endpoint_recorded = True

    try:
        solver.solve(
            post_iteration_callback=record_progress,
            post_player_update_callback=archive.capture_from_solver,
            max_training_seconds=requested_seconds,
        )
        archive.validate(require_aligned_players=True)
        if latest_progress is None:
            raise RuntimeError("The timed learner completed no full CFR iteration")
        final_iteration = int(latest_progress["iteration"])
        final_matches = [row for row in rows if int(row["iteration"]) == final_iteration]
        if final_matches:
            final_matches[-1]["is_final_endpoint"] = True
        else:
            rows.append(
                _checkpoint_row(
                    checkpoint_id="final",
                    checkpoint_kind="final",
                    seed=seed,
                    iteration=final_iteration,
                    nodes_touched=int(latest_progress["nodes_touched"]),
                    active_seconds=float(latest_progress["active_training_seconds"]),
                    is_final_endpoint=True,
                )
            )
        if not smoke and float(latest_progress["active_training_seconds"]) < requested_seconds:
            raise RuntimeError("Training stopped before the 36-hour active-time boundary")

        manifest_path = seed_dir / "checkpoint_manifest.csv"
        write_dict_rows_csv(rows, manifest_path)
        archive_path = archive.save(seed_dir / f"seed_{seed}_sd_cfr_archive.pt")
        result = {
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "seed": seed,
            "smoke": bool(smoke),
            "active_training_seconds": float(latest_progress["active_training_seconds"]),
            "requested_training_seconds": requested_seconds,
            "budget_overshoot_seconds": (
                float(latest_progress["active_training_seconds"]) - requested_seconds
            ),
            "final_iteration": final_iteration,
            "final_nodes_touched": int(latest_progress["nodes_touched"]),
            "num_archived_networks_per_player": int(
                len(archive.entries_by_player[0])
            ),
            "num_checkpoints": len(rows),
            "num_time_checkpoints": sum(
                str(row["time_checkpoint_index"]) not in {"", "None"} for row in rows
            ),
            "hit_node_15m": bool(node_endpoint_recorded),
            "archive_path": str(archive_path.relative_to(output_dir)),
            "archive_sha256": sha256(archive_path),
            "manifest_path": str(manifest_path.relative_to(output_dir)),
            "configuration": config,
        }
        write_json(seed_dir / "training_result.json", result)
        write_json(
            seed_dir / "TRAINING_SUCCESS.json",
            {"status": "complete", "seed": seed, "experiment_id": EXPERIMENT_ID},
        )
        return result
    finally:
        close = getattr(solver, "close", None)
        if callable(close):
            close()
        cleanup_training_memory()


def run_evaluation_seed(
    *, seed: int, training_dir: Path, output_dir: Path, smoke: bool = False
) -> dict:
    """Evaluate uniform and linear exact SD-CFR policies for every prefix."""
    config = build_config(smoke=smoke)
    seed = int(seed)
    training_dir = Path(training_dir)
    output_dir = Path(output_dir)
    training_result = read_json(training_dir / f"seed_{seed}" / "training_result.json")
    if int(training_result["experiment_id"]) != EXPERIMENT_ID:
        raise ValueError("Training result belongs to a different experiment")
    archive_path = training_dir / str(training_result["archive_path"])
    if sha256(archive_path) != str(training_result["archive_sha256"]):
        raise ValueError("SD-CFR archive checksum mismatch")
    archive = SDCFRArchive.load(archive_path)
    manifest = read_csv(training_dir / str(training_result["manifest_path"]))
    if not manifest:
        raise RuntimeError("Experiment 29 checkpoint manifest is empty")
    checkpoints = sorted({int(row["iteration"]) for row in manifest})
    game = pyspiel.load_game(str(config["game_name"]))
    value_target = game_value_player_0(config)
    started = time.perf_counter()
    policies = {
        weighting: exact_average_policies_at_checkpoints(
            game, archive, checkpoints, weighting=weighting
        )
        for weighting in config["sd_cfr_weightings"]
    }
    evaluated = {
        weighting: {
            checkpoint: _evaluate_policy(game, candidate, value_target)
            for checkpoint, candidate in by_checkpoint.items()
        }
        for weighting, by_checkpoint in policies.items()
    }
    rows = []
    for source in manifest:
        iteration = int(source["iteration"])
        row = dict(source)
        row["seed"] = seed
        for weighting in config["sd_cfr_weightings"]:
            metrics = evaluated[weighting][iteration]
            for key, value in metrics.items():
                row[f"sd_cfr_{weighting}_{key}"] = float(value)
        row["uniform_minus_linear_exploitability"] = float(
            row["sd_cfr_uniform_exploitability"]
            - row["sd_cfr_linear_exploitability"]
        )
        rows.append(row)

    evaluation_seed_dir = output_dir / f"seed_{seed}"
    evaluation_seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = evaluation_seed_dir / "checkpoint_metrics.csv"
    write_dict_rows_csv(rows, metrics_path)
    result = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": seed,
        "smoke": bool(smoke),
        "evaluation_seconds": time.perf_counter() - started,
        "num_metric_rows": len(rows),
        "num_unique_iterations": len(checkpoints),
        "metrics_path": str(metrics_path.relative_to(output_dir)),
        "source_archive_sha256": str(training_result["archive_sha256"]),
    }
    write_json(evaluation_seed_dir / "evaluation_result.json", result)
    write_json(
        evaluation_seed_dir / "EVALUATION_SUCCESS.json",
        {"status": "complete", "seed": seed, "experiment_id": EXPERIMENT_ID},
    )
    return result


def _truth(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _float(value) -> float:
    if value in (None, "", "None"):
        return float("nan")
    return float(value)


def _stats(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    se = std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    half = float(stats.t.ppf(0.975, len(array) - 1) * se) if len(array) > 1 else 0.0
    return {
        "n": int(len(array)),
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _summarise_groups(rows: Sequence[dict], group_field: str) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        raw = row.get(group_field)
        if raw not in (None, "", "None"):
            groups[int(float(raw))].append(row)
    output = []
    numeric_fields = (
        "scheduled_active_seconds",
        "active_training_seconds",
        "training_hours",
        "iteration",
        "nodes_touched",
        "sd_cfr_uniform_exploitability",
        "sd_cfr_linear_exploitability",
        "uniform_minus_linear_exploitability",
        "sd_cfr_uniform_policy_value",
        "sd_cfr_linear_policy_value",
    )
    for group, selected in sorted(groups.items()):
        summary = {group_field: group, "n_seeds": len({int(r["seed"]) for r in selected})}
        for field in numeric_fields:
            field_stats = _stats(_float(row.get(field)) for row in selected)
            for key, value in field_stats.items():
                summary[f"{field}_{key}"] = value
        output.append(summary)
    return output


def _endpoint_labels(row: Mapping[str, object]) -> list[str]:
    labels = []
    if _truth(row.get("is_12h_endpoint")):
        labels.append("time_12h")
    if _truth(row.get("is_24h_endpoint")):
        labels.append("time_24h")
    if _truth(row.get("is_36h_endpoint")):
        labels.append("time_36h")
    if _truth(row.get("is_node_15m_endpoint")):
        labels.append("node_15m")
    if _truth(row.get("is_final_endpoint")):
        labels.append("final")
    return labels


def _plot_trajectory(
    rows: Sequence[dict], *, x_field: str, xlabel: str, output: Path
) -> None:
    time_rows = [
        row for row in rows if row.get("time_checkpoint_index") not in (None, "", "None")
    ]
    colours = {"uniform": "#F58518", "linear": "#54A24B"}
    labels = {"uniform": "SD-CFR (uniform, selected)", "linear": "SD-CFR (linear)"}
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in time_rows:
        by_seed[int(row["seed"])].append(row)
    for weighting in (PRIMARY_WEIGHTING, SECONDARY_WEIGHTING):
        metric = f"sd_cfr_{weighting}_exploitability"
        for seed_rows in by_seed.values():
            seed_rows.sort(key=lambda item: int(float(item["time_checkpoint_index"])))
            ax.plot(
                [_float(item[x_field]) for item in seed_rows],
                [_float(item[metric]) for item in seed_rows],
                color=colours[weighting],
                alpha=0.16,
                linewidth=0.8,
            )
        groups: dict[int, list[dict]] = defaultdict(list)
        for row in time_rows:
            groups[int(float(row["time_checkpoint_index"]))].append(row)
        x_values, means, lows, highs = [], [], [], []
        for index in sorted(groups):
            selected = groups[index]
            x_values.append(float(np.mean([_float(item[x_field]) for item in selected])))
            summary = _stats(_float(item[metric]) for item in selected)
            means.append(summary["mean"])
            lows.append(summary["ci95_low"])
            highs.append(summary["ci95_high"])
        ax.plot(x_values, means, color=colours[weighting], linewidth=2.2, label=labels[weighting])
        ax.fill_between(x_values, lows, highs, color=colours[weighting], alpha=0.18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(
        ax,
        "Long-horizon SD-CFR exploitability",
        algorithm_variant="Experiment 28 selected uniform candidate",
        game_name="leduc_poker",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_difference(rows: Sequence[dict], output: Path) -> None:
    time_rows = [
        row for row in rows if row.get("time_checkpoint_index") not in (None, "", "None")
    ]
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in time_rows:
        groups[int(float(row["time_checkpoint_index"]))].append(row)
    x_values, means, lows, highs = [], [], [], []
    for index in sorted(groups):
        selected = groups[index]
        x_values.append(float(np.mean([_float(item["scheduled_training_hours"]) for item in selected])))
        summary = _stats(_float(item["uniform_minus_linear_exploitability"]) for item in selected)
        means.append(summary["mean"])
        lows.append(summary["ci95_low"])
        highs.append(summary["ci95_high"])
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.plot(x_values, means, color="#7A5195", linewidth=2.2)
    ax.fill_between(x_values, lows, highs, color="#7A5195", alpha=0.2)
    ax.set_xlabel("Active training time (hours)")
    ax.set_ylabel("Uniform minus linear exploitability")
    set_chart_title(
        ax,
        "SD-CFR averaging-rule difference",
        algorithm_variant="Negative values favour uniform weighting",
        game_name="leduc_poker",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def aggregate_results(
    *, metric_rows: Sequence[dict], output_dir: Path, expected_seed_count: int
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted({int(row["seed"]) for row in metric_rows})
    if len(seeds) != int(expected_seed_count):
        raise ValueError(f"Expected {expected_seed_count} seeds, found {seeds}")
    write_dict_rows_csv(metric_rows, output_dir / "checkpoint_seed_metrics.csv")
    time_summary = _summarise_groups(metric_rows, "time_checkpoint_index")
    write_dict_rows_csv(time_summary, output_dir / "trajectory_by_training_time_summary.csv")
    # The common temporal checkpoint index aligns seeds; mean nodes provides
    # the corresponding x-coordinate for the aggregate node trajectory.
    write_dict_rows_csv(time_summary, output_dir / "trajectory_by_nodes_summary.csv")

    endpoint_rows = []
    for row in metric_rows:
        for label in _endpoint_labels(row):
            endpoint_rows.append({"endpoint_id": label, **row})
    write_dict_rows_csv(endpoint_rows, output_dir / "endpoint_seed_metrics.csv")
    endpoint_summary = []
    for endpoint in ("time_12h", "time_24h", "time_36h", "node_15m", "final"):
        selected = [row for row in endpoint_rows if row["endpoint_id"] == endpoint]
        if not selected:
            continue
        summary = {"endpoint_id": endpoint, "n_seeds": len({int(r["seed"]) for r in selected})}
        for field in (
            "active_training_seconds",
            "iteration",
            "nodes_touched",
            "sd_cfr_uniform_exploitability",
            "sd_cfr_linear_exploitability",
            "uniform_minus_linear_exploitability",
            "sd_cfr_uniform_policy_value",
            "sd_cfr_linear_policy_value",
        ):
            for key, value in _stats(_float(row[field]) for row in selected).items():
                summary[f"{field}_{key}"] = value
        endpoint_summary.append(summary)
    write_dict_rows_csv(endpoint_summary, output_dir / "endpoint_aggregate_summary.csv")

    seed_summaries = []
    for seed in seeds:
        selected = [
            row
            for row in metric_rows
            if int(row["seed"]) == seed
            and row.get("time_checkpoint_index") not in (None, "", "None")
        ]
        selected.sort(key=lambda row: int(float(row["time_checkpoint_index"])))
        hours = [_float(row["scheduled_training_hours"]) for row in selected]
        uniform = [_float(row["sd_cfr_uniform_exploitability"]) for row in selected]
        linear = [_float(row["sd_cfr_linear_exploitability"]) for row in selected]
        final_window = uniform[-min(12, len(uniform)) :]
        seed_summaries.append(
            {
                "seed": seed,
                "final_uniform_exploitability": uniform[-1],
                "best_uniform_exploitability": float(np.min(uniform)),
                "final_minus_best_uniform_exploitability": uniform[-1] - float(np.min(uniform)),
                "final_window_uniform_std": (
                    float(np.std(final_window, ddof=1)) if len(final_window) > 1 else 0.0
                ),
                "uniform_auc_by_training_time": normalised_auc(hours, uniform),
                "linear_auc_by_training_time": normalised_auc(hours, linear),
                "final_uniform_minus_linear": uniform[-1] - linear[-1],
            }
        )
    write_dict_rows_csv(seed_summaries, output_dir / "seed_summary.csv")

    _plot_trajectory(
        metric_rows,
        x_field="scheduled_training_hours",
        xlabel="Active training time (hours)",
        output=output_dir / "exploitability_by_training_time.png",
    )
    _plot_trajectory(
        metric_rows,
        x_field="nodes_touched",
        xlabel="Nodes touched",
        output=output_dir / "exploitability_by_nodes.png",
    )
    _plot_trajectory(
        metric_rows,
        x_field="iteration",
        xlabel="Completed CFR iteration",
        output=output_dir / "exploitability_by_iteration.png",
    )
    _plot_difference(metric_rows, output_dir / "uniform_minus_linear_by_training_time.png")
    result = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "num_checkpoint_rows": len(metric_rows),
        "primary_endpoint": "time_36h exact uniform SD-CFR exploitability",
        "primary_weighting": PRIMARY_WEIGHTING,
        "secondary_weighting": SECONDARY_WEIGHTING,
        "seed_summary": seed_summaries,
        "endpoint_summary": endpoint_summary,
    }
    write_json(output_dir / "aggregate_summary.json", result)
    return result


__all__ = [
    "aggregate_results",
    "read_csv",
    "read_json",
    "run_evaluation_seed",
    "run_training_seed",
    "sha256",
    "write_json",
]
