"""Straight-through training with lightweight intermediate policy snapshots."""

from __future__ import annotations

import gc
import logging
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List, Mapping, Sequence

import pyspiel
import torch

from deep_cfr_poker.experiment_utils import make_solver, write_dict_rows_csv
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.snapshots import package_snapshot_filename


_LOGGER = logging.getLogger(__name__)

_SNAPSHOT_CONFIG_KEYS = (
    "policy_network_type",
    "policy_network_layers",
    "advantage_network_type",
    "advantage_network_layers",
    "num_iterations",
    "num_traversals",
    "learning_rate",
    "learning_rate_schedule",
    "batch_size_advantage",
    "batch_size_strategy",
    "memory_capacity",
    "reinitialize_advantage_networks",
    "policy_network_train_steps",
    "advantage_network_train_steps",
    "policy_network_train_every",
    "evaluation_interval",
    "target_processing",
    "target_standardize_epsilon",
    "advantage_replay_sampling",
    "average_strategy_weighting",
)


def validate_config(config: Mapping[str, object]) -> None:
    """Checks that every requested snapshot represents a freshly fitted policy."""
    schedule = tuple(int(value) for value in config["checkpoint_schedule"])
    num_iterations = int(config["num_iterations"])
    train_every = int(config["policy_network_train_every"])
    if not schedule or any(a >= b for a, b in zip(schedule, schedule[1:])):
        raise ValueError("checkpoint_schedule must be non-empty and strictly increasing")
    if schedule[-1] != num_iterations:
        raise ValueError(
            "The final checkpoint must equal num_iterations so all runs finish "
            "at the configured training budget"
        )
    stale = [value for value in schedule[:-1] if value % train_every != 0]
    if stale:
        raise ValueError(
            "Every intermediate checkpoint must coincide with average-policy "
            f"training; incompatible checkpoints: {stale}"
        )


def _cleanup_solver(solver) -> None:
    if solver is not None:
        close = getattr(solver, "close", None)
        if callable(close):
            close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_seed(
    *,
    seed: int,
    config: Mapping[str, object],
    run_dir: Path,
    existing_rows: List[dict],
    metrics_path: Path,
) -> List[dict]:
    seed_config = deepcopy(dict(config))
    set_seed(seed)
    game = pyspiel.load_game(str(seed_config["game_name"]))
    solver = make_solver(game, seed_config)
    schedule = {int(value) for value in seed_config["checkpoint_schedule"]}
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_config = {
        key: seed_config[key] for key in _SNAPSHOT_CONFIG_KEYS if key in seed_config
    }
    training_start = time.perf_counter()
    seed_rows: List[dict] = []

    def save_requested_snapshot(active_solver, completed_iteration: int) -> None:
        if completed_iteration not in schedule:
            return
        path = snapshots_dir / package_snapshot_filename(seed, completed_iteration)
        active_solver.save_policy_snapshot(
            path,
            seed=seed,
            target_iteration=completed_iteration,
            stage_label=f"straight-through checkpoint {completed_iteration}",
            experiment_name=str(seed_config["experiment_name"]),
            game_name=str(seed_config["game_name"]),
            solver_config=snapshot_config,
        )
        row = {
            "seed": int(seed),
            "checkpoint_iteration": int(completed_iteration),
            "checkpoint_fraction": float(
                completed_iteration / int(seed_config["num_iterations"])
            ),
            "nodes_touched": int(active_solver._nodes_touched),
            "wall_clock_seconds": float(time.perf_counter() - training_start),
            "policy_training_events": int(active_solver._policy_training_events),
            "policy_gradient_steps": int(active_solver._policy_gradient_steps),
            "strategy_buffer_size": int(len(active_solver._strategy_memories)),
            "advantage_buffer_size_player_0": int(
                len(active_solver._advantage_memories[0])
            ),
            "advantage_buffer_size_player_1": int(
                len(active_solver._advantage_memories[1])
            ),
            "policy_snapshot": str(path),
        }
        seed_rows.append(row)
        write_dict_rows_csv([*existing_rows, *seed_rows], metrics_path)
        _LOGGER.info(
            "Saved seed %s checkpoint %s at %s nodes",
            seed,
            completed_iteration,
            row["nodes_touched"],
        )

    try:
        solver.solve(post_iteration_callback=save_requested_snapshot)
        captured = {int(row["checkpoint_iteration"]) for row in seed_rows}
        missing = sorted(schedule - captured)
        if missing:
            raise RuntimeError(f"Training completed without snapshots at {missing}")
        return seed_rows
    finally:
        _cleanup_solver(solver)


def run_training(
    *,
    config: Mapping[str, object],
    seeds: Sequence[int],
    run_dir: Path,
) -> dict:
    """Trains one uninterrupted final-candidate trajectory per seed."""
    validate_config(config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "training_stage_metrics.csv"
    metrics_rows: List[dict] = []
    failed: List[dict] = []

    for seed_value in seeds:
        seed = int(seed_value)
        _LOGGER.info("Starting straight-through checkpoint run for seed %s", seed)
        try:
            metrics_rows.extend(
                _run_seed(
                    seed=seed,
                    config=config,
                    run_dir=run_dir,
                    existing_rows=metrics_rows,
                    metrics_path=metrics_path,
                )
            )
            write_dict_rows_csv(metrics_rows, metrics_path)
        except Exception as exc:  # pragma: no cover - exercised by cloud failures
            _LOGGER.exception("Seed %s failed: %s", seed, exc)
            failed.append(
                {
                    "seed": seed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    return {
        "metrics_rows": metrics_rows,
        "failed": failed,
        "metrics_csv": metrics_path,
        "snapshots_dir": run_dir / "snapshots",
    }


__all__ = ["run_training", "validate_config"]
