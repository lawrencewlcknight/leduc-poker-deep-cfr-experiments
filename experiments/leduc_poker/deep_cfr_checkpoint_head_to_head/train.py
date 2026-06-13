"""Staged Deep CFR trainer that emits one policy snapshot per schedule milestone.

The trainer iterates over the seeds in the configured ``DEFAULT_SEEDS`` list
(or the user's CLI override). For each seed it walks the
``checkpoint_schedule``, calling :meth:`DeepCFRSolver.solve` for the *delta*
between successive milestones while keeping one in-process solver alive across
the whole schedule. After every stage it writes:

* a checkpoint, ``checkpoints/seed_<seed>_iter_<iter>_full.pt``, containing the
  model/optimizer/training state and, optionally, replay buffers;
* a lightweight policy snapshot, ``snapshots/seed_<seed>_iter_<iter>_snapshot.pt``,
  suitable for cheap analysis.

Per-stage diagnostics are written to ``training_stage_metrics.csv``.
"""

from __future__ import annotations

import gc
import logging
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pyspiel
import torch

from deep_cfr_poker.constants import LEDUC_GAME_VALUE_PLAYER_0
from deep_cfr_poker.experiment_utils import (
    resolve_solver_batch_sizes,
    write_dict_rows_csv,
)
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.snapshots import (
    package_full_checkpoint_filename,
    package_snapshot_filename,
)
from deep_cfr_poker.solver import DeepCFRSolver, SolveResult


_LOGGER = logging.getLogger(__name__)


def _solver_kwargs_from_config(config: Mapping[str, object]) -> dict:
    """Maps a config dict to the kwargs accepted by :class:`DeepCFRSolver`."""
    batch_size_advantage, batch_size_strategy = resolve_solver_batch_sizes(config)
    if isinstance(config, dict):
        config["batch_size_advantage"] = batch_size_advantage
        config["batch_size_strategy"] = batch_size_strategy
    return dict(
        policy_network_layers=tuple(config["policy_network_layers"]),
        advantage_network_layers=tuple(config["advantage_network_layers"]),
        num_traversals=int(config["num_traversals"]),
        learning_rate=float(config["learning_rate"]),
        batch_size_advantage=batch_size_advantage,
        batch_size_strategy=batch_size_strategy,
        memory_capacity=int(config["memory_capacity"]),
        reinitialize_advantage_networks=bool(
            config["reinitialize_advantage_networks"]
        ),
        policy_network_train_steps=int(config["policy_network_train_steps"]),
        advantage_network_train_steps=int(config["advantage_network_train_steps"]),
        compute_exploitability=bool(config["compute_exploitability"]),
    )


def _make_training_stages(
    seed: int,
    schedule: Sequence[int],
    *,
    train_every_by_target: Mapping[int, int],
    checkpoints_dir: Path,
    snapshots_dir: Path,
) -> List[dict]:
    """Returns the list of stage descriptors for one seed."""
    stages: List[dict] = []
    previous = 0
    for target in schedule:
        if target <= previous:
            raise ValueError(
                f"checkpoint_schedule must be strictly increasing; got "
                f"{schedule!r}"
            )
        additional = int(target) - int(previous)
        save_full_path = checkpoints_dir / package_full_checkpoint_filename(seed, target)
        save_snapshot_path = snapshots_dir / package_snapshot_filename(seed, target)
        stages.append(
            {
                "seed": int(seed),
                "previous": int(previous),
                "target": int(target),
                "additional": int(additional),
                "save_full_path": save_full_path,
                "save_snapshot_path": save_snapshot_path,
                "policy_train_every": int(
                    train_every_by_target.get(target, 100)
                ),
                "label": (
                    f"seed {seed}: train to {target} iterations (+{additional})"
                ),
            }
        )
        previous = target
    return stages


def _safe_last(values: Sequence[object]):
    return values[-1] if values else None


def _last_finite(values: Iterable[Optional[float]]) -> float:
    last = float("nan")
    for value in values:
        if value is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            last = v
    return last


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_stage(
    *,
    solver: DeepCFRSolver,
    config: Mapping[str, object],
    stage: Mapping[str, object],
    experiment_name: str,
    solver_config_for_snapshot: Mapping[str, object],
) -> dict:
    """Runs one staged training increment and writes the corresponding files."""
    seed = int(stage["seed"])
    target_iteration = int(stage["target"])
    additional = int(stage["additional"])
    save_full_path: Path = stage["save_full_path"]  # type: ignore[assignment]
    save_snapshot_path: Path = stage["save_snapshot_path"]  # type: ignore[assignment]
    policy_train_every = int(stage["policy_train_every"])
    label = str(stage["label"])
    include_replay_buffers = bool(
        config.get("save_replay_buffers_in_checkpoints", False)
    )

    _LOGGER.info("=" * 80)
    _LOGGER.info(label)
    _LOGGER.info("=" * 80)

    stage_start = time.perf_counter()

    solver._num_iterations = additional
    solver._policy_network_train_every = policy_train_every
    solver._evaluation_interval = policy_train_every

    result: SolveResult = solver.solve()
    stage_wall_clock = time.perf_counter() - stage_start

    solver.save_full_model(
        save_full_path,
        include_buffers=include_replay_buffers,
    )
    _LOGGER.info(
        "Saved checkpoint: %s (include_replay_buffers=%s)",
        save_full_path,
        include_replay_buffers,
    )

    solver.save_policy_snapshot(
        save_snapshot_path,
        seed=seed,
        target_iteration=target_iteration,
        stage_label=label,
        experiment_name=experiment_name,
        game_name=str(config["game_name"]),
        solver_config=dict(solver_config_for_snapshot),
    )
    _LOGGER.info("Saved policy snapshot: %s", save_snapshot_path)

    final_nash_conv = _safe_last(result.nash_conv)
    final_exploitability = (
        float(final_nash_conv) / 2.0 if final_nash_conv is not None else float("nan")
    )
    final_policy_value = _safe_last(result.average_policy_value)
    value_target = float(
        config.get("average_policy_value_target", LEDUC_GAME_VALUE_PLAYER_0)
    )
    final_policy_value_error = (
        abs(float(final_policy_value) - value_target)
        if final_policy_value is not None
        else float("nan")
    )

    row = {
        "seed": seed,
        "target_iteration": target_iteration,
        "additional_iterations": additional,
        "policy_train_every": policy_train_every,
        "full_checkpoint": str(save_full_path),
        "full_checkpoint_includes_replay_buffers": include_replay_buffers,
        "policy_snapshot": str(save_snapshot_path),
        "internal_iteration_after_stage": int(getattr(solver, "_iteration", 0)),
        "nodes_touched_total": int(getattr(solver, "_nodes_touched", 0)),
        "stage_wall_clock_seconds": float(stage_wall_clock),
        "final_nash_conv": float(final_nash_conv) if final_nash_conv is not None else float("nan"),
        "final_exploitability": float(final_exploitability),
        "final_average_policy_value": (
            float(final_policy_value)
            if final_policy_value is not None
            else float("nan")
        ),
        "final_policy_value_error_from_known_value": float(final_policy_value_error),
        "policy_loss_last": _last_finite(result.policy_losses),
        "advantage_loss_player_0_last": _last_finite(
            result.advantage_losses.get(0, [])
        ),
        "advantage_loss_player_1_last": _last_finite(
            result.advantage_losses.get(1, [])
        ),
        "strategy_buffer_size": int(len(solver._strategy_memories)),
        "advantage_buffer_size_player_0": int(len(solver._advantage_memories[0])),
        "advantage_buffer_size_player_1": int(len(solver._advantage_memories[1])),
    }

    del result
    _cleanup_memory()
    return row


def run_training(
    *,
    config: Mapping[str, object],
    seeds: Sequence[int],
    run_dir: Path,
) -> dict:
    """Runs the full staged training loop for every seed.

    Returns a dict containing the per-stage metrics and the list of failed
    seeds (each entry has ``seed``, ``stage_target``, ``error``, ``traceback``).
    """
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    snapshots_dir = run_dir / "snapshots"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    schedule = list(config["checkpoint_schedule"])  # type: ignore[arg-type]
    train_every_by_target = dict(config["policy_train_every_by_target"])  # type: ignore[arg-type]
    experiment_name = str(config["experiment_name"])

    # Snapshot of the solver kwargs to embed in each policy snapshot.
    solver_config_for_snapshot = {
        k: config[k]
        for k in (
            "policy_network_layers",
            "advantage_network_layers",
            "num_traversals",
            "learning_rate",
            "batch_size_advantage",
            "batch_size_strategy",
            "memory_capacity",
            "reinitialize_advantage_networks",
            "policy_network_train_steps",
            "advantage_network_train_steps",
            "compute_exploitability",
        )
    }

    game = pyspiel.load_game(str(config["game_name"]))
    metrics_rows: List[dict] = []
    failed: List[dict] = []
    metrics_path = run_dir / "training_stage_metrics.csv"

    for seed in seeds:
        _LOGGER.info("#" * 80)
        _LOGGER.info("Starting checkpoint-stability run for seed %s", seed)
        _LOGGER.info("#" * 80)
        set_seed(int(seed))
        stages = _make_training_stages(
            int(seed),
            schedule,
            train_every_by_target=train_every_by_target,
            checkpoints_dir=checkpoints_dir,
            snapshots_dir=snapshots_dir,
        )
        solver = DeepCFRSolver(
            game,
            num_iterations=0,
            policy_network_train_every=1,
            **_solver_kwargs_from_config(config),
        )

        for stage in stages:
            try:
                row = _run_stage(
                    solver=solver,
                    config=config,
                    stage=stage,
                    experiment_name=experiment_name,
                    solver_config_for_snapshot=solver_config_for_snapshot,
                )
                metrics_rows.append(row)
                # Persist after every stage so progress is not lost on a
                # later failure.
                write_dict_rows_csv(metrics_rows, metrics_path)
                _LOGGER.info("Updated training_stage_metrics.csv")
            except Exception as exc:  # pragma: no cover -- runtime failures
                _LOGGER.exception(
                    "Seed %s stage targeting %s iterations failed: %s",
                    seed,
                    stage["target"],
                    exc,
                )
                failed.append(
                    {
                        "seed": int(seed),
                        "stage_target": int(stage["target"]),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                # Skip remaining stages for this seed; later stages depend on
                # the solver state accumulated by earlier stages.
                break
        del solver
        _cleanup_memory()

    return {
        "metrics_rows": metrics_rows,
        "failed": failed,
        "metrics_csv": metrics_path,
        "checkpoints_dir": checkpoints_dir,
        "snapshots_dir": snapshots_dir,
    }
