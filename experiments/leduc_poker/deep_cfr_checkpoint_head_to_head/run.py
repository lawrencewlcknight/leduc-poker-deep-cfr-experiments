"""CLI entry point for the Leduc poker Deep CFR checkpoint head-to-head experiment.

The experiment has two phases. By default ``run.py`` invokes both, but each is
also independently invokable so the analysis can be re-run without retraining:

* ``python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run train``
  trains and writes snapshots.
* ``python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run analyse --run-dir <existing>``
  re-runs analysis against an existing run directory's snapshots.
* ``python -m experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.run all``
  (the default) runs both phases in one process.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

# Keep execution CPU-only by default; the user can set CUDA_VISIBLE_DEVICES
# explicitly before launching to opt into GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    configure_run_logging,
    create_run_dir,
    write_experiment_metadata,
    write_failed_seeds,
)

from .analyse import run_analysis  # noqa: E402
from .config import DEFAULT_CONFIG, DEFAULT_SEEDS  # noqa: E402
from .train import run_training  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.checkpoint_head_to_head")


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_schedule(value: Optional[str]):
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
    """Builds an experiment config dict from defaults and CLI overrides."""
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "experiment_name": args.experiment_name,
        "checkpoint_schedule": parse_schedule(args.checkpoint_schedule),
        "policy_network_layers": parse_int_tuple(args.policy_network_layers),
        "advantage_network_layers": parse_int_tuple(args.advantage_network_layers),
        "num_traversals": args.num_traversals,
        "learning_rate": args.learning_rate,
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "memory_capacity": args.memory_capacity,
        "reinitialize_advantage_networks": args.reinitialize_advantage_networks,
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "compute_exploitability": args.compute_exploitability,
        "equivalence_epsilon": args.equivalence_epsilon,
        "run_monte_carlo_validation": args.run_monte_carlo_validation,
        "num_mc_episodes": args.num_mc_episodes,
        "mc_pair_mode": args.mc_pair_mode,
        "annotate_heatmap": args.annotate_heatmap,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.policy_train_every is not None:
        # Override the same value for every milestone in the schedule.
        config["policy_train_every_by_target"] = {
            int(it): int(args.policy_train_every)
            for it in config["checkpoint_schedule"]
        }
    return config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Leduc poker Deep CFR checkpoint head-to-head experiment "
            "(train, analyse, or both)."
        )
    )
    parser.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=["all", "train", "analyse"],
        help="Which phase to run. Default: all.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Root directory for outputs (a timestamped run-dir is created here).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Use an existing run directory instead of creating a new one. "
            "Required when phase=analyse on an existing training run."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list. Defaults to three fixed seeds.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--checkpoint-schedule",
        default=None,
        help="Comma-separated milestones, e.g. '100,300,500,750,1000,1250,1500'.",
    )
    parser.add_argument("--num-traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument(
        "--reinitialize-advantage-networks",
        type=_str2bool,
        default=None,
        help="Reset advantage networks each iteration (Brown-style).",
    )
    parser.add_argument(
        "--compute-exploitability",
        type=_str2bool,
        default=None,
    )
    parser.add_argument(
        "--policy-train-every",
        type=int,
        default=None,
        help=(
            "Override policy_network_train_every for every milestone in the "
            "schedule. By default the per-target dict in config.py is used."
        ),
    )
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--advantage-network-layers", default=None)
    parser.add_argument("--equivalence-epsilon", type=float, default=None)
    parser.add_argument(
        "--run-monte-carlo-validation", type=_str2bool, default=None
    )
    parser.add_argument("--num-mc-episodes", type=int, default=None)
    parser.add_argument(
        "--mc-pair-mode",
        choices=["adjacent", "milestones", "all_pairs"],
        default=None,
    )
    parser.add_argument("--annotate-heatmap", type=_str2bool, default=None)
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
    _LOGGER.info("Phase: %s", args.phase)
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Configuration: %s", config)
    _LOGGER.info("Seeds: %s", seeds)

    train_outcome = None
    if args.phase in {"all", "train"}:
        train_outcome = run_training(config=config, seeds=seeds, run_dir=run_dir)
        completed_seeds = sorted(
            {row["seed"] for row in train_outcome["metrics_rows"]}
        )
        if not train_outcome["metrics_rows"]:
            _LOGGER.error("Training produced no metrics. Aborting.")
            return 1
        write_experiment_metadata(
            run_dir,
            config=config,
            seeds=seeds,
            completed_seeds=completed_seeds,
            extra={
                "phase": args.phase,
                "training_stage_metrics_csv": str(train_outcome["metrics_csv"]),
            },
        )
        if train_outcome["failed"]:
            write_failed_seeds(run_dir, train_outcome["failed"])
            _LOGGER.warning(
                "%d stage(s) failed; see failed_seeds.json", len(train_outcome["failed"])
            )

    if args.phase in {"all", "analyse"}:
        snapshots_dir = run_dir / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.glob("*.pt")):
            _LOGGER.error(
                "Analysis phase requires snapshots in %s. Run the training phase first.",
                snapshots_dir,
            )
            return 2
        analysis_outputs = run_analysis(
            config=config, run_dir=run_dir, snapshots_dir=snapshots_dir
        )
        # If we ran analyse-only, write metadata too so the run directory is
        # self-describing.
        if args.phase == "analyse":
            write_experiment_metadata(
                run_dir,
                config=config,
                seeds=seeds,
                completed_seeds=None,
                extra={"phase": "analyse_only"},
            )
        _LOGGER.info("Analysis outputs: %s", analysis_outputs)

    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
