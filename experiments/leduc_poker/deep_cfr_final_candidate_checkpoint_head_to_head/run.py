"""CLI for final-candidate checkpoint head-to-head validation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/deep_cfr_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/deep_cfr_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    configure_run_logging,
    create_run_dir,
    write_experiment_metadata,
    write_failed_seeds,
)

from .analyse import run_analysis  # noqa: E402
from .config import DEFAULT_CONFIG, DEFAULT_SEEDS  # noqa: E402
from .train import run_training, validate_config  # noqa: E402


_LOGGER = logging.getLogger(
    "deep_cfr_poker.experiment.final_candidate_checkpoint_head_to_head"
)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_seeds(value: Optional[str]) -> List[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_config(args, *, base_config=None) -> dict:
    config = deepcopy(DEFAULT_CONFIG if base_config is None else base_config)
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "checkpoint_schedule": _parse_int_tuple(args.checkpoint_schedule),
        "evaluation_interval": args.evaluation_interval,
        "policy_network_train_every": args.policy_network_train_every,
        "learning_rate": args.learning_rate,
        "policy_network_layers": _parse_int_tuple(args.policy_network_layers),
        "advantage_network_layers": _parse_int_tuple(args.advantage_network_layers),
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "memory_capacity": args.memory_capacity,
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "compute_exploitability": args.compute_exploitability,
        "equivalence_epsilon": args.equivalence_epsilon,
        "annotate_heatmap": args.annotate_heatmap,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    validate_config(config)
    return config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the best Leduc Deep CFR configuration once per seed, save "
            "five intermediate policies, and compare them by exact head-to-head EV."
        )
    )
    parser.add_argument(
        "phase", nargs="?", default="all", choices=("all", "train", "analyse")
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--checkpoint-schedule", default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--advantage-network-layers", default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--equivalence-epsilon", type=float, default=None)
    parser.add_argument("--annotate-heatmap", type=_str2bool, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    stored_metadata = None
    stored_config = None
    if args.phase == "analyse" and args.run_dir:
        metadata_path = Path(args.run_dir) / "experiment_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as handle:
                stored_metadata = json.load(handle)
            stored_config = stored_metadata.get("experiment_config")
    config = build_config(args, base_config=stored_config)
    if args.seeds is None and stored_metadata and stored_metadata.get("seeds"):
        seeds = [int(seed) for seed in stored_metadata["seeds"]]
    else:
        seeds = _parse_seeds(args.seeds)
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(Path(args.output_root), str(config["experiment_name"]))

    configure_run_logging(run_dir, verbose=args.verbose)
    _LOGGER.info("Phase: %s", args.phase)
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Seeds: %s", seeds)
    _LOGGER.info("Configuration: %s", config)

    completed_seeds = None
    if args.phase in {"all", "train"}:
        outcome = run_training(config=config, seeds=seeds, run_dir=run_dir)
        required_count = len(config["checkpoint_schedule"])
        counts = {
            seed: sum(1 for row in outcome["metrics_rows"] if int(row["seed"]) == seed)
            for seed in seeds
        }
        completed_seeds = [seed for seed in seeds if counts[seed] == required_count]
        write_experiment_metadata(
            run_dir,
            config=config,
            seeds=seeds,
            completed_seeds=completed_seeds,
            extra={
                "phase": args.phase,
                "training_stage_metrics_csv": str(outcome["metrics_csv"]),
                "head_to_head_evaluation": "exact, seat-averaged OpenSpiel expected value",
                "statistical_unit": "independent training seed",
            },
        )
        write_failed_seeds(run_dir, outcome["failed"])
        if not completed_seeds:
            _LOGGER.error("No seed completed the full checkpoint schedule.")
            return 1

    if args.phase in {"all", "analyse"}:
        snapshots_dir = run_dir / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.glob("*.pt")):
            _LOGGER.error("No snapshots found in %s", snapshots_dir)
            return 2
        outputs = run_analysis(
            config=config,
            run_dir=run_dir,
            snapshots_dir=snapshots_dir,
        )
        if args.phase == "analyse" and stored_metadata is None:
            write_experiment_metadata(
                run_dir,
                config=config,
                seeds=seeds,
                completed_seeds=None,
                extra={
                    "phase": "analyse_only",
                    "head_to_head_evaluation": (
                        "exact, seat-averaged OpenSpiel expected value"
                    ),
                    "statistical_unit": "independent training seed",
                },
            )
        _LOGGER.info("Analysis outputs: %s", outputs)

    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
