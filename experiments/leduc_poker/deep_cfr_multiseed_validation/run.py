"""CLI entry point for the Leduc poker Deep CFR multi-seed validation experiment."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

# Keep execution CPU-only for reproducibility unless the user explicitly opts in
# by setting CUDA_VISIBLE_DEVICES before launching this script.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from tqdm import tqdm  # noqa: E402

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    DEFAULT_FINAL_WINDOW,
    create_run_dir,
    export_results,
    run_single_seed,
)
from deep_cfr_poker.plotting import plot_multiseed_results  # noqa: E402

from .config import DEFAULT_CONFIG, DEFAULT_SEEDS  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment")


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    """Parses a comma-separated seed list such as ``'1234,2025,7'``."""
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def parse_int_tuple(value: Optional[str]):
    """Parses a comma-separated integer tuple such as ``'32,32'``."""
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def build_config(args) -> dict:
    """Builds an experiment configuration from defaults and CLI overrides."""
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "evaluation_interval": args.evaluation_interval,
        "learning_rate": args.learning_rate,
        "experiment_name": args.experiment_name,
        "memory_capacity": args.memory_capacity,
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "reinitialize_advantage_networks": args.reinitialize_advantage_networks,
        "compute_exploitability": args.compute_exploitability,
        "policy_network_layers": parse_int_tuple(args.policy_network_layers),
        "advantage_network_layers": parse_int_tuple(args.advantage_network_layers),
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)
    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Leduc poker Deep CFR multi-seed validation experiment."
    )
    parser.add_argument(
        "--output-root", default="outputs", help="Root directory for outputs."
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list. Defaults to 10 fixed seeds.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
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
        help="Compute exact NashConv at every checkpoint (CPU-bound for larger games).",
    )
    parser.add_argument(
        "--policy-network-layers",
        default=None,
        help="Comma-separated MLP hidden sizes for the policy network, e.g. '32,32'.",
    )
    parser.add_argument(
        "--advantage-network-layers",
        default=None,
        help="Comma-separated MLP hidden sizes for each advantage network.",
    )
    parser.add_argument(
        "--final-window",
        type=int,
        default=DEFAULT_FINAL_WINDOW,
        help="Window length used for the final-window mean exploitability summary.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override experiment name used in the output directory.",
    )
    parser.add_argument(
        "--save-final-checkpoints",
        action="store_true",
        help="Save final model checkpoints for each seed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()

    config = build_config(args)
    seeds = parse_seeds(args.seeds)
    run_dir = create_run_dir(Path(args.output_root), config["experiment_name"])

    _configure_logging(run_dir, verbose=args.verbose)

    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Configuration: %s", config)
    _LOGGER.info("Running seeds: %s", seeds)

    results = []
    failed_seeds = []
    for seed in tqdm(seeds, desc="Deep CFR seeds"):
        try:
            results.append(
                run_single_seed(
                    seed,
                    config,
                    export_dir=run_dir,
                    save_final_checkpoint=args.save_final_checkpoints,
                    final_window=args.final_window,
                )
            )
        except Exception as exc:  # pragma: no cover -- catch-all for runtime errors
            _LOGGER.exception("Seed %s failed: %s", seed, exc)
            failed_seeds.append(
                {
                    "seed": int(seed),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    if not results:
        _LOGGER.error("All seeds failed; nothing to export.")
        return 1

    export_info = export_results(
        results,
        run_dir,
        config,
        seeds,
        failed_seeds=failed_seeds or None,
    )
    plot_multiseed_results(
        results,
        run_dir,
        exploitability_threshold=config["exploitability_threshold"],
        average_policy_value_target=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
    )

    _LOGGER.info("Completed %d/%d seeds", len(results), len(seeds))
    if failed_seeds:
        _LOGGER.warning(
            "%d seed(s) failed; see %s",
            len(failed_seeds),
            (run_dir / "failed_seeds.json").resolve(),
        )
    _LOGGER.info("Per-seed summary: %s", export_info["summary_csv"].resolve())
    _LOGGER.info("Checkpoint curves: %s", export_info["curve_csv"].resolve())
    _LOGGER.info("Aggregate summary: %s", export_info["aggregate_summary"].resolve())
    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())

    summary_numeric = export_info["summary_numeric"]
    for field_name in (
        "final_exploitability",
        "best_exploitability",
        "final_policy_value_error",
    ):
        if field_name in summary_numeric:
            _LOGGER.info(
                "Aggregate %s: %s", field_name, summary_numeric[field_name]
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
