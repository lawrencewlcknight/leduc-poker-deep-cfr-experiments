"""CLI for the Leduc poker Deep CFR policy-training-frequency ablation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

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
    run_single_seed,
    summarise_numeric_fields,
    write_dict_rows_csv,
    write_experiment_metadata,
    write_failed_seeds,
)
from .config import DEFAULT_CONFIG, DEFAULT_SEEDS, EXTENDED_SEEDS  # noqa: E402
from .plotting import plot_policy_training_frequency_ablation  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.policy_training_frequency")


def parse_seeds(seed_string: Optional[str]) -> List[int]:
    if not seed_string:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def parse_int_list(value: Optional[str]):
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_tuple(value: Optional[str]):
    parsed = parse_int_list(value)
    return None if parsed is None else tuple(parsed)


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
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "evaluation_interval": args.evaluation_interval,
        "policy_train_every_variants": parse_int_list(args.policy_train_every_variants),
        "reference_policy_train_every": args.reference_policy_train_every,
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
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["policy_train_every_variants"] = tuple(
        int(v) for v in config["policy_train_every_variants"]
    )
    config["reference_policy_train_every"] = int(
        config["reference_policy_train_every"]
    )
    if config["reference_policy_train_every"] not in config["policy_train_every_variants"]:
        config["reference_policy_train_every"] = int(
            config["policy_train_every_variants"][0]
        )
    return config


def _variant_config(base_config: dict, policy_train_every: int) -> dict:
    config = deepcopy(base_config)
    config["policy_network_train_every"] = int(policy_train_every)
    config["policy_training_events_expected"] = int(
        np.ceil(config["num_iterations"] / int(policy_train_every))
    )
    config["policy_gradient_steps_expected"] = int(
        config["policy_training_events_expected"]
        * int(config["policy_network_train_steps"])
    )
    return config


def _augment_result(
    result: dict,
    policy_train_every: int,
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    result["policy_network_train_every"] = int(policy_train_every)
    result["summary"] = dict(result["summary"])
    exploitability_curve = np.asarray(result["exploitability"], dtype=np.float64)
    finite_exploitability = exploitability_curve[np.isfinite(exploitability_curve)]
    fraction_below_threshold = (
        float(np.mean(finite_exploitability <= float(exploitability_threshold)))
        if finite_exploitability.size
        else float("nan")
    )
    result["summary"].update(
        {
            "policy_network_train_every": int(policy_train_every),
            "final_window_std_exploitability": final_window_std(
                result["exploitability"], window=final_window
            ),
            "exploitability_auc_by_iteration": normalised_auc(
                result["iterations"], result["exploitability"]
            ),
            "fraction_checkpoints_below_threshold": fraction_below_threshold,
        }
    )
    return result


def _group_by_variant(rows: Sequence[dict], variants: Sequence[int]) -> dict:
    grouped = {}
    for variant in variants:
        variant_rows = [
            row for row in rows
            if int(row["policy_network_train_every"]) == int(variant)
        ]
        grouped[str(int(variant))] = summarise_numeric_fields(variant_rows)
    return grouped


def _paired_differences(results: Sequence[dict], reference_variant: int) -> List[dict]:
    by_variant_seed = {
        (int(r["policy_network_train_every"]), int(r["seed"])): r
        for r in results
    }
    rows: List[dict] = []
    variants = sorted({int(r["policy_network_train_every"]) for r in results})
    seeds = sorted({int(r["seed"]) for r in results})
    for variant in variants:
        if variant == int(reference_variant):
            continue
        for seed in seeds:
            ref = by_variant_seed.get((int(reference_variant), seed))
            result = by_variant_seed.get((variant, seed))
            if ref is None or result is None:
                continue
            rows.append(
                {
                    "seed": seed,
                    "reference_policy_train_every": int(reference_variant),
                    "policy_network_train_every": int(variant),
                    "delta_final_exploitability": float(
                        result["summary"]["final_exploitability"]
                        - ref["summary"]["final_exploitability"]
                    ),
                    "delta_best_exploitability": float(
                        result["summary"]["best_exploitability"]
                        - ref["summary"]["best_exploitability"]
                    ),
                    "delta_auc": float(
                        result["summary"]["exploitability_auc_by_iteration"]
                        - ref["summary"]["exploitability_auc_by_iteration"]
                    ),
                }
            )
    return rows


def _export_ablation_npz(results: Sequence[dict], run_dir: Path, variants: Sequence[int]):
    payload = {
        "policy_train_every_variants": np.asarray(variants, dtype=np.int64),
        "seeds": np.asarray(sorted({int(r["seed"]) for r in results}), dtype=np.int64),
        "iterations": np.asarray(results[0]["iterations"], dtype=np.int64),
    }
    for variant in variants:
        variant_results = [
            r for r in results
            if int(r["policy_network_train_every"]) == int(variant)
        ]
        prefix = f"train_every_{int(variant)}"
        for key in (
            "exploitability",
            "policy_value_error",
            "nodes_touched",
            "wall_clock_seconds",
        ):
            payload[f"{prefix}_{key}"] = np.vstack(
                [np.asarray(r[key], dtype=np.float64) for r in variant_results]
            )
    path = run_dir / "ablation_curves.npz"
    np.savez_compressed(path, **payload)
    return path


def export_ablation_results(
    results: Sequence[dict],
    run_dir: Path,
    config: dict,
    seeds: Sequence[int],
    failed: Optional[Sequence[dict]] = None,
) -> dict:
    variants = [int(v) for v in config["policy_train_every_variants"]]
    info = export_results(
        results,
        run_dir,
        config,
        seeds,
        failed_seeds=failed,
        write_multiseed_npz=False,
    )
    summary_rows = [result["summary"] for result in results]

    aggregate_by_variant = {
        "by_policy_network_train_every": _group_by_variant(summary_rows, variants)
    }
    aggregate_path = run_dir / "aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate_by_variant), f, indent=2)

    paired_rows = _paired_differences(
        results, int(config["reference_policy_train_every"])
    )
    paired_csv = None
    if paired_rows:
        paired_csv = write_dict_rows_csv(
            paired_rows, run_dir / "paired_differences_vs_reference.csv"
        )

    npz_path = _export_ablation_npz(results, run_dir, variants)
    write_experiment_metadata(
        run_dir,
        config=config,
        seeds=seeds,
        completed_seeds=sorted({int(r["seed"]) for r in results}),
        extra={
            "extended_seeds_available": EXTENDED_SEEDS,
            "experiment_note": (
                "Controlled ablation varying only policy_network_train_every; "
                "evaluation_interval is fixed across all arms."
            ),
            "aggregate_summary_json": str(aggregate_path),
            "paired_differences_csv": str(paired_csv) if paired_csv else None,
            "ablation_curves_npz": str(npz_path),
        },
    )
    if failed:
        write_failed_seeds(run_dir, failed)

    return {
        **info,
        "aggregate_summary": aggregate_path,
        "paired_differences_csv": paired_csv,
        "ablation_curves_npz": npz_path,
        "aggregate_by_variant": aggregate_by_variant,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Leduc poker Deep CFR policy-training-frequency ablation."
    )
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument(
        "--policy-train-every-variants",
        default=None,
        help="Comma-separated ablation arms, e.g. '10,25,50,100'.",
    )
    parser.add_argument("--reference-policy-train-every", type=int, default=None)
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
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--save-final-checkpoints", action="store_true")
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
    _LOGGER.info("Run directory: %s", run_dir.resolve())
    _LOGGER.info("Configuration: %s", config)
    _LOGGER.info("Seeds: %s", seeds)

    results = []
    failed = []
    for policy_train_every in config["policy_train_every_variants"]:
        variant_config = _variant_config(config, int(policy_train_every))
        _LOGGER.info(
            "Running policy_network_train_every=%s (%s expected policy-training events)",
            policy_train_every,
            variant_config["policy_training_events_expected"],
        )
        for seed in tqdm(seeds, desc=f"train_every={policy_train_every}"):
            try:
                result = run_single_seed(
                    seed,
                    variant_config,
                    export_dir=run_dir,
                    save_final_checkpoint=args.save_final_checkpoints,
                    final_window=args.final_window,
                )
                results.append(
                    _augment_result(
                        result,
                        int(policy_train_every),
                        args.final_window,
                        config["exploitability_threshold"],
                    )
                )
            except Exception as exc:  # pragma: no cover
                _LOGGER.exception(
                    "Seed %s failed for policy_network_train_every=%s: %s",
                    seed,
                    policy_train_every,
                    exc,
                )
                failed.append(
                    {
                        "seed": int(seed),
                        "policy_network_train_every": int(policy_train_every),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    if not results:
        _LOGGER.error("All runs failed; nothing to export.")
        return 1

    export_info = export_ablation_results(
        results, run_dir, config, seeds, failed=failed or None
    )
    plot_policy_training_frequency_ablation(
        results,
        run_dir,
        variants=config["policy_train_every_variants"],
        reference_variant=config["reference_policy_train_every"],
        exploitability_threshold=config["exploitability_threshold"],
        average_policy_value_target=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
        aggregate_by_variant=export_info["aggregate_by_variant"],
    )

    _LOGGER.info("Completed %d/%d runs", len(results), len(seeds) * len(config["policy_train_every_variants"]))
    if failed:
        _LOGGER.warning("%d run(s) failed; see failed_seeds.json", len(failed))
    _LOGGER.info("Per-seed summary: %s", export_info["summary_csv"].resolve())
    _LOGGER.info("Checkpoint curves: %s", export_info["curve_csv"].resolve())
    _LOGGER.info("Aggregate summary: %s", export_info["aggregate_summary"].resolve())
    _LOGGER.info("All outputs saved to: %s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
