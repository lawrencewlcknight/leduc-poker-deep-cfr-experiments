"""Shared runner for targeted HP ablations on the best composite baseline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
import warnings
from copy import deepcopy
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

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

from deep_cfr_poker.constants import (  # noqa: E402
    DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    DEFAULT_EXPLOITABILITY_THRESHOLD,
)
from deep_cfr_poker.experiment_utils import (  # noqa: E402
    DEFAULT_FINAL_WINDOW,
    configure_run_logging,
    create_run_dir,
    export_results,
    json_safe,
    run_single_seed,
    write_dict_rows_csv,
    write_experiment_metadata,
    write_failed_seeds,
)
from deep_cfr_poker.plotting import set_chart_title  # noqa: E402
from experiments.leduc_poker.deep_cfr_composite_architecture_validation.config import (  # noqa: E402
    COMPOSITE_VARIANT_ID,
    DEEP_ADVANTAGE_LAYERS,
    POLICY_LAYERS,
)
from experiments.leduc_poker.deep_cfr_replay_averaging_ablation.run import (  # noqa: E402
    _augment_result as _augment_replay_result,
    _export_ablation_npz,
    _filter_variants,
    _group_by_variant,
    _paired_rows,
    _paired_summary,
    _str2bool,
    _variant_config,
    parse_int_tuple,
    parse_variant_ids,
)


DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]
DEFAULT_SEEDS_5 = DEFAULT_SEEDS
EXTENDED_SEEDS_10 = [1234, 2025, 31415, 27182, 16180, 4242, 8675309, 7, 99, 1001]

BASELINE_VARIANT_ID = "composite_best_baseline"
BASELINE_VARIANT = {
    "variant_id": BASELINE_VARIANT_ID,
    "label": "Composite best baseline",
    "hp_family": "baseline",
    "hp_value": "current_best",
    "advantage_replay_sampling": "uniform",
    "average_strategy_weighting": "uniform",
    "description": (
        "Current best candidate baseline: residual LayerNorm centred-advantage "
        "architecture, standardised advantage targets, uniform replay, and "
        "uniform average-strategy weighting."
    ),
}

BASE_COMPOSITE_HP_CONFIG = {
    "game_name": "leduc_poker",
    "num_iterations": 1500,
    "num_traversals": 320,
    "evaluation_interval": 25,
    "policy_network_layers": POLICY_LAYERS,
    "advantage_network_layers": DEEP_ADVANTAGE_LAYERS,
    "policy_network_type": "mlp",
    "advantage_network_type": "residual_layer_norm_centered_advantage_mlp",
    "learning_rate": 0.003,
    "learning_rate_schedule": "constant",
    "batch_size_advantage": 1024,
    "batch_size_strategy": 1024,
    "memory_capacity": int(1e7),
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 200,
    "advantage_network_train_steps": 200,
    "policy_network_train_every": 25,
    "policy_training_mode": "intermittent",
    "final_policy_network_train_steps": None,
    "compute_exploitability": True,
    "average_policy_value_target": DEFAULT_AVERAGE_POLICY_VALUE_TARGET,
    "exploitability_threshold": DEFAULT_EXPLOITABILITY_THRESHOLD,
    "target_processing": "standardize",
    "target_clip_value": 1.0,
    "target_standardize_epsilon": 1e-6,
    "advantage_replay_sampling": "uniform",
    "average_strategy_weighting": "uniform",
    "priority_alpha": 1.0,
    "priority_epsilon": 1e-6,
    "baseline_variant_id": BASELINE_VARIANT_ID,
    "reference_architecture_variant_id": COMPOSITE_VARIANT_ID,
}


def make_variant(
    variant_id: str,
    label: str,
    *,
    hp_family: str,
    hp_value: str,
    description: str,
    **overrides,
) -> dict:
    """Creates a variant that inherits the composite HP baseline by default."""
    variant = {
        "variant_id": variant_id,
        "label": label,
        "hp_family": hp_family,
        "hp_value": hp_value,
        "advantage_replay_sampling": "uniform",
        "average_strategy_weighting": "uniform",
        "description": description,
    }
    variant.update(overrides)
    return variant


def parse_seeds(seed_string: Optional[str], default_seeds: Sequence[int]) -> List[int]:
    if not seed_string:
        return list(default_seeds)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def build_config_from_args(
    args,
    *,
    default_config: Mapping[str, object],
    all_variants: Sequence[Mapping[str, object]],
) -> dict:
    config = deepcopy(dict(default_config))
    variant_ids = parse_variant_ids(args.variant_ids)
    variants = (
        _filter_variants(config["ablation_variants"], None)
        if variant_ids is None
        else _filter_variants(all_variants, variant_ids)
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
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["target_processing"] = "standardize"
    config["target_clip_value"] = 1.0
    config["advantage_replay_sampling"] = "uniform"
    config["average_strategy_weighting"] = "uniform"
    config["ablation_variants"] = tuple(variants)
    variant_ids = {str(v["variant_id"]) for v in variants}
    if str(config["baseline_variant_id"]) not in variant_ids:
        config["baseline_variant_id"] = str(variants[0]["variant_id"])
    return config


def _layers_label(layers: Sequence[int]) -> str:
    return "x".join(str(int(width)) for width in layers)


def _effective_final_policy_steps(config: Mapping[str, object]) -> int:
    if config.get("final_policy_network_train_steps") is not None:
        return int(config["final_policy_network_train_steps"])
    return int(config["policy_network_train_steps"])


def _augment_result(
    result: dict,
    variant_config: Mapping[str, object],
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    result = _augment_replay_result(
        result,
        variant_config,
        final_window=final_window,
        exploitability_threshold=exploitability_threshold,
    )
    result["target_processing"] = str(variant_config.get("target_processing", ""))
    result["target_clip_value"] = float(variant_config.get("target_clip_value", 1.0))
    result["learning_rate_schedule"] = str(
        variant_config.get("learning_rate_schedule", "constant")
    )
    result["policy_network_type"] = str(variant_config.get("policy_network_type", ""))
    result["advantage_network_type"] = str(
        variant_config.get("advantage_network_type", "")
    )
    result["policy_training_mode"] = str(
        variant_config.get("policy_training_mode", "intermittent")
    )
    result["final_policy_network_train_steps"] = _effective_final_policy_steps(
        variant_config
    )

    hp_fields = {
        "hp_family": str(variant_config.get("hp_family", "")),
        "hp_value": str(variant_config.get("hp_value", "")),
        "learning_rate": float(variant_config["learning_rate"]),
        "learning_rate_schedule": result["learning_rate_schedule"],
        "num_traversals": int(variant_config["num_traversals"]),
        "policy_network_layers": _layers_label(variant_config["policy_network_layers"]),
        "advantage_network_layers": _layers_label(
            variant_config["advantage_network_layers"]
        ),
        "policy_network_type": result["policy_network_type"],
        "advantage_network_type": result["advantage_network_type"],
        "batch_size_advantage": int(variant_config["batch_size_advantage"]),
        "batch_size_strategy": int(variant_config["batch_size_strategy"]),
        "memory_capacity": int(variant_config["memory_capacity"]),
        "policy_network_train_steps": int(
            variant_config["policy_network_train_steps"]
        ),
        "advantage_network_train_steps": int(
            variant_config["advantage_network_train_steps"]
        ),
        "policy_network_train_every": int(variant_config["policy_network_train_every"]),
        "final_policy_network_train_steps": result["final_policy_network_train_steps"],
        "target_processing": result["target_processing"],
        "target_clip_value": result["target_clip_value"],
        "target_standardize_epsilon": float(
            variant_config.get("target_standardize_epsilon", 1e-6)
        ),
    }
    result["summary"] = {**hp_fields, **result["summary"]}
    return result


def export_ablation_results(
    results: Sequence[dict],
    run_dir: Path,
    config: Mapping[str, object],
    seeds: Sequence[int],
    *,
    failed: Optional[Sequence[dict]] = None,
    experiment_note: str,
) -> dict:
    variants = list(config["ablation_variants"])
    info = export_results(
        results,
        run_dir,
        dict(config),
        seeds,
        failed_seeds=failed,
        write_multiseed_npz=False,
    )
    summary_rows = [result["summary"] for result in results]
    aggregate_by_variant = {"by_variant_id": _group_by_variant(summary_rows, variants)}
    aggregate_path = run_dir / "aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate_by_variant), f, indent=2)

    paired_rows = _paired_rows(results, str(config["baseline_variant_id"]))
    paired_csv = None
    paired_summary_path = None
    if paired_rows:
        paired_csv = write_dict_rows_csv(
            paired_rows, run_dir / "paired_differences_vs_baseline.csv"
        )
        paired_summary = _paired_summary(paired_rows)
        paired_summary_path = run_dir / "paired_difference_summary.json"
        with open(paired_summary_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(paired_summary), f, indent=2)

    npz_path = _export_ablation_npz(results, run_dir, variants)
    write_experiment_metadata(
        run_dir,
        config=dict(config),
        seeds=seeds,
        completed_seeds=sorted({int(r["seed"]) for r in results}),
        extra={
            "default_seeds_5": DEFAULT_SEEDS_5,
            "extended_seeds_10": EXTENDED_SEEDS_10,
            "experiment_note": experiment_note,
            "paired_difference_convention": (
                "variant minus baseline; negative exploitability, AUC, "
                "policy-value-error, or wall-clock deltas favour the variant"
            ),
            "aggregate_summary_json": str(aggregate_path),
            "paired_differences_csv": str(paired_csv) if paired_csv else None,
            "paired_difference_summary_json": (
                str(paired_summary_path) if paired_summary_path else None
            ),
            "ablation_curves_npz": str(npz_path),
        },
    )
    if failed:
        write_failed_seeds(run_dir, failed)

    return {
        **info,
        "aggregate_summary": aggregate_path,
        "paired_differences_csv": paired_csv,
        "paired_difference_summary": paired_summary_path,
        "ablation_curves_npz": npz_path,
        "aggregate_by_variant": aggregate_by_variant,
        "paired_rows": paired_rows,
    }


def _results_for_variant(results: Sequence[dict], variant_id: str):
    return [r for r in results if str(r["variant_id"]) == str(variant_id)]


def _stack(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [np.asarray(result[key], dtype=np.float64) for result in results]
    return np.vstack(arrays) if arrays else np.empty((0, 0))


def _stack_diag(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [
        np.asarray(result["diagnostics"][key], dtype=np.float64)
        for result in results
        if key in result["diagnostics"]
    ]
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


def _summary_stat(aggregate_by_variant: dict, variant_id: str, metric: str, stat: str):
    return aggregate_by_variant["by_variant_id"][variant_id][metric][stat]


def plot_hyperparameter_ablation(
    results: Sequence[dict],
    run_dir,
    *,
    variants: Sequence[Mapping[str, object]],
    baseline_variant_id: str,
    exploitability_threshold: float,
    average_policy_value_target: float,
    aggregate_by_variant: dict,
    paired_rows: Sequence[dict],
    title_prefix: str,
) -> None:
    if not results:
        raise ValueError("No results to plot.")

    run_dir = Path(run_dir)
    variant_ids = [str(v["variant_id"]) for v in variants]
    labels = {str(v["variant_id"]): str(v.get("label", v["variant_id"])) for v in variants}

    curve_specs = (
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            f"{title_prefix}: Exploitability",
            "exploitability_by_iteration.png",
            "iterations",
            True,
        ),
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            f"{title_prefix}: Exploitability by Nodes Touched",
            "exploitability_by_nodes.png",
            "nodes_touched",
            True,
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            f"{title_prefix}: Average Policy Value",
            "average_policy_value_by_iteration.png",
            "iterations",
            False,
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            f"{title_prefix}: Average Policy Value by Nodes Touched",
            "average_policy_value_by_nodes.png",
            "nodes_touched",
            False,
        ),
        (
            "policy_value_error",
            r"$|v(\sigma)-v^*|$",
            f"{title_prefix}: Policy-Value Error",
            "policy_value_error_by_iteration.png",
            "iterations",
            False,
        ),
        (
            "policy_value_error",
            r"$|v(\sigma)-v^*|$",
            f"{title_prefix}: Policy-Value Error by Nodes Touched",
            "policy_value_error_by_nodes.png",
            "nodes_touched",
            False,
        ),
    )
    for key, ylabel, title, filename, x_key, plot_threshold in curve_specs:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for variant_id in variant_ids:
            subset = _results_for_variant(results, variant_id)
            y_mean, y_se = _mean_and_se(_stack(subset, key))
            x_mean, _ = _mean_and_se(_stack(subset, x_key))
            ax.plot(x_mean, y_mean, linewidth=2, label=labels[variant_id])
            ax.fill_between(x_mean, y_mean - y_se, y_mean + y_se, alpha=0.15)
        if plot_threshold:
            ax.axhline(
                exploitability_threshold,
                linestyle="--",
                label="Exploitability threshold",
            )
        elif key == "average_policy_value":
            ax.axhline(
                average_policy_value_target,
                linestyle="--",
                label="Player 0 Nash value",
            )
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    x_pos = np.arange(len(variant_ids))
    final_means = [
        _summary_stat(aggregate_by_variant, variant_id, "final_exploitability", "mean")
        for variant_id in variant_ids
    ]
    final_ses = [
        _summary_stat(aggregate_by_variant, variant_id, "final_exploitability", "se")
        for variant_id in variant_ids
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos, final_means, yerr=final_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[v] for v in variant_ids], rotation=25, ha="right")
    ax.axhline(exploitability_threshold, linestyle="--", label="Exploitability threshold")
    ax.set_ylabel("Mean final exploitability")
    set_chart_title(ax, f"{title_prefix}: Final Exploitability by Variant")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_variant.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    final_value_means = [
        _summary_stat(aggregate_by_variant, variant_id, "final_policy_value", "mean")
        for variant_id in variant_ids
    ]
    final_value_ses = [
        _summary_stat(aggregate_by_variant, variant_id, "final_policy_value", "se")
        for variant_id in variant_ids
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos, final_value_means, yerr=final_value_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[v] for v in variant_ids], rotation=25, ha="right")
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_ylabel("Mean final average policy value for player 0")
    set_chart_title(ax, f"{title_prefix}: Final Average Policy Value by Variant")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_average_policy_value_by_variant.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if paired_rows:
        comparison_variants = sorted({str(row["variant_id"]) for row in paired_rows})
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, variant_id in enumerate(comparison_variants):
            vals = np.asarray(
                [
                    row["delta_final_exploitability_vs_baseline"]
                    for row in paired_rows
                    if str(row["variant_id"]) == variant_id
                ],
                dtype=np.float64,
            )
            ax.scatter(np.full(vals.size, i), vals, alpha=0.7)
            ax.errorbar(
                i,
                float(np.mean(vals)),
                yerr=float(stats.sem(vals)) if vals.size > 1 else 0.0,
                fmt="o",
                capsize=5,
                color="black",
            )
        ax.axhline(0.0, linestyle="--")
        ax.set_xticks(np.arange(len(comparison_variants)))
        ax.set_xticklabels([labels.get(v, v) for v in comparison_variants], rotation=25, ha="right")
        ax.set_ylabel(f"Delta final exploitability vs {baseline_variant_id}")
        set_chart_title(ax, f"{title_prefix}: Paired Differences Across Seeds")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "paired_deltas_vs_baseline.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    for metric, ylabel, filename, title in (
        (
            "advantage_target_variance",
            "Advantage-target variance",
            "advantage_target_variance_diagnostic.png",
            f"{title_prefix}: Advantage-Target Variance",
        ),
        (
            "policy_loss",
            "Policy-network loss",
            "policy_loss_diagnostic.png",
            f"{title_prefix}: Policy-Loss Diagnostic",
        ),
        (
            "policy_normalized_entropy_mean",
            "Normalised policy entropy",
            "policy_entropy_diagnostic.png",
            f"{title_prefix}: Policy Entropy",
        ),
        (
            "advantage_priority_effective_sample_size",
            "Priority effective sample size",
            "priority_effective_sample_size.png",
            f"{title_prefix}: Priority Effective Sample Size",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        plotted = False
        for variant_id in variant_ids:
            subset = _results_for_variant(results, variant_id)
            mean, se = _mean_and_se(_stack_diag(subset, metric))
            if mean.size == 0:
                continue
            nodes, _ = _mean_and_se(_stack(subset, "nodes_touched"))
            ax.plot(nodes, mean, linewidth=2, label=labels[variant_id])
            ax.fill_between(nodes, mean - se, mean + se, alpha=0.15)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Nodes Touched")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--variant-ids", default=None, help="Comma-separated variant ids.")
    parser.add_argument("--baseline-variant-id", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--policy-network-train-every", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--batch-size-advantage", type=int, default=None)
    parser.add_argument("--batch-size-strategy", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--advantage-network-train-steps", type=int, default=None)
    parser.add_argument("--final-policy-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--advantage-network-layers", default=None)
    parser.add_argument("--reinitialize-advantage-networks", type=_str2bool, default=None)
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--target-standardize-epsilon", type=float, default=None)
    parser.add_argument("--priority-alpha", type=float, default=None)
    parser.add_argument("--priority-epsilon", type=float, default=None)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--save-final-checkpoints", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_experiment(
    args,
    *,
    default_config: Mapping[str, object],
    all_variants: Sequence[Mapping[str, object]],
    default_seeds: Sequence[int],
    logger_name: str,
    experiment_note: str,
    plot_title_prefix: str,
) -> int:
    logger = logging.getLogger(logger_name)
    config = build_config_from_args(
        args, default_config=default_config, all_variants=all_variants
    )
    seeds = parse_seeds(args.seeds, default_seeds)

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(Path(args.output_root), str(config["experiment_name"]))

    configure_run_logging(run_dir, verbose=args.verbose)
    logger.info("Run directory: %s", run_dir.resolve())
    logger.info("Configuration: %s", config)
    logger.info("Seeds: %s", seeds)

    results = []
    failed = []
    for variant in config["ablation_variants"]:
        variant_config = _variant_config(config, variant)
        logger.info(
            "Running %s: %s",
            variant_config["variant_id"],
            variant_config.get("label", variant_config["variant_id"]),
        )
        for seed in tqdm(seeds, desc=str(variant_config["variant_id"])):
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
                        variant_config,
                        args.final_window,
                        float(config["exploitability_threshold"]),
                    )
                )
            except Exception as exc:  # pragma: no cover
                logger.exception(
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
        logger.error("All runs failed; nothing to export.")
        return 1

    export_info = export_ablation_results(
        results,
        run_dir,
        config,
        seeds,
        failed=failed or None,
        experiment_note=experiment_note,
    )
    plot_hyperparameter_ablation(
        results,
        run_dir,
        variants=config["ablation_variants"],
        baseline_variant_id=str(config["baseline_variant_id"]),
        exploitability_threshold=float(config["exploitability_threshold"]),
        average_policy_value_target=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
        aggregate_by_variant=export_info["aggregate_by_variant"],
        paired_rows=export_info["paired_rows"],
        title_prefix=plot_title_prefix,
    )

    logger.info(
        "Completed %d/%d runs",
        len(results),
        len(seeds) * len(config["ablation_variants"]),
    )
    if failed:
        logger.warning("%d run(s) failed; see failed_seeds.json", len(failed))
    logger.info("Per-seed summary: %s", export_info["summary_csv"].resolve())
    logger.info("Checkpoint curves: %s", export_info["curve_csv"].resolve())
    logger.info("Aggregate summary: %s", export_info["aggregate_summary"].resolve())
    logger.info("All outputs saved to: %s", run_dir.resolve())
    return 0


def run_experiment_from_cli(
    *,
    default_config: Mapping[str, object],
    all_variants: Sequence[Mapping[str, object]],
    default_seeds: Sequence[int],
    logger_name: str,
    description: str,
    experiment_note: str,
    plot_title_prefix: str,
) -> int:
    args = build_arg_parser(description).parse_args()
    return run_experiment(
        args,
        default_config=default_config,
        all_variants=all_variants,
        default_seeds=default_seeds,
        logger_name=logger_name,
        experiment_note=experiment_note,
        plot_title_prefix=plot_title_prefix,
    )


__all__ = [
    "BASELINE_VARIANT",
    "BASELINE_VARIANT_ID",
    "BASE_COMPOSITE_HP_CONFIG",
    "DEFAULT_SEEDS",
    "DEFAULT_SEEDS_5",
    "EXTENDED_SEEDS_10",
    "_augment_result",
    "_variant_config",
    "build_arg_parser",
    "build_config_from_args",
    "export_ablation_results",
    "make_variant",
    "run_experiment",
    "run_experiment_from_cli",
]
