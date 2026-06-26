"""Shared runner and plotting utilities for network-architecture ablations."""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import stats

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/deep_cfr_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/deep_cfr_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tqdm import tqdm  # noqa: E402

from deep_cfr_poker.constants import DEFAULT_AVERAGE_POLICY_VALUE_TARGET  # noqa: E402
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
from deep_cfr_poker.plotting import set_chart_title  # noqa: E402


def parse_seeds(seed_string: Optional[str], default_seeds: Sequence[int]) -> list[int]:
    if not seed_string:
        return list(default_seeds)
    return [int(item.strip()) for item in seed_string.split(",") if item.strip()]


def parse_variant_ids(value: Optional[str]):
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _filter_variants(
    variants: Sequence[Mapping[str, object]], ids: Optional[Sequence[str]]
):
    if ids is None:
        return [dict(v) for v in variants]
    by_id = {str(v["variant_id"]): dict(v) for v in variants}
    missing = [variant_id for variant_id in ids if variant_id not in by_id]
    if missing:
        raise ValueError(f"Unknown variant id(s): {missing}")
    return [by_id[variant_id] for variant_id in ids]


def build_config(args, default_config: dict) -> dict:
    config = deepcopy(default_config)
    variants = _filter_variants(
        config["architecture_variants"], parse_variant_ids(args.variant_ids)
    )
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "num_traversals": args.traversals,
        "evaluation_interval": args.evaluation_interval,
        "learning_rate": args.learning_rate,
        "batch_size_advantage": args.batch_size_advantage,
        "batch_size_strategy": args.batch_size_strategy,
        "memory_capacity": args.memory_capacity,
        "reinitialize_advantage_networks": args.reinitialize_advantage_networks,
        "policy_network_train_steps": args.policy_network_train_steps,
        "advantage_network_train_steps": args.advantage_network_train_steps,
        "policy_network_train_every": args.policy_network_train_every,
        "compute_exploitability": args.compute_exploitability,
        "baseline_variant_id": args.baseline_variant_id,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    config["architecture_variants"] = tuple(variants)
    variant_ids = {str(v["variant_id"]) for v in variants}
    if str(config["baseline_variant_id"]) not in variant_ids:
        config["baseline_variant_id"] = str(variants[0]["variant_id"])
    return config


def _variant_config(base_config: dict, variant: Mapping[str, object]) -> dict:
    config = deepcopy(base_config)
    config.update(dict(variant))
    config["policy_network_layers"] = tuple(variant["policy_network_layers"])
    config["advantage_network_layers"] = tuple(variant["advantage_network_layers"])
    config["policy_network_type"] = str(variant.get("policy_network_type", "mlp"))
    config["advantage_network_type"] = str(variant.get("advantage_network_type", "mlp"))
    return config


def _best_exploitability_iteration(iterations, exploitability) -> float:
    values = np.asarray(exploitability, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    finite_indices = np.where(finite)[0]
    local = int(np.nanargmin(values[finite]))
    return int(np.asarray(iterations)[finite_indices[local]])


def _variant_fields(variant_config: Mapping[str, object]) -> dict:
    variant_id = str(variant_config["variant_id"])
    fields = {
        "variant_id": variant_id,
        "variant_label": str(variant_config.get("label", variant_id)),
        "policy_network_type": str(variant_config.get("policy_network_type", "mlp")),
        "advantage_network_type": str(variant_config.get("advantage_network_type", "mlp")),
        "policy_network_layers": tuple(variant_config["policy_network_layers"]),
        "advantage_network_layers": tuple(variant_config["advantage_network_layers"]),
    }
    for key in (
        "network_treatment",
        "varied_network",
        "architecture_depth",
        "architecture_width",
        "dropout_probability",
        "policy_architecture_label",
        "advantage_architecture_label",
    ):
        if key in variant_config:
            fields[key] = variant_config[key]
    return fields


def _augment_result(
    result: dict,
    variant_config: Mapping[str, object],
    final_window: int,
    exploitability_threshold: float,
) -> dict:
    fields = _variant_fields(variant_config)
    result["variant_id"] = fields["variant_id"]
    result["variant_label"] = fields["variant_label"]
    result["config_label"] = fields["variant_id"]
    result["policy_network_type"] = fields["policy_network_type"]
    result["advantage_network_type"] = fields["advantage_network_type"]
    result["policy_training_mode"] = str(
        variant_config.get("policy_training_mode", "intermittent")
    )
    result["reinitialize_advantage_networks"] = bool(
        variant_config["reinitialize_advantage_networks"]
    )
    result["policy_network_train_every"] = int(
        variant_config["policy_network_train_every"]
    )

    exploitability_curve = np.asarray(result["exploitability"], dtype=np.float64)
    finite_exploitability = exploitability_curve[np.isfinite(exploitability_curve)]
    fraction_below_threshold = (
        float(np.mean(finite_exploitability <= float(exploitability_threshold)))
        if finite_exploitability.size
        else float("nan")
    )

    metric_fields = {
        "best_exploitability_iteration": _best_exploitability_iteration(
            result["iterations"], result["exploitability"]
        ),
        "final_window_std_exploitability": final_window_std(
            result["exploitability"], window=final_window
        ),
        "normalised_exploitability_auc_by_iteration": normalised_auc(
            result["iterations"], result["exploitability"]
        ),
        "normalised_exploitability_auc_by_nodes": normalised_auc(
            result["nodes_touched"], result["exploitability"]
        ),
        "fraction_checkpoints_below_threshold": fraction_below_threshold,
        "policy_training_events": int(
            result["summary"].get("final_policy_training_events", 0)
        ),
        "policy_network_gradient_steps": int(
            result["summary"].get("final_policy_gradient_steps", 0)
        ),
    }
    result["summary"] = {
        **fields,
        **dict(result["summary"]),
        **metric_fields,
    }
    return result


def _group_by_variant(rows: Sequence[dict], variants: Sequence[Mapping[str, object]]):
    grouped = {}
    for variant in variants:
        variant_id = str(variant["variant_id"])
        variant_rows = [row for row in rows if row["variant_id"] == variant_id]
        grouped[variant_id] = summarise_numeric_fields(variant_rows)
    return grouped


def _paired_rows(results: Sequence[dict], baseline_variant_id: str) -> list[dict]:
    by_variant_seed = {
        (str(result["variant_id"]), int(result["seed"])): result["summary"]
        for result in results
    }
    seeds = sorted({int(result["seed"]) for result in results})
    variants = sorted({str(result["variant_id"]) for result in results})
    rows = []
    for seed in seeds:
        baseline = by_variant_seed.get((str(baseline_variant_id), seed))
        if baseline is None:
            continue
        for variant_id in variants:
            if variant_id == str(baseline_variant_id):
                continue
            comparison = by_variant_seed.get((variant_id, seed))
            if comparison is None:
                continue
            rows.append(
                {
                    "seed": int(seed),
                    "baseline_variant_id": str(baseline_variant_id),
                    "variant_id": variant_id,
                    "delta_final_exploitability_vs_baseline": float(
                        comparison["final_exploitability"]
                        - baseline["final_exploitability"]
                    ),
                    "delta_best_exploitability_vs_baseline": float(
                        comparison["best_exploitability"]
                        - baseline["best_exploitability"]
                    ),
                    "delta_final_window_mean_exploitability_vs_baseline": float(
                        comparison["final_window_mean_exploitability"]
                        - baseline["final_window_mean_exploitability"]
                    ),
                    "delta_auc_by_iteration_vs_baseline": float(
                        comparison["normalised_exploitability_auc_by_iteration"]
                        - baseline["normalised_exploitability_auc_by_iteration"]
                    ),
                    "delta_auc_by_nodes_vs_baseline": float(
                        comparison["normalised_exploitability_auc_by_nodes"]
                        - baseline["normalised_exploitability_auc_by_nodes"]
                    ),
                    "delta_policy_value_error_vs_baseline": float(
                        comparison["final_policy_value_error"]
                        - baseline["final_policy_value_error"]
                    ),
                    "delta_wall_clock_seconds_vs_baseline": float(
                        comparison["final_wall_clock_seconds"]
                        - baseline["final_wall_clock_seconds"]
                    ),
                }
            )
    return rows


def _paired_summary(rows: Sequence[dict]) -> dict:
    output = {}
    for variant_id in sorted({str(row["variant_id"]) for row in rows}):
        variant_rows = [row for row in rows if str(row["variant_id"]) == variant_id]
        output[variant_id] = {}
        for field in variant_rows[0].keys():
            if field in {"seed", "baseline_variant_id", "variant_id"}:
                continue
            vals = np.asarray([row[field] for row in variant_rows], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                output[variant_id][field] = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                    "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
                    "n": int(finite.size),
                    "fraction_variant_better": float(np.mean(finite < 0.0)),
                }
    return output


def _stack_padded(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.empty((0, 0))
    arrays = [np.asarray(array, dtype=np.float64) for array in arrays]
    max_len = max(array.shape[0] for array in arrays)
    padded = []
    for array in arrays:
        if array.shape[0] == max_len:
            padded.append(array)
        else:
            pad = np.full(max_len - array.shape[0], np.nan, dtype=np.float64)
            padded.append(np.concatenate([array, pad]))
    return np.vstack(padded)


def _export_ablation_npz(
    results: Sequence[dict],
    run_dir: Path,
    variants: Sequence[Mapping[str, object]],
):
    payload = {
        "variant_ids": np.asarray([str(v["variant_id"]) for v in variants]),
        "seeds": np.asarray(sorted({int(r["seed"]) for r in results}), dtype=np.int64),
        "iterations": np.asarray(results[0]["iterations"], dtype=np.int64),
    }
    for variant in variants:
        variant_id = str(variant["variant_id"])
        subset = [r for r in results if str(r["variant_id"]) == variant_id]
        for key in (
            "exploitability",
            "average_policy_value",
            "policy_value_error",
            "nodes_touched",
            "wall_clock_seconds",
        ):
            payload[f"{variant_id}_{key}"] = _stack_padded(
                [np.asarray(r[key], dtype=np.float64) for r in subset]
            )
    path = run_dir / "ablation_curves.npz"
    np.savez_compressed(path, **payload)
    return path


def _summary_lookup(summary: Mapping[str, object], field: str, stat: str) -> float:
    try:
        return float(summary[field][stat])
    except KeyError:
        return float("nan")


def _variant_summary_rows(
    variants: Sequence[Mapping[str, object]], aggregate_by_variant: dict
) -> list[dict]:
    rows = []
    by_variant = aggregate_by_variant["by_variant_id"]
    for variant in variants:
        variant_id = str(variant["variant_id"])
        summary = by_variant.get(variant_id, {})
        row = _variant_fields(variant)
        row.update(
            {
                "final_exploitability_mean": _summary_lookup(
                    summary, "final_exploitability", "mean"
                ),
                "final_exploitability_se": _summary_lookup(
                    summary, "final_exploitability", "se"
                ),
                "final_policy_value_error_mean": _summary_lookup(
                    summary, "final_policy_value_error", "mean"
                ),
                "final_policy_value_error_se": _summary_lookup(
                    summary, "final_policy_value_error", "se"
                ),
                "normalised_exploitability_auc_by_iteration_mean": _summary_lookup(
                    summary, "normalised_exploitability_auc_by_iteration", "mean"
                ),
                "normalised_exploitability_auc_by_nodes_mean": _summary_lookup(
                    summary, "normalised_exploitability_auc_by_nodes", "mean"
                ),
                "final_wall_clock_seconds_mean": _summary_lookup(
                    summary, "final_wall_clock_seconds", "mean"
                ),
            }
        )
        rows.append(row)
    return rows


def export_ablation_results(
    results: Sequence[dict],
    run_dir: Path,
    config: dict,
    seeds: Sequence[int],
    *,
    metadata_extra: Optional[dict] = None,
    failed: Optional[Sequence[dict]] = None,
) -> dict:
    variants = list(config["architecture_variants"])
    info = export_results(
        results,
        run_dir,
        config,
        seeds,
        failed_seeds=failed,
        write_multiseed_npz=False,
    )
    summary_rows = [result["summary"] for result in results]
    aggregate_by_variant = {"by_variant_id": _group_by_variant(summary_rows, variants)}
    aggregate_path = run_dir / "aggregate_summary.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate_by_variant), f, indent=2)

    variant_summary_csv = write_dict_rows_csv(
        _variant_summary_rows(variants, aggregate_by_variant),
        run_dir / "variant_summary.csv",
    )
    paired_rows = _paired_rows(results, str(config["baseline_variant_id"]))
    paired_csv = None
    paired_summary_path = None
    if paired_rows:
        paired_csv = write_dict_rows_csv(
            paired_rows, run_dir / "paired_differences_vs_baseline.csv"
        )
        paired_summary_path = run_dir / "paired_difference_summary.json"
        with open(paired_summary_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(_paired_summary(paired_rows)), f, indent=2)

    npz_path = _export_ablation_npz(results, run_dir, variants)
    extra = {
        "aggregate_summary_json": str(aggregate_path),
        "variant_summary_csv": str(variant_summary_csv),
        "paired_differences_csv": str(paired_csv) if paired_csv else None,
        "paired_difference_summary_json": (
            str(paired_summary_path) if paired_summary_path else None
        ),
        "ablation_curves_npz": str(npz_path),
    }
    if metadata_extra:
        extra.update(metadata_extra)
    write_experiment_metadata(
        run_dir,
        config=config,
        seeds=seeds,
        completed_seeds=sorted({int(r["seed"]) for r in results}),
        extra=extra,
    )
    if failed:
        write_failed_seeds(run_dir, failed)
    return {
        **info,
        "aggregate_summary": aggregate_path,
        "variant_summary_csv": variant_summary_csv,
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
    mean = np.nanmean(matrix, axis=0)
    se = (
        stats.sem(matrix, axis=0, nan_policy="omit")
        if matrix.shape[0] > 1
        else np.zeros(matrix.shape[1])
    )
    return mean, se


def _summary_stat(
    aggregate_by_variant: dict, variant_id: str, metric: str, stat: str
) -> float:
    try:
        return float(aggregate_by_variant["by_variant_id"][variant_id][metric][stat])
    except KeyError:
        return float("nan")


def plot_architecture_ablation(
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
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variant_ids), 2)))

    for key, ylabel, suffix, filename, x_key in (
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Exploitability",
            "exploitability_by_iteration.png",
            "iterations",
        ),
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Exploitability by Nodes Touched",
            "exploitability_by_nodes.png",
            "nodes_touched",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Average Policy Value",
            "average_policy_value_by_iteration.png",
            "iterations",
        ),
        (
            "policy_value_error",
            r"$|v(\sigma)-(-1/18)|$",
            "Policy-Value Error",
            "policy_value_error_by_iteration.png",
            "iterations",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for color, variant_id in zip(colors, variant_ids):
            subset = _results_for_variant(results, variant_id)
            y_mean, y_se = _mean_and_se(_stack(subset, key))
            x_mean, _ = _mean_and_se(_stack(subset, x_key))
            ax.plot(x_mean, y_mean, linewidth=2, label=labels[variant_id], color=color)
            ax.fill_between(x_mean, y_mean - y_se, y_mean + y_se, alpha=0.12, color=color)
        if key == "exploitability":
            ax.axhline(0.0, linestyle="--", color="black", label="Nash equilibrium target")
            ax.axhline(
                exploitability_threshold,
                linestyle=":",
                color="black",
                label="Exploitability threshold",
            )
        elif key == "average_policy_value":
            ax.axhline(
                average_policy_value_target,
                linestyle="--",
                color="black",
                label="Player 0 Nash value",
            )
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        set_chart_title(ax, f"{title_prefix}: {suffix}")
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    for metric, ylabel, title, filename, fmt in (
        (
            "final_exploitability",
            "Mean final exploitability",
            "Final Exploitability by Variant",
            "final_exploitability_by_variant.png",
            ".3f",
        ),
        (
            "final_policy_value_error",
            "Mean final policy-value error",
            "Final Policy-Value Error by Variant",
            "final_policy_value_error_by_variant.png",
            ".4f",
        ),
        (
            "normalised_exploitability_auc_by_iteration",
            "Mean normalised exploitability AUC",
            "Exploitability AUC by Variant",
            "exploitability_auc_by_variant.png",
            ".3f",
        ),
    ):
        x_pos = np.arange(len(variant_ids))
        means = [
            _summary_stat(aggregate_by_variant, variant_id, metric, "mean")
            for variant_id in variant_ids
        ]
        ses = [
            _summary_stat(aggregate_by_variant, variant_id, metric, "se")
            for variant_id in variant_ids
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x_pos, means, yerr=ses, capsize=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels[v] for v in variant_ids], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, f"{title_prefix}: {title}")
        ax.grid(True, axis="y")
        for i, value in enumerate(means):
            if np.isfinite(value):
                ax.text(i, value, format(value, fmt), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if paired_rows:
        comparison_variants = [
            variant_id for variant_id in variant_ids if variant_id != str(baseline_variant_id)
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, variant_id in enumerate(comparison_variants):
            vals = np.asarray(
                [
                    row["delta_final_exploitability_vs_baseline"]
                    for row in paired_rows
                    if str(row["variant_id"]) == variant_id
                ],
                dtype=np.float64,
            )
            if vals.size == 0:
                continue
            ax.scatter(np.full(vals.size, i), vals, alpha=0.65)
            ax.errorbar(
                i,
                float(np.mean(vals)),
                yerr=float(stats.sem(vals)) if vals.size > 1 else 0.0,
                fmt="o",
                capsize=5,
                color="black",
            )
        ax.axhline(0.0, linestyle="--", color="black")
        ax.set_xticks(np.arange(len(comparison_variants)))
        ax.set_xticklabels([labels.get(v, v) for v in comparison_variants], rotation=25, ha="right")
        ax.set_ylabel(f"Delta final exploitability vs {baseline_variant_id}")
        set_chart_title(ax, f"{title_prefix}: Paired Differences Across Seeds")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "paired_deltas_vs_baseline.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    for diag_key, ylabel, filename, title in (
        (
            "policy_loss",
            "Policy-network loss",
            "policy_loss_diagnostic.png",
            "Policy-Loss Diagnostic",
        ),
        (
            "advantage_target_variance",
            "Advantage-target variance",
            "advantage_target_variance_diagnostic.png",
            "Advantage-Target Variance",
        ),
        (
            "policy_normalized_entropy_mean",
            "Mean policy normalised entropy",
            "policy_entropy_diagnostic.png",
            "Policy-Entropy Diagnostic",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for color, variant_id in zip(colors, variant_ids):
            subset = _results_for_variant(results, variant_id)
            mean, se = _mean_and_se(_stack_diag(subset, diag_key))
            if mean.size == 0:
                continue
            iterations = np.asarray(subset[0]["iterations"], dtype=np.float64)
            ax.plot(iterations, mean, linewidth=2, label=labels[variant_id], color=color)
            ax.fill_between(iterations, mean - se, mean + se, alpha=0.12, color=color)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, f"{title_prefix}: {title}")
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--variant-ids", default=None, help="Comma-separated subset of variant ids.")
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
    parser.add_argument("--reinitialize-advantage-networks", type=_str2bool, default=None)
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--final-window", type=int, default=DEFAULT_FINAL_WINDOW)
    parser.add_argument("--save-final-checkpoints", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main_from_config(
    *,
    default_config: dict,
    default_seeds: Sequence[int],
    description: str,
    logger_name: str,
    plot_title_prefix: str,
    metadata_extra: Optional[dict] = None,
) -> int:
    args = _build_arg_parser(description).parse_args()
    config = build_config(args, default_config)
    seeds = parse_seeds(args.seeds, default_seeds)
    logger = logging.getLogger(logger_name)

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = create_run_dir(Path(args.output_root), config["experiment_name"])

    configure_run_logging(run_dir, verbose=args.verbose)
    logger.info("Run directory: %s", run_dir.resolve())
    logger.info("Configuration: %s", config)
    logger.info("Seeds: %s", seeds)

    results = []
    failed = []
    for variant in config["architecture_variants"]:
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
                        config["exploitability_threshold"],
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
        metadata_extra=metadata_extra,
        failed=failed or None,
    )
    plot_architecture_ablation(
        results,
        run_dir,
        variants=config["architecture_variants"],
        baseline_variant_id=str(config["baseline_variant_id"]),
        exploitability_threshold=float(config["exploitability_threshold"]),
        average_policy_value_target=float(
            config.get("average_policy_value_target", DEFAULT_AVERAGE_POLICY_VALUE_TARGET)
        ),
        aggregate_by_variant=export_info["aggregate_by_variant"],
        paired_rows=export_info["paired_rows"],
        title_prefix=plot_title_prefix,
    )

    logger.info(
        "Completed %d/%d runs",
        len(results),
        len(seeds) * len(config["architecture_variants"]),
    )
    if failed:
        logger.warning("%d run(s) failed; see failed_seeds.json", len(failed))
    logger.info("Per-seed summary: %s", export_info["summary_csv"].resolve())
    logger.info("Checkpoint curves: %s", export_info["curve_csv"].resolve())
    logger.info("Aggregate summary: %s", export_info["aggregate_summary"].resolve())
    logger.info("Variant summary: %s", export_info["variant_summary_csv"].resolve())
    logger.info("All outputs saved to: %s", run_dir.resolve())
    return 0


__all__ = [
    "build_config",
    "export_ablation_results",
    "main_from_config",
    "parse_seeds",
    "plot_architecture_ablation",
]
