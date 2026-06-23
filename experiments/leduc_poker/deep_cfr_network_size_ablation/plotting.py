"""Plots for the Leduc poker Deep CFR network-size ablation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


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


def _summary_stat(
    aggregate_by_variant: dict, variant_id: str, metric: str, stat: str
) -> float:
    try:
        return float(aggregate_by_variant["by_variant_id"][variant_id][metric][stat])
    except KeyError:
        return float("nan")


def _architecture_axes(variants: Sequence[Mapping[str, object]]):
    depths = sorted({int(v["architecture_depth"]) for v in variants})
    widths = sorted({int(v["architecture_width"]) for v in variants})
    return depths, widths


def _architecture_grid(
    variants: Sequence[Mapping[str, object]],
    aggregate_by_variant: dict,
    metric: str,
    stat: str = "mean",
) -> tuple[list[int], list[int], np.ndarray]:
    depths, widths = _architecture_axes(variants)
    grid = np.full((len(depths), len(widths)), np.nan, dtype=np.float64)
    depth_index = {depth: i for i, depth in enumerate(depths)}
    width_index = {width: i for i, width in enumerate(widths)}
    for variant in variants:
        variant_id = str(variant["variant_id"])
        i = depth_index[int(variant["architecture_depth"])]
        j = width_index[int(variant["architecture_width"])]
        grid[i, j] = _summary_stat(aggregate_by_variant, variant_id, metric, stat)
    return depths, widths, grid


def _plot_architecture_heatmap(
    run_dir: Path,
    variants: Sequence[Mapping[str, object]],
    aggregate_by_variant: dict,
    *,
    metric: str,
    filename: str,
    title: str,
    colorbar_label: str,
    fmt: str = ".3f",
) -> None:
    depths, widths, grid = _architecture_grid(variants, aggregate_by_variant, metric)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(widths)))
    ax.set_xticklabels([str(width) for width in widths])
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([str(depth) for depth in depths])
    ax.set_xlabel("Hidden-layer width")
    ax.set_ylabel("Number of hidden layers")
    ax.set_title(title)
    for i in range(len(depths)):
        for j in range(len(widths)):
            value = grid[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center", color="white")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_network_size_ablation(
    results: Sequence[dict],
    run_dir,
    *,
    variants: Sequence[Mapping[str, object]],
    baseline_variant_id: str,
    exploitability_threshold: float,
    average_policy_value_target: float,
    aggregate_by_variant: dict,
    paired_rows: Sequence[dict],
) -> None:
    if not results:
        raise ValueError("No results to plot.")

    run_dir = Path(run_dir)
    variant_ids = [str(v["variant_id"]) for v in variants]
    labels = {str(v["variant_id"]): str(v.get("label", v["variant_id"])) for v in variants}
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_ids)))

    for key, ylabel, title, filename, x_key in (
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Network-Size Ablation: Exploitability",
            "exploitability_by_iteration.png",
            "iterations",
        ),
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Network-Size Ablation by Nodes Touched",
            "exploitability_by_nodes.png",
            "nodes_touched",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Network-Size Ablation: Average Policy Value",
            "average_policy_value_by_iteration.png",
            "iterations",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Network-Size Ablation by Nodes Touched",
            "average_policy_value_by_nodes.png",
            "nodes_touched",
        ),
        (
            "policy_value_error",
            r"$|v(\sigma)-(-1/18)|$",
            "Network-Size Ablation: Policy-Value Error",
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
        ax.set_title(title)
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    _plot_architecture_heatmap(
        run_dir,
        variants,
        aggregate_by_variant,
        metric="final_exploitability",
        filename="final_exploitability_by_architecture.png",
        title="Final Exploitability by Network Architecture",
        colorbar_label="Mean final exploitability",
        fmt=".3f",
    )
    _plot_architecture_heatmap(
        run_dir,
        variants,
        aggregate_by_variant,
        metric="final_policy_value_error",
        filename="final_policy_value_error_by_architecture.png",
        title="Final Policy-Value Error by Network Architecture",
        colorbar_label="Mean final policy-value error",
        fmt=".4f",
    )
    _plot_architecture_heatmap(
        run_dir,
        variants,
        aggregate_by_variant,
        metric="normalised_exploitability_auc_by_iteration",
        filename="exploitability_auc_by_architecture.png",
        title="Exploitability AUC by Network Architecture",
        colorbar_label="Mean normalised exploitability AUC",
        fmt=".3f",
    )
    _plot_architecture_heatmap(
        run_dir,
        variants,
        aggregate_by_variant,
        metric="final_wall_clock_seconds",
        filename="wall_clock_by_architecture.png",
        title="Wall-Clock Cost by Network Architecture",
        colorbar_label="Mean final wall-clock seconds",
        fmt=".1f",
    )

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
        ax.set_title("Paired Architecture Differences Across Seeds")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "paired_deltas_vs_baseline.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    for diag_key, ylabel, filename, title in (
        (
            "policy_loss",
            "Policy-network loss",
            "policy_loss_diagnostic.png",
            "Network-Size Ablation: Policy-Loss Diagnostic",
        ),
        (
            "advantage_target_variance",
            "Advantage-target variance",
            "advantage_target_variance_diagnostic.png",
            "Network-Size Ablation: Advantage-Target Variance",
        ),
        (
            "policy_normalized_entropy_mean",
            "Mean policy normalised entropy",
            "policy_entropy_diagnostic.png",
            "Network-Size Ablation: Policy-Entropy Diagnostic",
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
        ax.set_title(title)
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

