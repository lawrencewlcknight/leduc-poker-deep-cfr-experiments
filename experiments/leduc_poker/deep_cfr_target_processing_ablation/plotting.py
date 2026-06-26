"""Plots for the target-processing ablation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from deep_cfr_poker.plotting import set_chart_title


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


def _processed_variance_matrix(results: Sequence[dict]) -> np.ndarray:
    mats = []
    for result in results:
        diag = result["diagnostics"]
        p0 = np.asarray(
            diag["processed_advantage_target_variance_player_0"], dtype=np.float64
        )
        p1 = np.asarray(
            diag["processed_advantage_target_variance_player_1"], dtype=np.float64
        )
        mats.append(np.nanmean(np.vstack([p0, p1]), axis=0))
    return np.vstack(mats) if mats else np.empty((0, 0))


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


def plot_target_processing_ablation(
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

    for key, ylabel, title, filename, x_key in (
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Target-Processing Ablation: Exploitability",
            "exploitability_by_iteration.png",
            "iterations",
        ),
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Target-Processing Ablation by Nodes Touched",
            "exploitability_by_nodes.png",
            "nodes_touched",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Target-Processing Ablation: Average Policy Value",
            "average_policy_value_by_iteration.png",
            "iterations",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Target-Processing Ablation by Nodes Touched",
            "average_policy_value_by_nodes.png",
            "nodes_touched",
        ),
        (
            "policy_value_error",
            r"$|v(\sigma)-v^*|$",
            "Target-Processing Ablation: Policy-Value Error",
            "policy_value_error_by_iteration.png",
            "iterations",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for variant_id in variant_ids:
            subset = _results_for_variant(results, variant_id)
            y_mean, y_se = _mean_and_se(_stack(subset, key))
            x_mean, _ = _mean_and_se(_stack(subset, x_key))
            ax.plot(x_mean, y_mean, linewidth=2, label=labels[variant_id])
            ax.fill_between(x_mean, y_mean - y_se, y_mean + y_se, alpha=0.15)
        if key == "exploitability":
            ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
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
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_ylabel("Mean final exploitability")
    set_chart_title(ax, "Final Exploitability by Target-Processing Variant")
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
    set_chart_title(ax, "Final Average Policy Value by Target-Processing Variant")
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
        set_chart_title(ax, "Paired Target-Processing Differences Across Seeds")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "paired_deltas_vs_baseline.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    for diagnostic_matrix, ylabel, filename, title in (
        (
            _processed_variance_matrix,
            "Processed advantage-target variance",
            "processed_target_variance.png",
            "Target-Processing Ablation: Processed Target Variance",
        ),
        (
            lambda subset: _stack_diag(subset, "policy_loss"),
            "Policy-network loss",
            "policy_loss_diagnostic.png",
            "Target-Processing Ablation: Policy-Loss Diagnostic",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for variant_id in variant_ids:
            subset = _results_for_variant(results, variant_id)
            mean, se = _mean_and_se(diagnostic_matrix(subset))
            iterations = np.asarray(subset[0]["iterations"], dtype=np.float64)
            ax.plot(iterations, mean, linewidth=2, label=labels[variant_id])
            ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
