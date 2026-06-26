"""Plots for the fair warm-start/checkpoint ablation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from deep_cfr_poker.plotting import set_chart_title


def _results_for_arm(results: Sequence[dict], arm: str):
    return [r for r in results if str(r["arm"]) == str(arm)]


def _stack(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [np.asarray(result[key], dtype=np.float64) for result in results]
    return np.vstack(arrays) if arrays else np.empty((0, 0))


def _mean_and_se(matrix: np.ndarray):
    if matrix.size == 0:
        return np.array([]), np.array([])
    se = (
        stats.sem(matrix, axis=0, nan_policy="omit")
        if matrix.shape[0] > 1
        else np.zeros(matrix.shape[1])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
    return mean, se


def plot_warm_start_fair_ablation(
    results: Sequence[dict],
    paired_curve_rows: Sequence[dict],
    paired_rows: Sequence[dict],
    run_dir,
    *,
    warm_start_boundary: int,
    average_policy_value_target: float,
) -> None:
    if not results:
        raise ValueError("No results to plot.")

    run_dir = Path(run_dir)
    arm_labels = {
        "baseline_continuous": "Continuous baseline",
        "warm_start": "Checkpoint and resume",
    }

    for key, ylabel, title, filename, x_key in (
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Fair Warm-Start Ablation",
            "exploitability_by_iteration.png",
            "iterations",
        ),
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Fair Warm-Start Ablation by Nodes Touched",
            "exploitability_by_nodes.png",
            "nodes_touched",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Fair Warm-Start Ablation: Average Policy Value",
            "average_policy_value_by_iteration.png",
            "iterations",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Fair Warm-Start Ablation by Nodes Touched",
            "average_policy_value_by_nodes.png",
            "nodes_touched",
        ),
        (
            "policy_value_error",
            r"$|v(\sigma)-v^*|$",
            "Policy-Value Error",
            "policy_value_error_by_iteration.png",
            "iterations",
        ),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for arm in ("baseline_continuous", "warm_start"):
            subset = _results_for_arm(results, arm)
            y_mat = _stack(subset, key)
            x_mat = _stack(subset, x_key)
            mean_y, se_y = _mean_and_se(y_mat)
            mean_x, _ = _mean_and_se(x_mat)
            for result in subset:
                ax.plot(result[x_key], result[key], alpha=0.20, linewidth=1)
            ax.plot(mean_x, mean_y, linewidth=2, label=f"{arm_labels[arm]} mean")
            ax.fill_between(mean_x, mean_y - se_y, mean_y + se_y, alpha=0.15)
        if x_key == "iterations":
            ax.axvline(
                warm_start_boundary,
                linestyle=":",
                linewidth=1,
                label="Warm-start boundary",
            )
        if key == "exploitability":
            ax.axhline(0.0, linestyle="--", linewidth=1, label="Nash equilibrium target")
        elif key == "average_policy_value":
            ax.axhline(
                average_policy_value_target,
                linestyle="--",
                linewidth=1,
                label="Player 0 Nash value",
            )
        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if paired_curve_rows:
        seeds = sorted({int(row["seed"]) for row in paired_curve_rows})
        iterations = sorted({int(row["iteration"]) for row in paired_curve_rows})
        matrix = np.full((len(iterations), len(seeds)), np.nan, dtype=np.float64)
        seed_index = {seed: i for i, seed in enumerate(seeds)}
        iteration_index = {iteration: i for i, iteration in enumerate(iterations)}
        for row in paired_curve_rows:
            matrix[iteration_index[int(row["iteration"])], seed_index[int(row["seed"])]] = float(
                row["delta_exploitability_warm_minus_baseline"]
            )

        mean, se = _mean_and_se(matrix.T)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.asarray(iterations, dtype=np.float64)
        for col in range(matrix.shape[1]):
            ax.plot(x, matrix[:, col], alpha=0.20, linewidth=1)
        ax.plot(x, mean, linewidth=2, label="Mean paired difference")
        ax.fill_between(x, mean - se, mean + se, alpha=0.15)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.axvline(
            warm_start_boundary,
            linestyle=":",
            linewidth=1,
            label="Warm-start boundary",
        )
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Warm-start exploitability - baseline exploitability")
        set_chart_title(ax, "Paired Exploitability Difference Over Training")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "paired_delta_exploitability_warm_minus_baseline.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    if paired_rows:
        labels = ["Final exploitability", "Best exploitability", "AUC by nodes"]
        metrics = [
            "delta_final_exploitability_warm_minus_baseline",
            "delta_best_exploitability_warm_minus_baseline",
            "delta_auc_nodes_warm_minus_baseline",
        ]
        means = []
        ses = []
        for metric in metrics:
            vals = np.asarray([row[metric] for row in paired_rows], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            means.append(float(np.mean(finite)) if finite.size else float("nan"))
            ses.append(float(stats.sem(finite)) if finite.size > 1 else 0.0)
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(len(labels))
        ax.bar(x_pos, means, yerr=ses, capsize=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_ylabel("Warm-start - baseline")
        set_chart_title(ax, "Mean Paired Warm-Start Difference")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "paired_difference_summary_bar_chart.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
