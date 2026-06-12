"""Plots for the learning-rate schedule ablation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402


def _results_for_schedule(results: Sequence[dict], schedule: str):
    return [r for r in results if str(r["schedule"]) == str(schedule)]


def _stack(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [np.asarray(result[key], dtype=np.float64) for result in results]
    return np.vstack(arrays) if arrays else np.empty((0, 0))


def _stack_diag(results: Sequence[dict], key: str) -> np.ndarray:
    arrays = [
        np.asarray(result["diagnostics"][key], dtype=np.float64)
        for result in results
    ]
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


def _summary_stat(aggregate_by_schedule: dict, schedule: str, metric: str, stat: str):
    return aggregate_by_schedule["by_schedule"][schedule][metric][stat]


def plot_lr_schedule_ablation(
    results: Sequence[dict],
    run_dir,
    *,
    schedules: Sequence[Mapping[str, object]],
    baseline_schedule: str,
    exploitability_threshold: float,
    average_policy_value_target: float,
    aggregate_by_schedule: dict,
    paired_rows: Sequence[dict],
) -> None:
    if not results:
        raise ValueError("No results to plot.")

    run_dir = Path(run_dir)
    schedule_ids = [str(s["schedule"]) for s in schedules]
    labels = {str(s["schedule"]): str(s.get("label", s["schedule"])) for s in schedules}

    fig, ax = plt.subplots(figsize=(8, 5))
    for schedule in schedule_ids:
        first = _results_for_schedule(results, schedule)[0]
        ax.plot(first["iterations"], first["learning_rate"], label=labels[schedule])
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-Rate Schedules")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "learning_rates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for key, ylabel, title, filename, x_key in (
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Learning-Rate Schedule Ablation: Exploitability",
            "exploitability_by_iteration.png",
            "iterations",
        ),
        (
            "exploitability",
            "Exploitability (NashConv/2)",
            "Learning-Rate Schedule Ablation by Nodes Touched",
            "exploitability_by_nodes.png",
            "nodes_touched",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Learning-Rate Schedule Ablation: Average Policy Value",
            "average_policy_value_by_iteration.png",
            "iterations",
        ),
        (
            "average_policy_value",
            "Average policy value for player 0",
            "Learning-Rate Schedule Ablation by Nodes Touched",
            "average_policy_value_by_nodes.png",
            "nodes_touched",
        ),
        (
            "policy_value_error",
            r"$|v(\sigma)-v^*|$",
            "Learning-Rate Schedule Ablation: Policy-Value Error",
            "policy_value_error_by_iteration.png",
            "iterations",
        ),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for schedule in schedule_ids:
            subset = _results_for_schedule(results, schedule)
            y_mean, y_se = _mean_and_se(_stack(subset, key))
            x_mean, _ = _mean_and_se(_stack(subset, x_key))
            ax.plot(x_mean, y_mean, linewidth=2, label=labels[schedule])
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
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    x_pos = np.arange(len(schedule_ids))
    final_means = [
        _summary_stat(aggregate_by_schedule, schedule, "final_exploitability", "mean")
        for schedule in schedule_ids
    ]
    final_ses = [
        _summary_stat(aggregate_by_schedule, schedule, "final_exploitability", "se")
        for schedule in schedule_ids
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos, final_means, yerr=final_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[s] for s in schedule_ids], rotation=20, ha="right")
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Learning-rate schedule")
    ax.set_ylabel("Mean final exploitability")
    ax.set_title("Final Exploitability by Learning-Rate Schedule")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_schedule.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    final_value_means = [
        _summary_stat(aggregate_by_schedule, schedule, "final_policy_value", "mean")
        for schedule in schedule_ids
    ]
    final_value_ses = [
        _summary_stat(aggregate_by_schedule, schedule, "final_policy_value", "se")
        for schedule in schedule_ids
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos, final_value_means, yerr=final_value_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[s] for s in schedule_ids], rotation=20, ha="right")
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Learning-rate schedule")
    ax.set_ylabel("Mean final average policy value for player 0")
    ax.set_title("Final Average Policy Value by Learning-Rate Schedule")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_average_policy_value_by_schedule.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    if paired_rows:
        comparison_schedules = sorted({str(row["schedule"]) for row in paired_rows})
        x_pos = np.arange(len(comparison_schedules))
        means = []
        ses = []
        for schedule in comparison_schedules:
            vals = np.asarray(
                [
                    row["delta_final_exploitability_vs_baseline"]
                    for row in paired_rows
                    if str(row["schedule"]) == schedule
                ],
                dtype=np.float64,
            )
            means.append(float(np.mean(vals)))
            ses.append(float(stats.sem(vals)) if vals.size > 1 else 0.0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x_pos, means, yerr=ses, capsize=4)
        ax.axhline(0.0, linestyle="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels.get(s, s) for s in comparison_schedules], rotation=20, ha="right")
        ax.set_xlabel("Schedule")
        ax.set_ylabel(f"Delta final exploitability vs {baseline_schedule}")
        ax.set_title("Paired Change vs Constant Baseline")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "delta_final_exploitability_vs_baseline.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    for diagnostic_key, ylabel, filename, title in (
        ("policy_loss", "Policy MSE loss", "policy_loss_diagnostic.png", "Policy-Network Loss Diagnostic"),
        ("advantage_target_variance", "Advantage target variance", "advantage_target_variance_diagnostic.png", "Advantage-Target Variance Diagnostic"),
        ("policy_normalized_entropy_mean", "Normalised policy entropy", "policy_entropy_diagnostic.png", "Policy Entropy Diagnostic"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for schedule in schedule_ids:
            subset = _results_for_schedule(results, schedule)
            iterations = np.asarray(subset[0]["iterations"], dtype=np.float64)
            mean, se = _mean_and_se(_stack_diag(subset, diagnostic_key))
            ax.plot(iterations, mean, linewidth=2, label=labels[schedule])
            ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)
