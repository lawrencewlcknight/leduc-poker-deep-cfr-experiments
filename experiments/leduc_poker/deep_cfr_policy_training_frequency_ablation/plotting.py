"""Plots for the policy-training-frequency ablation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from deep_cfr_poker.plotting import set_chart_title


def _results_for_variant(results: Sequence[dict], variant: int):
    return [
        r for r in results
        if int(r["policy_network_train_every"]) == int(variant)
    ]


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
    se = stats.sem(matrix, axis=0, nan_policy="omit") if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(matrix, axis=0)
    return mean, se


def _summary_stat(aggregate_by_variant: dict, variant: int, metric: str, stat: str):
    return aggregate_by_variant["by_policy_network_train_every"][str(int(variant))][metric][stat]


def plot_policy_training_frequency_ablation(
    results: Sequence[dict],
    run_dir,
    *,
    variants: Sequence[int],
    reference_variant: int,
    exploitability_threshold: float,
    average_policy_value_target: float,
    aggregate_by_variant: dict,
) -> None:
    if not results:
        raise ValueError("No results to plot.")

    run_dir = Path(run_dir)
    iterations = np.asarray(results[0]["iterations"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        variant_results = _results_for_variant(results, int(variant))
        mean, se = _mean_and_se(_stack(variant_results, "exploitability"))
        ax.plot(iterations, mean, linewidth=2, label=f"train every {variant}")
        ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv/2)")
    set_chart_title(ax, "Policy-Training-Frequency Ablation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exploitability_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        variant_results = _results_for_variant(results, int(variant))
        mean_exp, se_exp = _mean_and_se(_stack(variant_results, "exploitability"))
        mean_nodes, _ = _mean_and_se(_stack(variant_results, "nodes_touched"))
        ax.plot(mean_nodes, mean_exp, linewidth=2, label=f"train every {variant}")
        ax.fill_between(mean_nodes, mean_exp - se_exp, mean_exp + se_exp, alpha=0.15)
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv/2)")
    set_chart_title(ax, "Policy-Training-Frequency Ablation by Nodes")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exploitability_by_nodes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        variant_results = _results_for_variant(results, int(variant))
        mean, se = _mean_and_se(_stack(variant_results, "average_policy_value"))
        ax.plot(iterations, mean, linewidth=2, label=f"train every {variant}")
        ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average policy value for player 0")
    set_chart_title(ax, "Average Policy Value by Policy-Training Frequency")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        variant_results = _results_for_variant(results, int(variant))
        mean_value, se_value = _mean_and_se(_stack(variant_results, "average_policy_value"))
        mean_nodes, _ = _mean_and_se(_stack(variant_results, "nodes_touched"))
        ax.plot(mean_nodes, mean_value, linewidth=2, label=f"train every {variant}")
        ax.fill_between(mean_nodes, mean_value - se_value, mean_value + se_value, alpha=0.15)
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Average policy value for player 0")
    set_chart_title(ax, "Average Policy Value by Nodes Touched")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_nodes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in variants:
        variant_results = _results_for_variant(results, int(variant))
        mean, se = _mean_and_se(_stack(variant_results, "policy_value_error"))
        ax.plot(iterations, mean, linewidth=2, label=f"train every {variant}")
        ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$|v(\sigma) - v^*|$")
    set_chart_title(ax, "Policy-Value Error by Policy-Training Frequency")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "policy_value_error_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    labels = [str(v) for v in variants]
    x_pos = np.arange(len(labels))
    final_means = [
        _summary_stat(aggregate_by_variant, int(v), "final_exploitability", "mean")
        for v in variants
    ]
    final_ses = [
        _summary_stat(aggregate_by_variant, int(v), "final_exploitability", "se")
        for v in variants
    ]
    final_value_means = [
        _summary_stat(aggregate_by_variant, int(v), "final_policy_value", "mean")
        for v in variants
    ]
    final_value_ses = [
        _summary_stat(aggregate_by_variant, int(v), "final_policy_value", "se")
        for v in variants
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos, final_means, yerr=final_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("CFR iterations between policy-network training events")
    ax.set_ylabel("Final exploitability")
    set_chart_title(ax, "Final Exploitability by Policy-Training Frequency")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_frequency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos, final_value_means, yerr=final_value_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("CFR iterations between policy-network training events")
    ax.set_ylabel("Final average policy value for player 0")
    set_chart_title(ax, "Final Average Policy Value by Policy-Training Frequency")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_average_policy_value_by_frequency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    auc_means = [
        _summary_stat(aggregate_by_variant, int(v), "exploitability_auc_by_iteration", "mean")
        for v in variants
    ]
    auc_ses = [
        _summary_stat(aggregate_by_variant, int(v), "exploitability_auc_by_iteration", "se")
        for v in variants
    ]
    window_means = [
        _summary_stat(aggregate_by_variant, int(v), "final_window_mean_exploitability", "mean")
        for v in variants
    ]
    window_ses = [
        _summary_stat(aggregate_by_variant, int(v), "final_window_mean_exploitability", "se")
        for v in variants
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(variants, window_means, yerr=window_ses, marker="o", capsize=4, label="Final-window mean")
    ax.errorbar(variants, auc_means, yerr=auc_ses, marker="s", capsize=4, label="Normalised AUC")
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("CFR iterations between policy-network training events")
    ax.set_ylabel("Exploitability summary; lower is better")
    set_chart_title(ax, "Stability and Overall Exploitability")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "stability_summaries_by_frequency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for key, ylabel, title, filename in (
        ("policy_loss", "Average-policy MSE loss", "Average-Policy Loss", "policy_loss_by_frequency.png"),
        ("policy_training_events", "Cumulative policy-training events", "Cumulative Policy-Training Events", "cumulative_policy_training_events.png"),
        ("policy_normalized_entropy_mean", "Mean normalised policy entropy", "Policy Entropy by Training Frequency", "policy_entropy_by_frequency.png"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in variants:
            variant_results = _results_for_variant(results, int(variant))
            mean, _ = _mean_and_se(_stack_diag(variant_results, key))
            ax.plot(iterations, mean, linewidth=2, label=f"train every {variant}")
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    paired = []
    ref_by_seed = {
        int(r["seed"]): r for r in _results_for_variant(results, int(reference_variant))
    }
    for variant in variants:
        if int(variant) == int(reference_variant):
            continue
        deltas = []
        for result in _results_for_variant(results, int(variant)):
            ref = ref_by_seed.get(int(result["seed"]))
            if ref is not None:
                deltas.append(
                    result["summary"]["final_exploitability"]
                    - ref["summary"]["final_exploitability"]
                )
        if deltas:
            paired.append((int(variant), np.asarray(deltas, dtype=np.float64)))
    if paired:
        labels = [str(item[0]) for item in paired]
        x_pos = np.arange(len(labels))
        means = [float(np.mean(item[1])) for item in paired]
        ses = [
            float(stats.sem(item[1])) if item[1].size > 1 else 0.0
            for item in paired
        ]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x_pos, means, yerr=ses, capsize=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("Comparison policy-training frequency")
        ax.set_ylabel(f"Delta final exploitability vs {reference_variant}")
        set_chart_title(ax, "Paired Final-Exploitability Differences")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "paired_final_exploitability_delta_vs_reference.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
