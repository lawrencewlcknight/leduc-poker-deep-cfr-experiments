"""Plots for the final-only average-policy training ablation."""

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


def _summary_stat(aggregate_by_variant: dict, variant_id: str, metric: str, stat: str):
    return aggregate_by_variant["by_variant_id"][variant_id][metric][stat]


def _short_labels(variants: Sequence[Mapping[str, object]]) -> list[str]:
    return [str(v["variant_id"]).replace("_", "\n") for v in variants]


def plot_final_only_policy_training_ablation(
    results: Sequence[dict],
    run_dir,
    *,
    variants: Sequence[Mapping[str, object]],
    reference_variant_id: str,
    exploitability_threshold: float,
    average_policy_value_target: float,
    aggregate_by_variant: dict,
) -> None:
    if not results:
        raise ValueError("No results to plot.")

    run_dir = Path(run_dir)
    iterations = np.asarray(results[0]["iterations"], dtype=np.float64)
    variant_ids = [str(v["variant_id"]) for v in variants]
    variant_labels = {str(v["variant_id"]): str(v["label"]) for v in variants}

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in variant_ids:
        mean, se = _mean_and_se(_stack(_results_for_variant(results, variant_id), "exploitability"))
        ax.plot(iterations, mean, linewidth=2, label=variant_labels[variant_id])
        ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv/2)")
    ax.set_title("Average-Policy Training Timing: Exploitability")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exploitability_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in variant_ids:
        variant_results = _results_for_variant(results, variant_id)
        exp_mean, exp_se = _mean_and_se(_stack(variant_results, "exploitability"))
        nodes_mean, _ = _mean_and_se(_stack(variant_results, "nodes_touched"))
        ax.plot(nodes_mean, exp_mean, linewidth=2, label=variant_labels[variant_id])
        ax.fill_between(nodes_mean, exp_mean - exp_se, exp_mean + exp_se, alpha=0.15)
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv/2)")
    ax.set_title("Average-Policy Training Timing by Nodes Touched")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exploitability_by_nodes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in variant_ids:
        mean, se = _mean_and_se(
            _stack(_results_for_variant(results, variant_id), "average_policy_value")
        )
        ax.plot(iterations, mean, linewidth=2, label=variant_labels[variant_id])
        ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average policy value for player 0")
    ax.set_title("Average-Policy Training Timing: Average Policy Value")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in variant_ids:
        variant_results = _results_for_variant(results, variant_id)
        value_mean, value_se = _mean_and_se(_stack(variant_results, "average_policy_value"))
        nodes_mean, _ = _mean_and_se(_stack(variant_results, "nodes_touched"))
        ax.plot(nodes_mean, value_mean, linewidth=2, label=variant_labels[variant_id])
        ax.fill_between(nodes_mean, value_mean - value_se, value_mean + value_se, alpha=0.15)
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Average policy value for player 0")
    ax.set_title("Average-Policy Training Timing by Nodes Touched")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_nodes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in variant_ids:
        mean, se = _mean_and_se(_stack(_results_for_variant(results, variant_id), "policy_value_error"))
        ax.plot(iterations, mean, linewidth=2, label=variant_labels[variant_id])
        ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$|v(\sigma) - v^*|$")
    ax.set_title("Average-Policy Training Timing: Policy-Value Error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "policy_value_error_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    labels = _short_labels(variants)
    x_pos = np.arange(len(labels))
    final_means = [
        _summary_stat(aggregate_by_variant, variant_id, "final_exploitability", "mean")
        for variant_id in variant_ids
    ]
    final_ses = [
        _summary_stat(aggregate_by_variant, variant_id, "final_exploitability", "se")
        for variant_id in variant_ids
    ]
    value_means = [
        _summary_stat(aggregate_by_variant, variant_id, "final_policy_value_error", "mean")
        for variant_id in variant_ids
    ]
    value_ses = [
        _summary_stat(aggregate_by_variant, variant_id, "final_policy_value_error", "se")
        for variant_id in variant_ids
    ]
    final_value_means = [
        _summary_stat(aggregate_by_variant, variant_id, "final_policy_value", "mean")
        for variant_id in variant_ids
    ]
    final_value_ses = [
        _summary_stat(aggregate_by_variant, variant_id, "final_policy_value", "se")
        for variant_id in variant_ids
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos, final_means, yerr=final_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Average-policy training regime")
    ax.set_ylabel("Final exploitability")
    ax.set_title("Final Exploitability by Training Regime")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_regime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos, final_value_means, yerr=final_value_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Average-policy training regime")
    ax.set_ylabel("Final average policy value for player 0")
    ax.set_title("Final Average Policy Value by Training Regime")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_average_policy_value_by_regime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x_pos, value_means, yerr=value_ses, capsize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Average-policy training regime")
    ax.set_ylabel(r"Final $|v(\sigma) - v^*|$")
    ax.set_title("Final Policy-Value Error by Training Regime")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(run_dir / "final_policy_value_error_by_regime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for key, ylabel, title, filename in (
        ("policy_training_events", "Cumulative policy-training events", "Cumulative Average-Policy Training Events", "cumulative_policy_training_events.png"),
        ("policy_gradient_steps", "Cumulative policy-gradient steps", "Cumulative Average-Policy Gradient Steps", "cumulative_policy_gradient_steps.png"),
        ("policy_loss", "Average-policy MSE loss", "Average-Policy Loss by Training Regime", "policy_loss_by_regime.png"),
        ("policy_normalized_entropy_mean", "Mean normalised policy entropy", "Policy Entropy by Training Regime", "policy_entropy_by_regime.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        for variant_id in variant_ids:
            mean, se = _mean_and_se(_stack_diag(_results_for_variant(results, variant_id), key))
            ax.plot(iterations, mean, linewidth=2, label=variant_labels[variant_id])
            if key in {"policy_loss", "policy_normalized_entropy_mean"}:
                ax.fill_between(iterations, mean - se, mean + se, alpha=0.15)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    ref_by_seed = {
        int(r["seed"]): r
        for r in _results_for_variant(results, str(reference_variant_id))
    }
    paired = []
    for variant_id in variant_ids:
        if variant_id == str(reference_variant_id):
            continue
        deltas = []
        for result in _results_for_variant(results, variant_id):
            ref = ref_by_seed.get(int(result["seed"]))
            if ref is not None:
                deltas.append(
                    result["summary"]["final_exploitability"]
                    - ref["summary"]["final_exploitability"]
                )
        if deltas:
            paired.append((variant_id, np.asarray(deltas, dtype=np.float64)))
    if paired:
        labels = [item[0].replace("_", "\n") for item in paired]
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
        ax.set_ylabel("Delta final exploitability vs intermittent baseline")
        ax.set_title("Paired Final-Exploitability Differences")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "paired_final_exploitability_delta_vs_reference.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
