"""Plots for the advantage-network reinitialisation ablation."""

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
    return [str(v["label"]) for v in variants]


def plot_advantage_reinitialisation_ablation(
    results: Sequence[dict],
    run_dir,
    *,
    variants: Sequence[Mapping[str, object]],
    reference_variant_id: str,
    comparison_variant_id: str,
    exploitability_threshold: float,
    average_policy_value_target: float,
    aggregate_by_variant: dict,
) -> None:
    if not results:
        raise ValueError("No results to plot.")

    run_dir = Path(run_dir)
    iterations = np.asarray(results[0]["iterations"], dtype=np.float64)
    variant_ids = [str(v["variant_id"]) for v in variants]
    labels = {str(v["variant_id"]): str(v["label"]) for v in variants}

    colours = {
        "reinit_false_warm_started_advantage": "tab:blue",
        "reinit_true_reset_advantage": "tab:orange",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_id in variant_ids:
        subset = _results_for_variant(results, variant_id)
        mean, se = _mean_and_se(_stack(subset, "exploitability"))
        for result in subset:
            ax.plot(
                result["iterations"],
                result["exploitability"],
                alpha=0.12,
                linewidth=1,
                color=colours.get(variant_id),
            )
        ax.plot(iterations, mean, linewidth=2, color=colours.get(variant_id), label=labels[variant_id])
        ax.fill_between(iterations, mean - se, mean + se, alpha=0.18, color=colours.get(variant_id))
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1, label="Nash equilibrium target")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv/2)")
    set_chart_title(ax, "Advantage-Network Reinitialisation Ablation")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exploitability_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_id in variant_ids:
        subset = _results_for_variant(results, variant_id)
        exp_mean, exp_se = _mean_and_se(_stack(subset, "exploitability"))
        nodes_mean, _ = _mean_and_se(_stack(subset, "nodes_touched"))
        ax.plot(nodes_mean, exp_mean, linewidth=2, color=colours.get(variant_id), label=labels[variant_id])
        ax.fill_between(nodes_mean, exp_mean - exp_se, exp_mean + exp_se, alpha=0.18, color=colours.get(variant_id))
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1, label="Nash equilibrium target")
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv/2)")
    set_chart_title(ax, "Sample Efficiency by Reinitialisation Setting")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exploitability_by_nodes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_id in variant_ids:
        subset = _results_for_variant(results, variant_id)
        mean, se = _mean_and_se(_stack(subset, "average_policy_value"))
        for result in subset:
            ax.plot(
                result["iterations"],
                result["average_policy_value"],
                alpha=0.12,
                linewidth=1,
                color=colours.get(variant_id),
            )
        ax.plot(
            iterations,
            mean,
            linewidth=2,
            color=colours.get(variant_id),
            label=labels[variant_id],
        )
        ax.fill_between(
            iterations,
            mean - se,
            mean + se,
            alpha=0.18,
            color=colours.get(variant_id),
        )
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        color="black",
        linewidth=1,
        label="Player 0 Nash value",
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average policy value for player 0")
    set_chart_title(ax, "Average Policy Value by Reinitialisation Setting")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant_id in variant_ids:
        subset = _results_for_variant(results, variant_id)
        value_mean, value_se = _mean_and_se(_stack(subset, "average_policy_value"))
        nodes_mean, _ = _mean_and_se(_stack(subset, "nodes_touched"))
        ax.plot(nodes_mean, value_mean, linewidth=2, color=colours.get(variant_id), label=labels[variant_id])
        ax.fill_between(nodes_mean, value_mean - value_se, value_mean + value_se, alpha=0.18, color=colours.get(variant_id))
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        color="black",
        linewidth=1,
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
    for variant_id in variant_ids:
        mean, se = _mean_and_se(_stack(_results_for_variant(results, variant_id), "policy_value_error"))
        ax.plot(iterations, mean, linewidth=2, color=colours.get(variant_id), label=labels[variant_id])
        ax.fill_between(iterations, mean - se, mean + se, alpha=0.18, color=colours.get(variant_id))
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$|v(\sigma) - v^*|$")
    set_chart_title(ax, "Policy-Value Error by Reinitialisation Setting")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "policy_value_error_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    x_labels = _short_labels(variants)
    x_pos = np.arange(len(x_labels))
    for metric, title, filename in (
        ("final_exploitability", "Final Exploitability", "final_exploitability_by_variant.png"),
        ("best_exploitability", "Best Exploitability", "best_exploitability_by_variant.png"),
        ("final_window_mean_exploitability", "Final-Window Mean Exploitability", "final_window_exploitability_by_variant.png"),
        ("exploitability_auc_by_iteration", "Normalised Exploitability AUC", "exploitability_auc_by_variant.png"),
        ("final_policy_value", "Final Average Policy Value", "final_average_policy_value_by_variant.png"),
    ):
        means = [
            _summary_stat(aggregate_by_variant, variant_id, metric, "mean")
            for variant_id in variant_ids
        ]
        ses = [
            _summary_stat(aggregate_by_variant, variant_id, metric, "se")
            for variant_id in variant_ids
        ]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x_pos, means, yerr=ses, capsize=4)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        if metric == "final_policy_value":
            ax.axhline(
                average_policy_value_target,
                linestyle="--",
                label="Player 0 Nash value",
            )
        else:
            ax.axhline(0.0, linestyle="--", label="Nash equilibrium target")
        ax.set_ylabel(metric)
        set_chart_title(ax, title)
        ax.grid(True, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    for diag_key, title, ylabel in (
        ("advantage_target_variance", "Advantage-Target Variance", "Variance"),
        ("advantage_grad_norm_player_0", "Advantage-Network Gradient Norm: Player 0", "Gradient norm"),
        ("advantage_grad_norm_player_1", "Advantage-Network Gradient Norm: Player 1", "Gradient norm"),
        ("policy_loss", "Average-Policy Network Loss", "MSE loss"),
        ("policy_normalized_entropy_mean", "Average-Policy Entropy", "Normalised entropy"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for variant_id in variant_ids:
            mean, se = _mean_and_se(_stack_diag(_results_for_variant(results, variant_id), diag_key))
            ax.plot(iterations, mean, linewidth=2, color=colours.get(variant_id), label=labels[variant_id])
            ax.fill_between(iterations, mean - se, mean + se, alpha=0.18, color=colours.get(variant_id))
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / f"{diag_key}_diagnostic.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    by_variant_seed = {
        (str(r["variant_id"]), int(r["seed"])): r
        for r in results
    }
    seeds = sorted({int(r["seed"]) for r in results})
    deltas = []
    kept_seeds = []
    for seed in seeds:
        ref = by_variant_seed.get((str(reference_variant_id), seed))
        comp = by_variant_seed.get((str(comparison_variant_id), seed))
        if ref is not None and comp is not None:
            kept_seeds.append(seed)
            deltas.append(comp["summary"]["final_exploitability"] - ref["summary"]["final_exploitability"])
    if deltas:
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(len(kept_seeds))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.bar(x_pos, deltas)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(seed) for seed in kept_seeds])
        ax.set_xlabel("Seed")
        ax.set_ylabel("Final exploitability difference\n(True - False)")
        set_chart_title(ax, "Paired Difference in Final Exploitability")
        ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(run_dir / "paired_final_exploitability_delta_true_minus_false.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
