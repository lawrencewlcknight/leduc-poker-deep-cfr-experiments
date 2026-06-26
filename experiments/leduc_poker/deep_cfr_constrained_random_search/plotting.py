"""Plots for the constrained Deep CFR random-search experiment."""

from __future__ import annotations

import warnings
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from deep_cfr_poker.plotting import set_chart_title


def _mean_se(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), 0.0
    return (
        float(np.mean(finite)),
        float(stats.sem(finite)) if finite.size > 1 else 0.0,
    )


def _barh_final_exploitability(rows, run_dir: Path, filename: str, title: str, threshold: float):
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get("final_exploitability_mean", np.inf)),
            str(row["config_label"]),
        ),
        reverse=True,
    )
    labels = [row["config_label"] for row in ordered]
    means = [float(row.get("final_exploitability_mean", np.nan)) for row in ordered]
    ses = [float(row.get("final_exploitability_se", 0.0)) for row in ordered]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels) + 1.5)))
    ax.barh(labels, means, xerr=ses, capsize=3, alpha=0.85)
    ax.axvline(0.0, linestyle="--", label="Nash equilibrium target")
    ax.set_xlabel("Mean final exploitability")
    set_chart_title(ax, title)
    ax.grid(True, axis="x")
    ax.legend()
    fig.tight_layout()
    path = run_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _barh_final_average_policy_value(
    rows,
    run_dir: Path,
    filename: str,
    title: str,
    target: float,
):
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get("final_policy_value_mean", np.inf)),
            str(row["config_label"]),
        ),
        reverse=True,
    )
    labels = [row["config_label"] for row in ordered]
    means = [float(row.get("final_policy_value_mean", np.nan)) for row in ordered]
    ses = [float(row.get("final_policy_value_se", 0.0)) for row in ordered]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(labels) + 1.5)))
    ax.barh(labels, means, xerr=ses, capsize=3, alpha=0.85)
    ax.axvline(target, linestyle="--", label="Player 0 Nash value")
    ax.set_xlabel("Mean final average policy value for player 0")
    set_chart_title(ax, title)
    ax.grid(True, axis="x")
    ax.legend()
    fig.tight_layout()
    path = run_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _curve_groups(curve_rows: Sequence[Mapping[str, object]], stage: str):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in curve_rows:
        if str(row.get("stage")) != str(stage):
            continue
        label = str(row["config_label"])
        iteration = int(row["iteration"])
        grouped[label][iteration].append(row)
    return grouped


def _plot_curve(
    curve_rows,
    run_dir: Path,
    *,
    stage: str,
    y_key: str,
    x_key: str,
    ylabel: str,
    title: str,
    filename: str,
    threshold: Optional[float] = None,
    threshold_label: str = "Nash equilibrium target",
):
    grouped = _curve_groups(curve_rows, stage)
    if not grouped:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, by_iteration in sorted(grouped.items()):
        xs = []
        means = []
        ses = []
        for iteration, rows in sorted(by_iteration.items()):
            if x_key == "iteration":
                x_value = float(iteration)
            else:
                x_value, _ = _mean_se([row.get(x_key, np.nan) for row in rows])
            mean, se = _mean_se([row.get(y_key, np.nan) for row in rows])
            xs.append(x_value)
            means.append(mean)
            ses.append(se)
        x_arr = np.asarray(xs, dtype=np.float64)
        mean_arr = np.asarray(means, dtype=np.float64)
        se_arr = np.asarray(ses, dtype=np.float64)
        ax.plot(x_arr, mean_arr, linewidth=2, label=label)
        ax.fill_between(x_arr, mean_arr - se_arr, mean_arr + se_arr, alpha=0.15)
    if threshold is not None:
        ax.axhline(threshold, linestyle="--", label=threshold_label)
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    set_chart_title(ax, title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    path = run_dir / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_random_search(
    *,
    run_dir,
    screening_config_summary: Sequence[Mapping[str, object]],
    confirmation_config_summary: Sequence[Mapping[str, object]],
    curve_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
    exploitability_threshold: float,
    average_policy_value_target: float,
) -> list[Path]:
    run_dir = Path(run_dir)
    paths = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        path = _barh_final_exploitability(
            screening_config_summary,
            run_dir,
            "screening_final_exploitability_by_config.png",
            "Screening Stage: Mean Final Exploitability",
            exploitability_threshold,
        )
        if path:
            paths.append(path)

        path = _barh_final_average_policy_value(
            screening_config_summary,
            run_dir,
            "screening_final_average_policy_value_by_config.png",
            "Screening Stage: Mean Final Average Policy Value",
            average_policy_value_target,
        )
        if path:
            paths.append(path)

        path = _barh_final_exploitability(
            confirmation_config_summary,
            run_dir,
            "confirmation_final_exploitability_by_config.png",
            "Confirmation Stage: Mean Final Exploitability",
            exploitability_threshold,
        )
        if path:
            paths.append(path)

        path = _barh_final_average_policy_value(
            confirmation_config_summary,
            run_dir,
            "confirmation_final_average_policy_value_by_config.png",
            "Confirmation Stage: Mean Final Average Policy Value",
            average_policy_value_target,
        )
        if path:
            paths.append(path)

        for path in (
            _plot_curve(
                curve_rows,
                run_dir,
                stage="confirmation",
                y_key="exploitability",
                x_key="iteration",
                ylabel="Exploitability (NashConv/2)",
                title="Confirmation Stage: Exploitability Curves",
                filename="confirmation_exploitability_by_iteration.png",
                threshold=0.0,
            ),
            _plot_curve(
                curve_rows,
                run_dir,
                stage="confirmation",
                y_key="average_policy_value",
                x_key="iteration",
                ylabel="Average policy value for player 0",
                title="Confirmation Stage: Average Policy Value Curves",
                filename="confirmation_average_policy_value_by_iteration.png",
                threshold=average_policy_value_target,
                threshold_label="Player 0 Nash value",
            ),
            _plot_curve(
                curve_rows,
                run_dir,
                stage="confirmation",
                y_key="average_policy_value",
                x_key="nodes_touched",
                ylabel="Average policy value for player 0",
                title="Confirmation Stage: Average Policy Value by Nodes Touched",
                filename="confirmation_average_policy_value_by_nodes.png",
                threshold=average_policy_value_target,
                threshold_label="Player 0 Nash value",
            ),
            _plot_curve(
                curve_rows,
                run_dir,
                stage="confirmation",
                y_key="exploitability",
                x_key="nodes_touched",
                ylabel="Exploitability (NashConv/2)",
                title="Confirmation Stage: Exploitability by Nodes Touched",
                filename="confirmation_exploitability_by_nodes.png",
                threshold=0.0,
            ),
            _plot_curve(
                curve_rows,
                run_dir,
                stage="confirmation",
                y_key="policy_value_error",
                x_key="iteration",
                ylabel=r"$|v(\sigma)-v^*|$",
                title="Confirmation Stage: Policy-Value Error",
                filename="confirmation_policy_value_error_by_iteration.png",
            ),
        ):
            if path:
                paths.append(path)

        if paired_rows:
            labels = sorted({str(row["config_label"]) for row in paired_rows})
            means = []
            ses = []
            for label in labels:
                vals = [
                    row["delta_final_exploitability_vs_baseline"]
                    for row in paired_rows
                    if str(row["config_label"]) == label
                ]
                mean, se = _mean_se(vals)
                means.append(mean)
                ses.append(se)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(labels) + 1.5)))
            ax.barh(labels, means, xerr=ses, capsize=3, alpha=0.85)
            ax.axvline(0.0, color="black", linewidth=1)
            ax.set_xlabel("Final exploitability difference vs baseline")
            set_chart_title(ax, "Paired Difference Relative to Experiment 2 Baseline")
            ax.grid(True, axis="x")
            fig.tight_layout()
            path = run_dir / "confirmation_paired_difference_vs_baseline.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

    return paths
