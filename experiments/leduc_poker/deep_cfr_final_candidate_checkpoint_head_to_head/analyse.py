"""Exact head-to-head analysis and seed-level inference for Experiment 27."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deep_cfr_poker.experiment_utils import json_safe, write_dict_rows_csv  # noqa: E402
from deep_cfr_poker.plotting import set_chart_title  # noqa: E402
from experiments.leduc_poker.deep_cfr_checkpoint_head_to_head.analyse import (  # noqa: E402
    run_analysis as run_shared_checkpoint_analysis,
)

from .statistics import build_inference_tables  # noqa: E402


def _read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_nodes_by_checkpoint(run_dir: Path) -> dict:
    values = defaultdict(list)
    for row in _read_csv(run_dir / "training_stage_metrics.csv"):
        values[int(row["checkpoint_iteration"])].append(float(row["nodes_touched"]))
    return {
        checkpoint: float(np.mean(checkpoint_values))
        for checkpoint, checkpoint_values in values.items()
    }


def _plot_seed_effects(seed_rows: List[dict], output_path: Path) -> None:
    values = np.asarray(
        [row["mean_later_vs_earlier_ev"] for row in seed_rows], dtype=np.float64
    )
    labels = [str(row["seed"]) for row in seed_rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(values.size)
    ax.scatter(x, values, s=42, label="Per-seed mean")
    if values.size:
        ax.axhline(float(np.mean(values)), linewidth=2, label="Mean across seeds")
    ax.axhline(0.0, linestyle="--", linewidth=1, label="No head-to-head difference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean exact EV of later vs earlier checkpoints")
    set_chart_title(ax, "Final Candidate Checkpoint Improvement Across Seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    *,
    config: Mapping[str, object],
    run_dir: Path,
    snapshots_dir: Optional[Path] = None,
) -> dict:
    """Runs shared exact evaluation, then adds seed-level statistical tests."""
    run_dir = Path(run_dir)
    outputs = run_shared_checkpoint_analysis(
        config=config,
        run_dir=run_dir,
        snapshots_dir=snapshots_dir,
    )
    pairwise_rows = _read_csv(Path(outputs["head_to_head_pairwise"]))
    seed_rows, summary_rows, pair_rows = build_inference_tables(
        pairwise_rows,
        config["checkpoint_schedule"],
    )
    node_lookup = _mean_nodes_by_checkpoint(run_dir)
    for row in pair_rows:
        row["later_nodes_touched_mean"] = node_lookup.get(
            int(row["later_checkpoint"]), float("nan")
        )
        row["earlier_nodes_touched_mean"] = node_lookup.get(
            int(row["earlier_checkpoint"]), float("nan")
        )

    seed_effects_path = run_dir / "head_to_head_primary_effect_by_seed.csv"
    inference_summary_path = run_dir / "head_to_head_inference_summary.csv"
    pairwise_inference_path = run_dir / "head_to_head_pairwise_inference.csv"
    write_dict_rows_csv(seed_rows, seed_effects_path)
    write_dict_rows_csv(summary_rows, inference_summary_path)
    write_dict_rows_csv(pair_rows, pairwise_inference_path)

    aggregate_path = run_dir / "aggregate_summary.json"
    summaries_by_estimand = {row["estimand"]: row for row in summary_rows}
    with open(aggregate_path, "w", encoding="utf-8") as handle:
        json.dump(
            json_safe(
                {
                    "analysis_unit": "independent_training_seed",
                    "evaluation": "exact OpenSpiel expected value, averaged over seats",
                    "primary_estimand": summaries_by_estimand.get(
                        "seed_mean_ev_later_vs_all_earlier_checkpoints"
                    ),
                    "adjacent_checkpoint_estimand": summaries_by_estimand.get(
                        "seed_mean_ev_vs_immediately_previous_checkpoint"
                    ),
                    "final_vs_first_estimand": summaries_by_estimand.get(
                        "final_checkpoint_ev_vs_first_checkpoint"
                    ),
                    "multiple_testing": (
                        "Secondary checkpoint-pair sign-flip p-values use Holm "
                        "family-wise error correction."
                    ),
                }
            ),
            handle,
            indent=2,
        )

    seed_effect_plot = run_dir / "head_to_head_primary_effect_by_seed.png"
    _plot_seed_effects(seed_rows, seed_effect_plot)

    outputs.update(
        {
            "head_to_head_primary_effect_by_seed": seed_effects_path,
            "head_to_head_inference_summary": inference_summary_path,
            "head_to_head_pairwise_inference": pairwise_inference_path,
            "aggregate_summary": aggregate_path,
            "head_to_head_primary_effect_plot": seed_effect_plot,
        }
    )
    return outputs


__all__ = ["run_analysis"]
