"""Head-to-head analysis pipeline for the checkpoint stability experiment.

Walks the experiment's ``snapshots/`` directory, loads every recognised
snapshot into a :class:`deep_cfr_poker.snapshots.LoadedPolicy`, and produces:

* per-checkpoint exploitability + value metrics;
* exact pairwise head-to-head EV (both long-form CSV and matrix CSVs);
* per-seed monotonicity summaries and across-seed strength aggregates;
* heatmaps and strength curves;
* an optional Monte Carlo cross-check.

All outputs are placed directly in ``run_dir`` so the run directory looks the
same as every other experiment in the repository.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pyspiel

from deep_cfr_poker.evaluation import (
    aggregate_head_to_head,
    aggregate_strength,
    evaluate_checkpoint_metrics,
    exact_pairwise_head_to_head,
    monotonicity_summary,
    monte_carlo_head_to_head,
    strength_per_checkpoint,
)
from deep_cfr_poker.experiment_utils import (
    write_dict_rows_csv,
    write_matrix_csv,
)
from deep_cfr_poker.plotting import (
    plot_pairwise_heatmap,
    plot_scatter_annotated,
    plot_strength_curve_with_errorbars,
)
from deep_cfr_poker.snapshots import (
    LoadedPolicy,
    SnapshotMetadata,
    discover_snapshots,
)


_LOGGER = logging.getLogger(__name__)


def _node_summary_by_checkpoint(run_dir: Path) -> Dict[int, dict]:
    """Returns mean and SE nodes touched for each saved checkpoint."""
    path = Path(run_dir) / "training_stage_metrics.csv"
    if not path.exists():
        return {}
    values: Dict[int, List[float]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            checkpoint = row.get("checkpoint_iteration", row.get("target_iteration"))
            nodes = row.get("nodes_touched", row.get("nodes_touched_total"))
            if checkpoint in (None, "") or nodes in (None, ""):
                continue
            values[int(checkpoint)].append(float(nodes))

    summaries: Dict[int, dict] = {}
    for checkpoint, checkpoint_values in values.items():
        arr = np.asarray(checkpoint_values, dtype=np.float64)
        summaries[checkpoint] = {
            "nodes_touched_mean": float(np.mean(arr)),
            "nodes_touched_sem": (
                float(np.std(arr, ddof=1) / np.sqrt(arr.size))
                if arr.size > 1
                else 0.0
            ),
            "nodes_touched_n": int(arr.size),
        }
    return summaries


def _load_policies(
    game,
    snapshot_records: Sequence[SnapshotMetadata],
    *,
    allowed_iterations: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[int, LoadedPolicy]]:
    """Loads every accepted snapshot into a :class:`LoadedPolicy`."""
    allowed = set(int(x) for x in allowed_iterations) if allowed_iterations else None
    policies: Dict[str, Dict[int, LoadedPolicy]] = defaultdict(dict)
    for record in snapshot_records:
        if allowed is not None and int(record.iteration) not in allowed:
            continue
        # Prefer policy_snapshot files when both kinds exist for the same
        # (seed, iteration) pair.
        existing = policies.get(str(record.seed), {}).get(int(record.iteration))
        if existing is not None and record.kind != "policy_snapshot":
            continue
        _LOGGER.info(
            "Loading snapshot seed=%s iteration=%s (%s) from %s",
            record.seed,
            record.iteration,
            record.kind,
            record.path,
        )
        policies[str(record.seed)][int(record.iteration)] = LoadedPolicy(
            game, record.path
        )
    return policies


def _build_inventory_rows(
    snapshot_records: Sequence[SnapshotMetadata],
) -> List[dict]:
    rows: List[dict] = []
    for record in snapshot_records:
        rows.append(
            {
                "seed": str(record.seed),
                "iteration": int(record.iteration),
                "kind": str(record.kind),
                "path": str(record.path),
                "size_mb": float(Path(record.path).stat().st_size / (1024 ** 2))
                if Path(record.path).exists()
                else float("nan"),
            }
        )
    rows.sort(key=lambda row: (str(row["seed"]), int(row["iteration"])))
    return rows


def _build_loaded_policy_rows(
    policies_by_seed: Mapping[str, Mapping[int, LoadedPolicy]],
) -> List[dict]:
    rows: List[dict] = []
    for seed, policies in policies_by_seed.items():
        for iteration in sorted(policies):
            pol = policies[iteration]
            meta = pol.metadata
            rows.append(
                {
                    "seed": str(seed),
                    "filename_iteration": int(iteration),
                    "internal_checkpoint_iteration": meta.get("checkpoint_iteration"),
                    "internal_training_iteration": meta.get("internal_iteration"),
                    "kind": meta.get("type", ""),
                    "path": str(meta.get("path", "")),
                }
            )
    rows.sort(key=lambda row: (str(row["seed"]), int(row["filename_iteration"])))
    return rows


def _seed_to_matrix_for_records(
    records,
    sorted_iterations: Sequence[int],
) -> Dict[str, np.ndarray]:
    by_seed: Dict[str, Dict[int, Dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        by_seed[record.seed][record.checkpoint_a][record.checkpoint_b] = (
            record.A_EV_seat_averaged
        )
    out: Dict[str, np.ndarray] = {}
    for seed, table in by_seed.items():
        matrix = np.full(
            (len(sorted_iterations), len(sorted_iterations)),
            np.nan,
            dtype=np.float64,
        )
        for r_idx, row_iter in enumerate(sorted_iterations):
            for c_idx, col_iter in enumerate(sorted_iterations):
                value = table.get(row_iter, {}).get(col_iter)
                if value is None and row_iter == col_iter:
                    value = 0.0
                if value is not None:
                    matrix[r_idx, c_idx] = float(value)
        out[seed] = matrix
    return out


def _select_mc_pairs(
    iterations: Sequence[int],
    mode: str,
) -> List[tuple]:
    iterations = list(iterations)
    if mode == "all_pairs":
        return [(a, b) for a in iterations for b in iterations if a != b]
    if mode == "milestones":
        if not iterations:
            return []
        first = iterations[0]
        last = iterations[-1]
        if first == last:
            return []
        return [(last, first), (first, last)]
    # Default: adjacent.
    return [(iterations[i], iterations[i - 1]) for i in range(1, len(iterations))]


def run_analysis(
    *,
    config: Mapping[str, object],
    run_dir: Path,
    snapshots_dir: Optional[Path] = None,
) -> dict:
    """Runs the head-to-head analysis. Returns a dict of output paths."""
    run_dir = Path(run_dir)
    if snapshots_dir is None:
        snapshots_dir = run_dir / "snapshots"

    game = pyspiel.load_game(str(config["game_name"]))

    snapshot_records = discover_snapshots(snapshots_dir)
    if not snapshot_records:
        raise FileNotFoundError(
            f"No policy snapshots discovered under {snapshots_dir}. Run the "
            f"trainer first."
        )

    inventory_rows = _build_inventory_rows(snapshot_records)
    inventory_csv = run_dir / "checkpoint_inventory.csv"
    write_dict_rows_csv(inventory_rows, inventory_csv)

    policies_by_seed = _load_policies(
        game,
        snapshot_records,
        allowed_iterations=tuple(config["checkpoint_schedule"]),  # type: ignore[arg-type]
    )
    if bool(config.get("require_complete_checkpoint_schedule", False)):
        required = {int(value) for value in config["checkpoint_schedule"]}
        policies_by_seed = {
            seed: policies
            for seed, policies in policies_by_seed.items()
            if set(policies) == required
        }
    if not policies_by_seed:
        raise RuntimeError(
            "Snapshot inventory is non-empty but no policies were loaded — "
            "check that the snapshot iterations match checkpoint_schedule."
        )

    loaded_policy_rows = _build_loaded_policy_rows(policies_by_seed)
    write_dict_rows_csv(loaded_policy_rows, run_dir / "loaded_policy_inventory.csv")

    value_target = float(
        config.get(
            "known_leduc_game_value",
            config.get("average_policy_value_target", -0.085606424078),
        )
    )

    metrics = evaluate_checkpoint_metrics(
        game,
        policies_by_seed,
        known_game_value=value_target,
    )
    metric_rows = [dataclasses.asdict(m) for m in metrics]
    metrics_csv = run_dir / "checkpoint_exploitability_metrics.csv"
    write_dict_rows_csv(metric_rows, metrics_csv)

    h2h_records = exact_pairwise_head_to_head(game, policies_by_seed)
    h2h_rows = [dataclasses.asdict(r) for r in h2h_records]
    pairwise_csv = run_dir / "head_to_head_pairwise.csv"
    write_dict_rows_csv(h2h_rows, pairwise_csv)

    epsilon = float(config.get("equivalence_epsilon", 1e-3))
    mean_matrix, win_fraction_matrix, sorted_iterations = aggregate_head_to_head(
        h2h_records, equivalence_epsilon=epsilon
    )

    mean_matrix_csv = run_dir / "head_to_head_mean_matrix.csv"
    write_matrix_csv(
        mean_matrix,
        sorted_iterations,
        sorted_iterations,
        mean_matrix_csv,
        index_name="checkpoint_A",
    )
    win_fraction_csv = run_dir / "head_to_head_seed_win_fraction_matrix.csv"
    write_matrix_csv(
        win_fraction_matrix,
        sorted_iterations,
        sorted_iterations,
        win_fraction_csv,
        index_name="checkpoint_A",
    )

    seed_to_matrix = _seed_to_matrix_for_records(h2h_records, sorted_iterations)

    monotonicity_rows = [
        dataclasses.asdict(s)
        for s in monotonicity_summary(
            seed_to_matrix,
            sorted_iterations,
            equivalence_epsilon=epsilon,
        )
    ]
    monotonicity_csv = run_dir / "head_to_head_monotonicity_by_seed.csv"
    write_dict_rows_csv(monotonicity_rows, monotonicity_csv)

    strength_rows = strength_per_checkpoint(seed_to_matrix, sorted_iterations)
    strength_dict_rows = [dataclasses.asdict(s) for s in strength_rows]

    # Join exploitability metrics into the strength CSV so the thesis can
    # read a single file when comparing equilibrium quality with strength.
    metric_lookup = {(m.seed, m.checkpoint): m for m in metrics}
    for row in strength_dict_rows:
        key = (str(row["seed"]), int(row["checkpoint"]))
        m = metric_lookup.get(key)
        row["nash_conv"] = float(m.nash_conv) if m is not None else float("nan")
        row["exploitability"] = float(m.exploitability) if m is not None else float("nan")
        row["average_policy_value"] = (
            float(m.average_policy_value) if m is not None else float("nan")
        )
        row["policy_value_error"] = (
            float(m.policy_value_error) if m is not None else float("nan")
        )
    strength_csv = run_dir / "head_to_head_strength_by_checkpoint.csv"
    write_dict_rows_csv(strength_dict_rows, strength_csv)

    aggregate_rows = aggregate_strength(strength_rows)
    node_summaries = _node_summary_by_checkpoint(run_dir)
    # Add aggregated policy-quality columns for convenience.
    expl_lookup: Dict[int, List[float]] = defaultdict(list)
    avg_value_lookup: Dict[int, List[float]] = defaultdict(list)
    for m in metrics:
        if np.isfinite(m.exploitability):
            expl_lookup[m.checkpoint].append(m.exploitability)
        if np.isfinite(m.average_policy_value):
            avg_value_lookup[m.checkpoint].append(m.average_policy_value)
    for row in aggregate_rows:
        ckpt = int(row["checkpoint"])
        row.update(node_summaries.get(ckpt, {}))
        values = np.asarray(expl_lookup.get(ckpt, []), dtype=np.float64)
        row["exploitability_mean"] = (
            float(np.mean(values)) if values.size else float("nan")
        )
        row["exploitability_sem"] = (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0 if values.size == 1 else float("nan")
        )
        values = np.asarray(avg_value_lookup.get(ckpt, []), dtype=np.float64)
        row["average_policy_value_mean"] = (
            float(np.mean(values)) if values.size else float("nan")
        )
        row["average_policy_value_sem"] = (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0 if values.size == 1 else float("nan")
        )
    aggregate_csv = run_dir / "head_to_head_strength_aggregate.csv"
    write_dict_rows_csv(aggregate_rows, aggregate_csv)

    # Best checkpoint per seed under each criterion.
    best_rows: List[dict] = []
    for seed, policies in policies_by_seed.items():
        per_seed = [m for m in metrics if m.seed == seed]
        per_seed_strength = [s for s in strength_rows if s.seed == seed]
        if per_seed:
            best_by_expl = min(per_seed, key=lambda m: m.exploitability)
            best_rows.append(
                {
                    "seed": str(seed),
                    "criterion": "lowest_exploitability",
                    "checkpoint": int(best_by_expl.checkpoint),
                    "value": float(best_by_expl.exploitability),
                }
            )
        if per_seed_strength:
            best_by_strength = max(
                per_seed_strength,
                key=lambda s: (
                    s.mean_EV_vs_all_other_checkpoints
                    if np.isfinite(s.mean_EV_vs_all_other_checkpoints)
                    else -np.inf
                ),
            )
            best_rows.append(
                {
                    "seed": str(seed),
                    "criterion": "strongest_head_to_head",
                    "checkpoint": int(best_by_strength.checkpoint),
                    "value": float(best_by_strength.mean_EV_vs_all_other_checkpoints),
                }
            )
    best_csv = run_dir / "best_checkpoint_summary.csv"
    write_dict_rows_csv(best_rows, best_csv)

    # ---------------- plots ----------------
    annotate = bool(config.get("annotate_heatmap", True))
    use_nodes = str(config.get("temporal_x_axis", "checkpoint")) == "nodes_touched"
    if use_nodes:
        checkpoint_labels = [
            f"{node_summaries[i]['nodes_touched_mean'] / 1e6:.1f}M"
            if i in node_summaries
            else str(i)
            for i in sorted_iterations
        ]
        heatmap_xlabel = "Policy checkpoint B (nodes touched)"
        heatmap_ylabel = "Policy checkpoint A (nodes touched)"
    else:
        checkpoint_labels = list(sorted_iterations)
        heatmap_xlabel = "Checkpoint B iteration"
        heatmap_ylabel = "Checkpoint A iteration"
    plot_pairwise_heatmap(
        mean_matrix,
        checkpoint_labels,
        checkpoint_labels,
        output_path=run_dir / "head_to_head_mean_matrix.png",
        title=(
            "Mean exact head-to-head EV across seeds\n"
            "Positive means row checkpoint beats column checkpoint"
        ),
        xlabel=heatmap_xlabel,
        ylabel=heatmap_ylabel,
        annotate=annotate,
    )

    later_vs_earlier = mean_matrix.copy()
    for r_idx in range(len(sorted_iterations)):
        for c_idx in range(len(sorted_iterations)):
            if r_idx <= c_idx:
                later_vs_earlier[r_idx, c_idx] = np.nan
    plot_pairwise_heatmap(
        later_vs_earlier,
        checkpoint_labels,
        checkpoint_labels,
        output_path=run_dir / "head_to_head_later_vs_earlier.png",
        title=(
            "Later-vs-earlier checkpoint EV only\n"
            "Positive cells indicate higher exact EV for the later policy"
        ),
        xlabel=(
            "Earlier policy checkpoint (nodes touched)"
            if use_nodes
            else "Earlier checkpoint iteration"
        ),
        ylabel=(
            "Later policy checkpoint (nodes touched)"
            if use_nodes
            else "Later checkpoint iteration"
        ),
        annotate=annotate,
    )

    plot_pairwise_heatmap(
        win_fraction_matrix,
        checkpoint_labels,
        checkpoint_labels,
        output_path=run_dir / "head_to_head_seed_win_fraction.png",
        title="Fraction of seeds where checkpoint A clearly beats checkpoint B",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Fraction of seeds where A beats B",
        xlabel=heatmap_xlabel,
        ylabel=heatmap_ylabel,
        annotate=annotate,
    )

    aggregate_lookup = {int(row["checkpoint"]): row for row in aggregate_rows}
    iterations_for_plot = list(sorted_iterations)
    if use_nodes:
        x_for_plot = [
            aggregate_lookup[i].get("nodes_touched_mean", float("nan"))
            for i in iterations_for_plot
        ]
        temporal_xlabel = "Nodes Touched"
        strength_earlier_filename = "head_to_head_strength_vs_earlier_by_nodes.png"
        strength_previous_filename = "head_to_head_strength_vs_previous_by_nodes.png"
        exploitability_filename = "exploitability_by_nodes.png"
        average_value_filename = "average_policy_value_by_nodes.png"
    else:
        x_for_plot = iterations_for_plot
        temporal_xlabel = "Checkpoint iteration"
        strength_earlier_filename = "head_to_head_strength_vs_earlier.png"
        strength_previous_filename = "head_to_head_strength_vs_previous.png"
        exploitability_filename = "exploitability_by_checkpoint.png"
        average_value_filename = "average_policy_value_by_checkpoint.png"
    plot_strength_curve_with_errorbars(
        x_for_plot,
        [
            aggregate_lookup[i].get("mean_EV_vs_earlier_checkpoints_mean", float("nan"))
            for i in iterations_for_plot
        ],
        [
            aggregate_lookup[i].get("mean_EV_vs_earlier_checkpoints_sem", float("nan"))
            for i in iterations_for_plot
        ],
        output_path=run_dir / strength_earlier_filename,
        title="Does later training improve head-to-head performance?",
        xlabel=temporal_xlabel,
        ylabel="Mean EV vs earlier checkpoints",
    )
    plot_strength_curve_with_errorbars(
        x_for_plot,
        [
            aggregate_lookup[i].get("EV_vs_previous_checkpoint_mean", float("nan"))
            for i in iterations_for_plot
        ],
        [
            aggregate_lookup[i].get("EV_vs_previous_checkpoint_sem", float("nan"))
            for i in iterations_for_plot
        ],
        output_path=run_dir / strength_previous_filename,
        title="Adjacent-checkpoint improvement",
        xlabel=temporal_xlabel,
        ylabel="EV vs immediately previous checkpoint",
    )
    plot_strength_curve_with_errorbars(
        x_for_plot,
        [aggregate_lookup[i].get("exploitability_mean", float("nan")) for i in iterations_for_plot],
        [aggregate_lookup[i].get("exploitability_sem", float("nan")) for i in iterations_for_plot],
        output_path=run_dir / exploitability_filename,
        title="Checkpoint exploitability over training",
        xlabel=temporal_xlabel,
        ylabel="Exploitability",
        reference_line_label="Nash equilibrium target",
    )
    plot_strength_curve_with_errorbars(
        x_for_plot,
        [
            aggregate_lookup[i].get("average_policy_value_mean", float("nan"))
            for i in iterations_for_plot
        ],
        [
            aggregate_lookup[i].get("average_policy_value_sem", float("nan"))
            for i in iterations_for_plot
        ],
        output_path=run_dir / average_value_filename,
        title="Checkpoint average policy value over training",
        xlabel=temporal_xlabel,
        ylabel="Average policy value for player 0",
        reference_line_value=float(config.get("average_policy_value_target", -0.085606424078)),
        reference_line_label="Player 0 Nash value",
    )

    plot_scatter_annotated(
        [row["exploitability"] for row in strength_dict_rows],
        [row["mean_EV_vs_all_other_checkpoints"] for row in strength_dict_rows],
        [
            f"{row['seed']}:{int(row['checkpoint'])}"
            for row in strength_dict_rows
        ],
        output_path=run_dir / "strength_vs_exploitability.png",
        title="Equilibrium quality versus head-to-head strength",
        xlabel="Exploitability",
        ylabel="Mean EV vs all other checkpoints",
    )
    plot_scatter_annotated(
        [row["average_policy_value"] for row in strength_dict_rows],
        [row["mean_EV_vs_all_other_checkpoints"] for row in strength_dict_rows],
        [
            f"{row['seed']}:{int(row['checkpoint'])}"
            for row in strength_dict_rows
        ],
        output_path=run_dir / "strength_vs_average_policy_value.png",
        title="Average policy value versus head-to-head strength",
        xlabel="Average policy value for player 0",
        ylabel="Mean EV vs all other checkpoints",
        x_reference_line_value=float(
            config.get("average_policy_value_target", -0.085606424078)
        ),
        x_reference_line_label="Player 0 Nash value",
    )

    # ---------------- optional Monte Carlo ----------------
    mc_csv: Optional[Path] = None
    if bool(config.get("run_monte_carlo_validation", False)):
        mc_pairs = _select_mc_pairs(
            sorted_iterations, str(config.get("mc_pair_mode", "adjacent"))
        )
        mc_rows: List[dict] = []
        num_episodes = int(config.get("num_mc_episodes", 20_000))
        mc_seed = int(config.get("mc_seed", 12345))
        alternate_seats = bool(config.get("mc_alternate_seats", True))
        for seed, policies in policies_by_seed.items():
            for ckpt_a, ckpt_b in mc_pairs:
                if ckpt_a not in policies or ckpt_b not in policies:
                    continue
                pol_a = policies[ckpt_a]
                pol_b = policies[ckpt_b]
                mc = monte_carlo_head_to_head(
                    game,
                    pol_a,
                    pol_b,
                    num_episodes=num_episodes,
                    seed=mc_seed,
                    alternate_seats=alternate_seats,
                )
                row = dataclasses.asdict(mc)
                row.update(
                    {
                        "seed": str(seed),
                        "checkpoint_A": int(ckpt_a),
                        "checkpoint_B": int(ckpt_b),
                    }
                )
                mc_rows.append(row)
        if mc_rows:
            mc_csv = run_dir / "head_to_head_monte_carlo.csv"
            write_dict_rows_csv(mc_rows, mc_csv)

    return {
        "checkpoint_inventory": inventory_csv,
        "checkpoint_exploitability_metrics": metrics_csv,
        "head_to_head_pairwise": pairwise_csv,
        "head_to_head_mean_matrix": mean_matrix_csv,
        "head_to_head_seed_win_fraction_matrix": win_fraction_csv,
        "head_to_head_monotonicity_by_seed": monotonicity_csv,
        "head_to_head_strength_by_checkpoint": strength_csv,
        "head_to_head_strength_aggregate": aggregate_csv,
        "best_checkpoint_summary": best_csv,
        "monte_carlo": mc_csv,
        "policies_loaded": sum(len(v) for v in policies_by_seed.values()),
        "seeds_loaded": list(policies_by_seed.keys()),
    }
