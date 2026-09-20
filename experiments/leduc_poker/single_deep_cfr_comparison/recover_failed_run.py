"""Recover Experiment 28 analysis from final policies and SD-CFR archives.

The original failed run persisted every final conventional Deep CFR policy and
the complete per-iteration SD-CFR advantage-network archives, but it failed
before exporting its in-memory checkpoint table. This module therefore
recovers:

* exact final exploitability and policy value for conventional Deep CFR;
* exact uniform- and linear-weighted SD-CFR trajectories by CFR iteration;
* paired five-seed final comparisons and SD-CFR iteration AUCs.

It deliberately does not fabricate conventional Deep CFR temporal, node-count,
or wall-clock curves, because those observations were not saved.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/deep_cfr_poker_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/deep_cfr_poker_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python import policy as osp_policy  # noqa: E402
from open_spiel.python.algorithms import expected_game_score, exploitability  # noqa: E402

from deep_cfr_poker.experiment_utils import (  # noqa: E402
    game_value_player_0,
    json_safe,
    normalised_auc,
    summarise_numeric_fields,
    write_dict_rows_csv,
)
from deep_cfr_poker.plotting import set_chart_title  # noqa: E402
from deep_cfr_poker.sd_cfr import (  # noqa: E402
    SDCFRArchive,
    exact_average_policies_at_checkpoints,
)
from deep_cfr_poker.snapshots import LoadedPolicy  # noqa: E402


_LOGGER = logging.getLogger("deep_cfr_poker.experiment.single_deep_cfr.recovery")
_SEED_PATTERN = re.compile(r"seed_(\d+)_")
_WEIGHTINGS = ("uniform", "linear")


def _seed_from_path(path: Path) -> int:
    match = _SEED_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Cannot extract a seed from {path}")
    return int(match.group(1))


def _discover(input_dir: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    archives = {
        _seed_from_path(path): path
        for path in sorted(input_dir.rglob("seed_*_sd_cfr_archive.pt"))
    }
    snapshots = {
        _seed_from_path(path): path
        for path in sorted(input_dir.rglob("seed_*_deep_cfr_policy.pt"))
    }
    if not archives:
        raise FileNotFoundError(f"No SD-CFR archives found below {input_dir}")
    if set(archives) != set(snapshots):
        raise ValueError(
            "Archive/snapshot seed mismatch: "
            f"archives={sorted(archives)}, snapshots={sorted(snapshots)}"
        )
    return archives, snapshots


def _evaluate_policy(game, policy, value_target: float) -> dict:
    tabular = osp_policy.tabular_policy_from_callable(
        game, policy.action_probabilities
    )
    nash_conv = float(exploitability.nash_conv(game, tabular))
    value = float(
        expected_game_score.policy_value(
            game.new_initial_state(), [tabular] * game.num_players()
        )[0]
    )
    return {
        "nash_conv": nash_conv,
        "exploitability": nash_conv / 2.0,
        "policy_value": value,
        "policy_value_error": abs(value - value_target),
    }


def _checkpoint_schedule(
    archive: SDCFRArchive, snapshot_metadata: Mapping[str, object]
) -> list[int]:
    iterations = [
        int(entry.iteration) for entry in archive.entries_by_player[0]
    ]
    if not iterations:
        raise ValueError("The SD-CFR archive contains no iterations")
    solver_config = dict(snapshot_metadata.get("solver_config", {}))
    interval = int(solver_config.get("evaluation_interval", 25))
    if interval < 1:
        raise ValueError(f"Invalid evaluation interval {interval}")
    checkpoints = [iteration for iteration in iterations if iteration % interval == 0]
    if not checkpoints or checkpoints[-1] != iterations[-1]:
        checkpoints.append(iterations[-1])
    return checkpoints


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    temporary.replace(path)


def recover_seed(seed: int, archive_path: Path, snapshot_path: Path) -> dict:
    _LOGGER.info("Recovering seed %s", seed)
    archive = SDCFRArchive.load(archive_path)
    game = pyspiel.load_game(str(archive.game_name))
    deep_policy = LoadedPolicy(game, snapshot_path)
    metadata = deep_policy.metadata
    metadata_seed = metadata.get("seed")
    if metadata_seed is not None and int(metadata_seed) != int(seed):
        raise ValueError(
            f"Snapshot {snapshot_path} declares seed {metadata_seed}, expected {seed}"
        )
    config = dict(metadata.get("solver_config", {}))
    config.setdefault("game_name", archive.game_name)
    value_target = game_value_player_0(config)
    checkpoints = _checkpoint_schedule(archive, metadata)

    deep_metrics = _evaluate_policy(game, deep_policy, value_target)
    rows_by_checkpoint = {
        checkpoint: {"seed": int(seed), "iteration": int(checkpoint)}
        for checkpoint in checkpoints
    }
    final_sd_metrics = {}
    for weighting in _WEIGHTINGS:
        policies = exact_average_policies_at_checkpoints(
            game, archive, checkpoints, weighting=weighting
        )
        for checkpoint, policy in policies.items():
            metrics = _evaluate_policy(game, policy, value_target)
            rows_by_checkpoint[checkpoint].update(
                {
                    f"sd_cfr_{weighting}_exploitability": metrics["exploitability"],
                    f"sd_cfr_{weighting}_policy_value": metrics["policy_value"],
                    f"sd_cfr_{weighting}_policy_value_error": metrics[
                        "policy_value_error"
                    ],
                }
            )
            if checkpoint == checkpoints[-1]:
                final_sd_metrics[weighting] = metrics

    curves = [rows_by_checkpoint[checkpoint] for checkpoint in checkpoints]
    summary = {
        "seed": int(seed),
        "final_iteration": int(checkpoints[-1]),
        "num_archived_networks_per_player": int(
            len(archive.entries_by_player[0])
        ),
        "deep_cfr_final_exploitability": deep_metrics["exploitability"],
        "deep_cfr_final_policy_value": deep_metrics["policy_value"],
        "deep_cfr_final_policy_value_error": deep_metrics["policy_value_error"],
        "sd_cfr_archive": str(archive_path),
        "deep_cfr_policy_snapshot": str(snapshot_path),
    }
    iterations = [row["iteration"] for row in curves]
    for weighting in _WEIGHTINGS:
        values = [row[f"sd_cfr_{weighting}_exploitability"] for row in curves]
        final_metrics = final_sd_metrics[weighting]
        summary.update(
            {
                f"sd_cfr_{weighting}_final_exploitability": final_metrics[
                    "exploitability"
                ],
                f"sd_cfr_{weighting}_minus_deep_cfr_final_exploitability": (
                    final_metrics["exploitability"]
                    - deep_metrics["exploitability"]
                ),
                f"sd_cfr_{weighting}_best_exploitability": float(np.min(values)),
                f"sd_cfr_{weighting}_exploitability_auc_by_iteration": (
                    normalised_auc(iterations, values)
                ),
                f"sd_cfr_{weighting}_final_policy_value": final_metrics[
                    "policy_value"
                ],
                f"sd_cfr_{weighting}_final_policy_value_error": final_metrics[
                    "policy_value_error"
                ],
            }
        )
    return {"seed": int(seed), "summary": summary, "curves": curves}


def _mean_and_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    if values.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _plot_sd_trajectories(results: Sequence[dict], output_dir: Path) -> None:
    iterations = np.asarray(
        [[row["iteration"] for row in result["curves"]] for result in results],
        dtype=np.float64,
    )
    colours = {"uniform": "#F58518", "linear": "#54A24B"}
    labels = {
        "uniform": "SD-CFR (uniform, weighting-matched)",
        "linear": "SD-CFR (linear, canonical)",
    }
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    for weighting in _WEIGHTINGS:
        values = np.asarray(
            [
                [
                    row[f"sd_cfr_{weighting}_exploitability"]
                    for row in result["curves"]
                ]
                for result in results
            ],
            dtype=np.float64,
        )
        mean, se = _mean_and_se(values)
        mean_x = np.mean(iterations, axis=0)
        for seed_index, row in enumerate(values):
            ax.plot(
                iterations[seed_index], row, color=colours[weighting],
                alpha=0.16, linewidth=0.8,
            )
        ax.plot(mean_x, mean, color=colours[weighting], linewidth=2.2,
                label=labels[weighting])
        ax.fill_between(mean_x, mean - se, mean + se,
                        color=colours[weighting], alpha=0.2)
    ax.set_xlabel("CFR iteration")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(
        ax,
        "Recovered exact SD-CFR exploitability",
        algorithm_variant="Uniform and canonical linear historical mixtures",
        game_name="leduc_poker",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "recovered_sd_cfr_exploitability_by_iteration.png", dpi=180)
    plt.close(fig)


def _plot_final_comparison(results: Sequence[dict], output_dir: Path) -> None:
    fields = (
        "deep_cfr_final_exploitability",
        "sd_cfr_uniform_final_exploitability",
        "sd_cfr_linear_final_exploitability",
    )
    labels = ("Deep CFR", "SD-CFR uniform", "SD-CFR linear")
    colours = ("#4C78A8", "#F58518", "#54A24B")
    values = [
        np.asarray([result["summary"][field] for result in results], dtype=np.float64)
        for field in fields
    ]
    means = [float(np.mean(series)) for series in values]
    ses = [
        float(np.std(series, ddof=1) / np.sqrt(len(series)))
        if len(series) > 1 else 0.0
        for series in values
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.bar(labels, means, yerr=ses, capsize=5, color=colours)
    for comparison_index in (1, 2):
        for seed_index in range(len(results)):
            ax.plot(
                [0, comparison_index],
                [values[0][seed_index], values[comparison_index][seed_index]],
                color="black", alpha=0.2, linewidth=0.7, marker="o", markersize=2.5,
            )
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(
        ax,
        "Recovered paired final-policy comparison",
        algorithm_variant="Deep CFR and exact SD-CFR historical mixtures",
        game_name="leduc_poker",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "recovered_final_exploitability_paired.png", dpi=180)
    plt.close(fig)


def export_recovery(results: Sequence[dict], input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [result["summary"] for result in results]
    curves = [row for result in results for row in result["curves"]]
    write_dict_rows_csv(summaries, output_dir / "recovered_seed_summary.csv")
    write_dict_rows_csv(curves, output_dir / "recovered_sd_cfr_curves.csv")
    _write_json(
        output_dir / "recovered_aggregate_summary.json",
        summarise_numeric_fields(summaries),
    )
    paired = {
        "difference_convention": "SD-CFR minus Deep CFR; negative favours SD-CFR",
        "comparisons": {},
    }
    for weighting in _WEIGHTINGS:
        field = f"sd_cfr_{weighting}_minus_deep_cfr_final_exploitability"
        values = np.asarray([row[field] for row in summaries], dtype=np.float64)
        paired["comparisons"][weighting] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "se": (
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0
            ),
            "per_seed": {
                str(row["seed"]): float(row[field]) for row in summaries
            },
        }
    _write_json(output_dir / "recovered_paired_difference_summary.json", paired)
    _write_json(
        output_dir / "recovery_manifest.json",
        {
            "status": "recovered_from_saved_final_policies_and_sd_cfr_archives",
            "input_dir": str(input_dir.resolve()),
            "seeds": [int(result["seed"]) for result in results],
            "recovered": [
                "conventional Deep CFR final exact metrics",
                "uniform SD-CFR exact metrics by evaluation iteration",
                "linear SD-CFR exact metrics by evaluation iteration",
                "paired final comparisons",
            ],
            "not_recoverable": [
                "conventional Deep CFR temporal exploitability",
                "nodes touched at evaluation checkpoints",
                "training wall-clock time at evaluation checkpoints",
                "AUC comparisons by nodes or training time",
            ],
        },
    )
    _plot_sd_trajectories(results, output_dir)
    _plot_final_comparison(results, output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate seeds even when a durable recovered result exists.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    archives, snapshots = _discover(args.input_dir)
    seed_dir = args.output_dir / "seed_results"
    results = []
    for seed in sorted(archives):
        result_path = seed_dir / f"seed_{seed}_recovered_result.json"
        if result_path.exists() and not args.overwrite:
            _LOGGER.info("Loading existing recovered result for seed %s", seed)
            with open(result_path, "r", encoding="utf-8") as handle:
                result = json.load(handle)
        else:
            result = recover_seed(seed, archives[seed], snapshots[seed])
            _write_json(result_path, result)
        results.append(result)
        # Keep a usable partial table if recovery is interrupted between seeds.
        write_dict_rows_csv(
            [item["summary"] for item in results],
            args.output_dir / "recovered_seed_summary.partial.csv",
        )
    export_recovery(results, args.input_dir, args.output_dir)
    _LOGGER.info("Recovered analysis written to %s", args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
