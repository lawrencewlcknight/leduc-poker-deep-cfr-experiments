"""Cloud workers and aggregation for Deep CFR Experiment 29."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .config import EXPERIMENT_ID, EXPERIMENT_NAME, PRODUCTION_SEEDS, SMOKE_SEEDS
from .run import (
    aggregate_results,
    read_csv,
    read_json,
    run_evaluation_seed,
    run_training_seed,
    sha256,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be non-empty and unique")
    return seeds


def task_name(task_index: int, seeds: Sequence[int]) -> str:
    if task_index < 0 or task_index >= len(seeds):
        raise ValueError(f"Task index {task_index} is outside the seed schedule")
    return f"task_{task_index:03d}_seed_{int(seeds[task_index])}"


def _normalise_seeds(args) -> tuple[int, ...]:
    if args.smoke and tuple(args.seeds) == tuple(PRODUCTION_SEEDS):
        return SMOKE_SEEDS
    return tuple(args.seeds)


def _training_is_valid(task_dir: Path, seed: int, smoke: bool, commit: str) -> bool:
    wrapper_path = task_dir / "training_worker_result.json"
    result_path = task_dir / f"seed_{seed}" / "training_result.json"
    if not wrapper_path.is_file() or not result_path.is_file():
        return False
    try:
        wrapper = read_json(wrapper_path)
        result = read_json(result_path)
        archive_path = task_dir / str(result["archive_path"])
        manifest = read_csv(task_dir / str(result["manifest_path"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        wrapper.get("status") == "complete"
        and wrapper.get("repository_commit") == commit
        and bool(wrapper.get("smoke")) == bool(smoke)
        and int(wrapper.get("seed", -1)) == int(seed)
        and result.get("status") == "complete"
        and int(result.get("experiment_id", -1)) == EXPERIMENT_ID
        and archive_path.is_file()
        and sha256(archive_path) == result.get("archive_sha256")
        and bool(manifest)
        and (task_dir / f"seed_{seed}" / "TRAINING_SUCCESS.json").is_file()
    )


def run_training_worker(
    *, task_index: int, seeds: Sequence[int], output_root: Path, smoke: bool, resume: bool
) -> dict:
    seed = int(seeds[task_index])
    name = task_name(task_index, seeds)
    task_dir = Path(output_root) / "training" / name
    task_dir.mkdir(parents=True, exist_ok=True)
    commit = repository_commit()
    if resume and _training_is_valid(task_dir, seed, smoke, commit):
        return read_json(task_dir / "training_worker_result.json")
    result = run_training_seed(seed=seed, output_dir=task_dir, smoke=smoke)
    wrapper = {
        "status": "complete",
        "stage": "training",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "task_index": int(task_index),
        "task_name": name,
        "seed": seed,
        "smoke": bool(smoke),
        "repository_commit": commit,
        "training_result": result,
    }
    write_json(task_dir / "training_worker_result.json", wrapper)
    return wrapper


def _evaluation_is_valid(task_dir: Path, seed: int, smoke: bool, commit: str) -> bool:
    wrapper_path = task_dir / "evaluation_worker_result.json"
    result_path = task_dir / f"seed_{seed}" / "evaluation_result.json"
    if not wrapper_path.is_file() or not result_path.is_file():
        return False
    try:
        wrapper = read_json(wrapper_path)
        result = read_json(result_path)
        rows = read_csv(task_dir / str(result["metrics_path"]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return (
        wrapper.get("status") == "complete"
        and wrapper.get("repository_commit") == commit
        and bool(wrapper.get("smoke")) == bool(smoke)
        and int(wrapper.get("seed", -1)) == int(seed)
        and result.get("status") == "complete"
        and int(result.get("experiment_id", -1)) == EXPERIMENT_ID
        and len(rows) == int(result.get("num_metric_rows", -1))
        and sum(str(row.get("is_final_endpoint", "False")).lower() == "true" for row in rows) == 1
        and (task_dir / f"seed_{seed}" / "EVALUATION_SUCCESS.json").is_file()
    )


def run_evaluation_worker(
    *,
    task_index: int,
    seeds: Sequence[int],
    training_root: Path,
    output_root: Path,
    smoke: bool,
    resume: bool,
) -> dict:
    seed = int(seeds[task_index])
    name = task_name(task_index, seeds)
    training_task_dir = Path(training_root) / "training" / name
    task_dir = Path(output_root) / "evaluation" / name
    task_dir.mkdir(parents=True, exist_ok=True)
    commit = repository_commit()
    training_wrapper = read_json(training_task_dir / "training_worker_result.json")
    if (
        training_wrapper.get("status") != "complete"
        or int(training_wrapper.get("seed", -1)) != seed
        or bool(training_wrapper.get("smoke")) != bool(smoke)
        or training_wrapper.get("repository_commit") != commit
    ):
        raise RuntimeError(f"Training artifacts for seed {seed} do not match this commit")
    if resume and _evaluation_is_valid(task_dir, seed, smoke, commit):
        return read_json(task_dir / "evaluation_worker_result.json")
    result = run_evaluation_seed(
        seed=seed,
        training_dir=training_task_dir,
        output_dir=task_dir,
        smoke=smoke,
    )
    wrapper = {
        "status": "complete",
        "stage": "evaluation",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "task_index": int(task_index),
        "task_name": name,
        "seed": seed,
        "smoke": bool(smoke),
        "repository_commit": commit,
        "evaluation_result": result,
    }
    write_json(task_dir / "evaluation_worker_result.json", wrapper)
    return wrapper


def aggregate_workers(
    *, evaluation_root: Path, seeds: Sequence[int], output_dir: Path, smoke: bool
) -> dict:
    metric_rows: list[dict] = []
    commits = set()
    for index, seed in enumerate(seeds):
        name = task_name(index, seeds)
        task_dir = Path(evaluation_root) / "evaluation" / name
        wrapper = read_json(task_dir / "evaluation_worker_result.json")
        if (
            wrapper.get("status") != "complete"
            or int(wrapper.get("seed", -1)) != int(seed)
            or bool(wrapper.get("smoke")) != bool(smoke)
        ):
            raise ValueError(f"Invalid evaluation worker for seed {seed}")
        commits.add(str(wrapper["repository_commit"]))
        result = read_json(task_dir / f"seed_{seed}" / "evaluation_result.json")
        metric_rows.extend(read_csv(task_dir / str(result["metrics_path"])))
    if len(commits) != 1:
        raise ValueError(f"Evaluation workers used different commits: {sorted(commits)}")
    result = aggregate_results(
        metric_rows=metric_rows,
        output_dir=Path(output_dir),
        expected_seed_count=len(seeds),
    )
    result.update(
        {
            "smoke": bool(smoke),
            "repository_commit": next(iter(commits)),
            "execution_contract": "one isolated training and evaluation task per seed",
        }
    )
    write_json(Path(output_dir) / "aggregate_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--task-index", type=int, required=True)
        sub.add_argument("--seeds", type=parse_seeds, default=PRODUCTION_SEEDS)
        sub.add_argument("--output-root", type=Path, required=True)
        sub.add_argument("--smoke", action="store_true")
        sub.add_argument("--no-resume", dest="resume", action="store_false")
        sub.set_defaults(resume=True)
        if command == "evaluate":
            sub.add_argument("--training-root", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--evaluation-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    aggregate.add_argument("--seeds", type=parse_seeds, default=PRODUCTION_SEEDS)
    aggregate.add_argument("--smoke", action="store_true")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(resume=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "smoke":
        run_training_worker(
            task_index=0,
            seeds=SMOKE_SEEDS,
            output_root=args.output_root,
            smoke=True,
            resume=args.resume,
        )
        run_evaluation_worker(
            task_index=0,
            seeds=SMOKE_SEEDS,
            training_root=args.output_root,
            output_root=args.output_root,
            smoke=True,
            resume=args.resume,
        )
        aggregate_workers(
            evaluation_root=args.output_root,
            seeds=SMOKE_SEEDS,
            output_dir=args.output_root / "analysis",
            smoke=True,
        )
        return
    seeds = _normalise_seeds(args)
    if args.command == "train":
        run_training_worker(
            task_index=args.task_index,
            seeds=seeds,
            output_root=args.output_root,
            smoke=args.smoke,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        run_evaluation_worker(
            task_index=args.task_index,
            seeds=seeds,
            training_root=args.training_root,
            output_root=args.output_root,
            smoke=args.smoke,
            resume=args.resume,
        )
    else:
        aggregate_workers(
            evaluation_root=args.evaluation_root,
            seeds=seeds,
            output_dir=args.output_dir,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()

