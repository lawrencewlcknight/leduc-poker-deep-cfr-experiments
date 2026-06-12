#!/usr/bin/env python3
"""Promote lightweight experiment outputs into a tracked thesis artifact tree.

The cloud workflow intentionally downloads complete Batch outputs into a scratch
directory. This script copies only thesis-facing files from one or more run
directories into ``thesis_artifacts/`` so heavy files such as checkpoints,
snapshots, and NumPy arrays stay out of git.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_DESTINATION_ROOT = Path("thesis_artifacts")

DEFAULT_INCLUDE_PATTERNS = (
    "*.png",
    "*.csv",
    "aggregate_summary.json",
    "paired_difference_summary.json",
    "paired_aggregate_summary.json",
    "best_checkpoint_summary.json",
    "experiment_metadata.json",
)

DEFAULT_EXCLUDE_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.npz",
    "*.log",
    "failed_seeds.json",
    "checkpoints/*",
    "snapshots/*",
    "traces/*",
)


@dataclass(frozen=True)
class RunArtifactSet:
    source_run_dir: Path
    experiment_name: str
    destination_dir: Path
    files: tuple[Path, ...]


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _matches_any(path: Path, patterns: Sequence[str]) -> bool:
    path_text = path.as_posix()
    name = path.name
    return any(
        path.match(pattern) or name == pattern or path_text == pattern
        for pattern in patterns
    )


def _read_experiment_name(run_dir: Path) -> str:
    metadata_path = run_dir / "experiment_metadata.json"
    if not metadata_path.exists():
        return run_dir.name
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return run_dir.name
    config = metadata.get("experiment_config", {})
    experiment_name = config.get("experiment_name")
    if isinstance(experiment_name, str) and experiment_name:
        return experiment_name
    return run_dir.name


def discover_run_dirs(sources: Iterable[Path]) -> list[Path]:
    """Returns every directory under ``sources`` that looks like an experiment run."""
    run_dirs: set[Path] = set()
    for source in sources:
        source = source.expanduser()
        if source.is_file():
            raise ValueError(f"Source must be a directory, not a file: {source}")
        if not source.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")
        if (source / "experiment_metadata.json").exists():
            run_dirs.add(source.resolve())
            continue
        for metadata_path in source.rglob("experiment_metadata.json"):
            run_dirs.add(metadata_path.parent.resolve())
    return sorted(run_dirs)


def selected_files(
    run_dir: Path,
    *,
    include_patterns: Sequence[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> tuple[Path, ...]:
    """Returns relative file paths that should be promoted from ``run_dir``."""
    files: list[Path] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(run_dir)
        if _matches_any(relative_path, exclude_patterns):
            continue
        if _matches_any(relative_path, include_patterns):
            files.append(relative_path)
    return tuple(files)


def build_artifact_sets(
    run_dirs: Sequence[Path],
    *,
    destination_root: Path = DEFAULT_DESTINATION_ROOT,
    include_patterns: Sequence[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[RunArtifactSet]:
    artifact_sets = []
    for run_dir in run_dirs:
        experiment_name = _read_experiment_name(run_dir)
        destination_dir = destination_root / experiment_name / run_dir.name
        artifact_sets.append(
            RunArtifactSet(
                source_run_dir=run_dir,
                experiment_name=experiment_name,
                destination_dir=destination_dir,
                files=selected_files(
                    run_dir,
                    include_patterns=include_patterns,
                    exclude_patterns=exclude_patterns,
                ),
            )
        )
    return artifact_sets


def promote_artifacts(
    artifact_sets: Sequence[RunArtifactSet],
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    copied: list[dict] = []
    skipped: list[dict] = []

    for artifact_set in artifact_sets:
        for relative_path in artifact_set.files:
            source = artifact_set.source_run_dir / relative_path
            destination = artifact_set.destination_dir / relative_path
            record = {
                "source": str(source),
                "destination": str(destination),
            }
            if destination.exists() and not overwrite:
                skipped.append({**record, "reason": "destination_exists"})
                continue
            copied.append(record)
            if dry_run:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "overwrite": overwrite,
        "runs": [
            {
                "source_run_dir": str(artifact_set.source_run_dir),
                "experiment_name": artifact_set.experiment_name,
                "destination_dir": str(artifact_set.destination_dir),
                "selected_file_count": len(artifact_set.files),
                "selected_files": [path.as_posix() for path in artifact_set.files],
            }
            for artifact_set in artifact_sets
        ],
        "copied": copied,
        "skipped": skipped,
    }

    if not dry_run:
        for artifact_set in artifact_sets:
            artifact_set.destination_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = artifact_set.destination_dir / "promotion_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy thesis-facing experiment outputs from downloaded Batch outputs "
            "into thesis_artifacts/."
        )
    )
    parser.add_argument(
        "sources",
        nargs="+",
        type=Path,
        help=(
            "One or more run directories, downloaded job directories, or parent "
            "directories containing experiment_metadata.json files."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DESTINATION_ROOT,
        help="Destination root for promoted artifacts. Defaults to thesis_artifacts.",
    )
    parser.add_argument(
        "--include",
        default=None,
        help=(
            "Comma-separated glob/name patterns to include in addition to the "
            "default figure/table/provenance patterns."
        ),
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Comma-separated glob/name patterns to exclude in addition to the defaults.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing promoted files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    include_patterns = DEFAULT_INCLUDE_PATTERNS + _parse_csv(args.include)
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + _parse_csv(args.exclude)
    run_dirs = discover_run_dirs(args.sources)
    if not run_dirs:
        print("No experiment run directories found.")
        return 1

    artifact_sets = build_artifact_sets(
        run_dirs,
        destination_root=args.dest,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    manifest = promote_artifacts(
        artifact_sets,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
