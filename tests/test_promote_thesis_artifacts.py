"""Tests for promoting lightweight thesis artifacts from experiment outputs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "promote_thesis_artifacts.py"
SPEC = importlib.util.spec_from_file_location("promote_thesis_artifacts", SCRIPT_PATH)
assert SPEC is not None
promote_thesis_artifacts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = promote_thesis_artifacts
SPEC.loader.exec_module(promote_thesis_artifacts)


def _write(path: Path, text: str = "content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_selected_files_keeps_thesis_artifacts_and_excludes_heavy_outputs(tmp_path):
    run_dir = tmp_path / "outputs" / "cloud" / "exp1" / "run_20260517_120000"
    _write(
        run_dir / "experiment_metadata.json",
        json.dumps({"experiment_config": {"experiment_name": "exp1"}}),
    )
    _write(run_dir / "exploitability_by_iteration.png")
    _write(run_dir / "seed_summary.csv")
    _write(run_dir / "aggregate_summary.json")
    _write(run_dir / "ablation_curves.npz")
    _write(run_dir / "experiment.log")
    _write(run_dir / "checkpoints" / "seed_1234_final_model.pt")
    _write(run_dir / "snapshots" / "seed_1234_iter_10_snapshot.pt")
    _write(run_dir / "traces" / "screening_config_seed_1234.json")

    selected = {
        path.as_posix()
        for path in promote_thesis_artifacts.selected_files(run_dir)
    }

    assert selected == {
        "aggregate_summary.json",
        "experiment_metadata.json",
        "exploitability_by_iteration.png",
        "seed_summary.csv",
    }


def test_promote_discovers_run_dirs_and_copies_to_experiment_tree(tmp_path):
    job_dir = tmp_path / "cloud_outputs" / "job-1"
    run_dir = job_dir / "outputs" / "cloud" / "exp1" / "run_20260517_120000"
    _write(
        run_dir / "experiment_metadata.json",
        json.dumps(
            {"experiment_config": {"experiment_name": "leduc_poker_deep_cfr_validation"}}
        ),
    )
    _write(run_dir / "seed_summary.csv", "seed,final_exploitability\n1234,0.1\n")
    _write(run_dir / "plot.png", "png bytes")

    destination_root = tmp_path / "thesis_artifacts"
    run_dirs = promote_thesis_artifacts.discover_run_dirs([job_dir])
    artifact_sets = promote_thesis_artifacts.build_artifact_sets(
        run_dirs,
        destination_root=destination_root,
    )
    promote_thesis_artifacts.promote_artifacts(artifact_sets)

    destination_dir = (
        destination_root
        / "leduc_poker_deep_cfr_validation"
        / "run_20260517_120000"
    )
    assert (destination_dir / "seed_summary.csv").read_text(encoding="utf-8").startswith(
        "seed,"
    )
    assert (destination_dir / "plot.png").exists()
    assert (destination_dir / "promotion_manifest.json").exists()
