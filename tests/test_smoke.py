"""End-to-end smoke test: a tiny multi-seed run must produce all expected outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyspiel")
pytest.importorskip("torch")

from deep_cfr_poker.experiment_utils import export_results, run_single_seed


SMOKE_CONFIG = {
    "experiment_name": "smoke",
    "game_name": "leduc_poker",
    "num_iterations": 3,
    "num_traversals": 4,
    "evaluation_interval": 1,
    "policy_network_layers": (8, 8),
    "advantage_network_layers": (8, 8),
    "learning_rate": 0.003,
    "batch_size_advantage": 2,
    "batch_size_strategy": 2,
    "memory_capacity": 256,
    "reinitialize_advantage_networks": False,
    "policy_network_train_steps": 1,
    "advantage_network_train_steps": 1,
    "compute_exploitability": True,
    "exploitability_threshold": 0.5,
}


@pytest.mark.smoke
def test_two_seed_smoke_run_writes_expected_artifacts(tmp_path):
    results = []
    for seed in (1234, 2025):
        results.append(run_single_seed(seed, SMOKE_CONFIG, export_dir=tmp_path))

    info = export_results(results, tmp_path, SMOKE_CONFIG, [1234, 2025])

    summary_csv = Path(info["summary_csv"])
    curve_csv = Path(info["curve_csv"])
    aggregate = Path(info["aggregate_summary"])
    metadata = Path(info["metadata"])
    npz_path = tmp_path / "multiseed_curves.npz"

    for path in (summary_csv, curve_csv, aggregate, metadata, npz_path):
        assert path.exists(), f"missing {path}"

    summary_text = summary_csv.read_text(encoding="utf-8").splitlines()
    assert summary_text[0].startswith("seed,")
    assert len(summary_text) == 1 + len(results)

    aggregate_payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert "final_exploitability" in aggregate_payload
    assert aggregate_payload["final_exploitability"]["n_finite"] == len(results)

    npz = np.load(npz_path)
    assert npz["exploitability"].shape[0] == len(results)
    assert npz["exploitability"].shape[1] >= 1
