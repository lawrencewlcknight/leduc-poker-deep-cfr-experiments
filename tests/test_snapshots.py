"""Tests for the policy snapshot module."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from deep_cfr_poker.snapshots import (
    LoadedPolicy,
    POLICY_SNAPSHOT_TYPE,
    POLICY_SNAPSHOT_VERSION,
    discover_snapshots,
    package_full_checkpoint_filename,
    package_snapshot_filename,
    parse_snapshot_filename,
    save_policy_snapshot,
)


def test_filename_helpers_produce_expected_strings():
    assert package_snapshot_filename(1234, 100) == "seed_1234_iter_100_snapshot.pt"
    assert package_full_checkpoint_filename(1234, 100) == "seed_1234_iter_100_full.pt"


def test_parse_snapshot_filename_recognises_package_format(tmp_path):
    path = tmp_path / "seed_1234_iter_100_snapshot.pt"
    path.write_bytes(b"")
    parsed = parse_snapshot_filename(path)
    assert parsed is not None
    assert parsed.seed == "1234"
    assert parsed.iteration == 100
    assert parsed.kind == "policy_snapshot"


def test_parse_snapshot_filename_recognises_legacy_notebook_format(tmp_path):
    path = tmp_path / "leduc_poker_deep_cfr_seed_42_policy_snapshot_500_iters.pt"
    path.write_bytes(b"")
    parsed = parse_snapshot_filename(path)
    assert parsed is not None
    assert parsed.seed == "42"
    assert parsed.iteration == 500
    assert parsed.kind == "policy_snapshot"


def test_parse_snapshot_filename_recognises_full_checkpoints(tmp_path):
    new_style = tmp_path / "seed_42_iter_500_full.pt"
    legacy = tmp_path / "leduc_poker_deep_cfr_seed_42_ckpt_500_iters.pt"
    new_style.write_bytes(b"")
    legacy.write_bytes(b"")
    assert parse_snapshot_filename(new_style).kind == "full_checkpoint"
    assert parse_snapshot_filename(legacy).kind == "full_checkpoint"


def test_parse_snapshot_filename_returns_none_for_unrelated_files(tmp_path):
    path = tmp_path / "some_other_file.pt"
    path.write_bytes(b"")
    assert parse_snapshot_filename(path) is None


def test_discover_snapshots_walks_directory(tmp_path):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "seed_1_iter_100_snapshot.pt").write_bytes(b"")
    (tmp_path / "snapshots" / "seed_2_iter_300_snapshot.pt").write_bytes(b"")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "seed_1_iter_100_full.pt").write_bytes(b"")
    found = discover_snapshots(tmp_path)
    iterations = sorted({rec.iteration for rec in found})
    assert 100 in iterations
    assert 300 in iterations


@pytest.fixture
def trained_solver(leduc_game):
    from deep_cfr_poker.solver import DeepCFRSolver

    solver = DeepCFRSolver(
        leduc_game,
        policy_network_layers=(8, 8),
        advantage_network_layers=(8, 8),
        num_iterations=2,
        num_traversals=4,
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=256,
        policy_network_train_steps=1,
        policy_network_train_every=1,
        advantage_network_train_steps=1,
        reinitialize_advantage_networks=False,
        compute_exploitability=False,
    )
    solver.solve()
    return solver


def test_save_policy_snapshot_round_trip(tmp_path, leduc_game, trained_solver):
    snapshot_path = tmp_path / "seed_42_iter_100_snapshot.pt"
    save_policy_snapshot(
        trained_solver,
        snapshot_path,
        seed=42,
        target_iteration=100,
        stage_label="test",
        experiment_name="snapshot_test",
        game_name="leduc_poker",
        solver_config={"num_traversals": 4},
    )

    loaded = LoadedPolicy(leduc_game, snapshot_path)
    meta = loaded.metadata
    assert meta["version"] == POLICY_SNAPSHOT_VERSION
    assert meta["type"] == POLICY_SNAPSHOT_TYPE
    assert meta["seed"] == 42
    assert meta["checkpoint_iteration"] == 100

    state = leduc_game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    a = trained_solver.action_probabilities(state)
    b = loaded.action_probabilities(state)
    for action, prob in a.items():
        assert b[action] == pytest.approx(prob, abs=1e-5)


def test_loaded_policy_handles_legacy_format(tmp_path, leduc_game, trained_solver):
    """Legacy notebook snapshots have version=1 and may omit solver_config."""
    snapshot_path = tmp_path / "leduc_poker_deep_cfr_seed_7_policy_snapshot_50_iters.pt"
    legacy_payload = {
        "version": 1,
        "type": POLICY_SNAPSHOT_TYPE,
        "seed": 7,
        "checkpoint_iteration": 50,
        "internal_iteration": 50,
        "stage_label": "legacy",
        "game": "leduc_poker",
        "policy_state_dict": trained_solver._policy_network.state_dict(),
        "policy_network_layers": (8, 8),
        "num_actions": trained_solver._num_actions,
        "embedding_size": trained_solver._embedding_size,
    }
    torch.save(legacy_payload, snapshot_path)

    loaded = LoadedPolicy(leduc_game, snapshot_path)
    assert loaded.metadata["version"] == 1
    assert loaded.metadata["seed"] == 7
    assert loaded.metadata["checkpoint_iteration"] == 50

    # Inference should still work.
    state = leduc_game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    probs = loaded.action_probabilities(state)
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
