"""Tests for the Deep CFR solver core operations.

These tests exercise the parts of the solver that we can check without running
a full multi-iteration training pass:

* regret-matching: zero, single-positive, all-positive cases
* legal-action mass invariant in :meth:`action_probabilities`
* short smoke test of :meth:`solve` that asserts the returned dataclass shape
  and that exploitability is computed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deep_cfr_poker.solver import DeepCFRSolver, SolveResult


def _build_solver(leduc_game, **overrides):
    defaults = dict(
        policy_network_layers=(8, 8),
        advantage_network_layers=(8, 8),
        num_iterations=2,
        num_traversals=4,
        learning_rate=1e-3,
        batch_size_advantage=None,
        batch_size_strategy=None,
        memory_capacity=1024,
        policy_network_train_steps=1,
        policy_network_train_every=1,
        advantage_network_train_steps=1,
        reinitialize_advantage_networks=False,
        compute_exploitability=False,
    )
    defaults.update(overrides)
    return DeepCFRSolver(leduc_game, **defaults)


def test_regret_matching_uniform_fallback_when_all_advantages_zero(monkeypatch, leduc_game):
    solver = _build_solver(leduc_game)
    state = leduc_game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    legal = state.legal_actions()
    assert len(legal) > 1

    # Force the advantage network to return all zeros for this state.
    def all_zeros(_tensor):
        return torch.zeros((1, solver._num_actions))

    monkeypatch.setattr(
        solver._advantage_networks[state.current_player()],
        "forward",
        all_zeros,
    )

    _, matched = solver._sample_action_from_advantage(state, state.current_player())
    np.testing.assert_allclose(
        [matched[a] for a in legal],
        [1.0 / len(legal)] * len(legal),
    )
    # Illegal actions get zero mass.
    illegal = [a for a in range(solver._num_actions) if a not in legal]
    for a in illegal:
        assert matched[a] == 0.0


def test_regret_matching_single_positive_advantage_concentrates_mass(monkeypatch, leduc_game):
    solver = _build_solver(leduc_game)
    state = leduc_game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    legal = state.legal_actions()

    target_action = legal[0]
    forced = torch.full((1, solver._num_actions), -1.0)
    forced[0, target_action] = 5.0

    def forced_forward(_tensor):
        return forced

    monkeypatch.setattr(
        solver._advantage_networks[state.current_player()],
        "forward",
        forced_forward,
    )

    _, matched = solver._sample_action_from_advantage(state, state.current_player())
    assert matched[target_action] == pytest.approx(1.0)
    for action in legal:
        if action != target_action:
            assert matched[action] == pytest.approx(0.0)


def test_action_probabilities_sum_to_one_over_legal_actions(leduc_game):
    solver = _build_solver(leduc_game)
    state = leduc_game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    probs = solver.action_probabilities(state)
    assert set(probs.keys()) == set(state.legal_actions())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


def test_solve_returns_dataclass_and_runs_end_to_end(leduc_game):
    solver = _build_solver(
        leduc_game,
        num_iterations=2,
        num_traversals=4,
        compute_exploitability=True,
    )
    result = solver.solve()

    assert isinstance(result, SolveResult)
    assert isinstance(result.policy_network, torch.nn.Module)
    assert len(result.policy_losses) >= 1
    assert len(result.nash_conv) >= 1
    assert len(result.nodes_touched) >= 1
    assert len(result.average_policy_value) >= 1
    # Diagnostic series have one entry per checkpoint.
    expected_len = len(result.policy_losses)
    for key in (
        "iteration",
        "policy_loss",
        "policy_grad_norm",
        "advantage_target_variance",
        "legal_action_mass_mean",
    ):
        assert len(result.diagnostics[key]) == expected_len


def test_final_only_policy_training_marks_intermediate_metrics_missing(leduc_game):
    solver = _build_solver(
        leduc_game,
        num_iterations=3,
        evaluation_interval=1,
        policy_training_mode="final_only",
        final_policy_network_train_steps=1,
        compute_exploitability=True,
    )
    result = solver.solve()

    assert len(result.nash_conv) == 3
    assert np.isnan(result.nash_conv[0])
    assert np.isnan(result.average_policy_value[0])
    assert np.isfinite(result.nash_conv[-1])
    assert result.diagnostics["policy_training_events"] == [0, 0, 1]
    assert result.diagnostics["policy_gradient_steps"] == [0, 0, 1]
    assert result.diagnostics["policy_network_has_been_trained"] == [
        False,
        False,
        True,
    ]
    assert result.diagnostics["trained_policy_this_iteration"] == [
        False,
        False,
        True,
    ]


def test_learning_rate_schedule_is_recorded_at_checkpoints(leduc_game):
    solver = _build_solver(
        leduc_game,
        num_iterations=4,
        evaluation_interval=1,
        learning_rate=0.01,
        learning_rate_schedule="cosine_decay",
        learning_rate_end=0.001,
        compute_exploitability=False,
    )
    result = solver.solve()

    learning_rates = result.diagnostics["learning_rate"]
    assert len(learning_rates) == 4
    assert learning_rates[0] > learning_rates[-1]
    assert learning_rates[-1] == pytest.approx(0.001)


def test_target_processing_records_processed_target_diagnostics(leduc_game):
    solver = _build_solver(
        leduc_game,
        num_iterations=2,
        evaluation_interval=1,
        target_processing="standardize_clip",
        target_clip_value=1.0,
        compute_exploitability=False,
    )
    result = solver.solve()

    assert "processed_advantage_target_variance_player_0" in result.diagnostics
    assert "target_clip_fraction_player_0" in result.diagnostics
    assert len(result.diagnostics["processed_advantage_target_variance_player_0"]) == 2
    assert all(
        0.0 <= value <= 1.0
        for value in result.diagnostics["target_clip_fraction_player_0"]
    )


def test_priority_replay_and_uniform_average_weighting_record_diagnostics(leduc_game):
    solver = _build_solver(
        leduc_game,
        num_iterations=2,
        evaluation_interval=1,
        batch_size_advantage=2,
        advantage_replay_sampling="priority_abs_adv",
        average_strategy_weighting="uniform",
        compute_exploitability=False,
    )
    result = solver.solve()

    assert "advantage_priority_effective_sample_size" in result.diagnostics
    assert len(result.diagnostics["advantage_priority_effective_sample_size"]) == 2
    assert all(
        np.isfinite(value)
        for value in result.diagnostics["advantage_priority_effective_sample_size"]
    )


def test_extract_and_load_full_model_round_trip(leduc_game, tmp_path):
    solver = _build_solver(leduc_game)
    solver.solve()
    ckpt = solver.extract_full_model()

    fresh = _build_solver(leduc_game)
    fresh.load_full_model(ckpt)
    assert fresh._iteration == solver._iteration
    assert fresh._nodes_touched == solver._nodes_touched

    # Same logits on the same input.
    state = leduc_game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.legal_actions()[0])
    a = solver.action_probabilities(state)
    b = fresh.action_probabilities(state)
    for action, prob in a.items():
        assert b[action] == pytest.approx(prob)


def test_load_full_model_rejects_mismatched_meta(leduc_game):
    solver = _build_solver(leduc_game)
    ckpt = solver.extract_full_model()
    ckpt["meta"]["num_actions"] = solver._num_actions + 7
    with pytest.raises(ValueError, match="Checkpoint metadata mismatch"):
        solver.load_full_model(ckpt)
