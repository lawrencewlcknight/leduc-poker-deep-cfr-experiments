"""Correctness tests for Single Deep CFR archiving and evaluation."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from open_spiel.python import policy as osp_policy
from open_spiel.python.algorithms import expected_game_score, exploitability

from deep_cfr_poker.sd_cfr import (
    HistoricalSDCFRPolicy,
    SDCFRArchive,
    SampledSDCFRPolicy,
    build_information_set_catalog,
    exact_average_policies_at_checkpoints,
)
from deep_cfr_poker.seeding import set_seed
from deep_cfr_poker.solver import DeepCFRSolver


def _solver(game, *, iterations=3):
    return DeepCFRSolver(
        game,
        policy_network_layers=(8, 8),
        advantage_network_layers=(8, 8),
        num_iterations=iterations,
        num_traversals=3,
        learning_rate=1e-3,
        batch_size_advantage=2,
        batch_size_strategy=2,
        memory_capacity=256,
        policy_network_train_steps=1,
        policy_network_train_every=1,
        evaluation_interval=1,
        advantage_network_train_steps=1,
        reinitialize_advantage_networks=False,
        compute_exploitability=False,
    )


def _train_archive(game, *, iterations=3):
    solver = _solver(game, iterations=iterations)
    archive = SDCFRArchive.from_solver(solver, game_name="leduc_poker")
    solver.solve(post_player_update_callback=archive.capture_from_solver)
    return solver, archive


def test_information_set_catalog_encodes_perfect_recall_sequences(leduc_game):
    catalog = build_information_set_catalog(leduc_game)
    assert catalog
    assert {record.player for record in catalog.values()} == {0, 1}
    assert any(record.own_action_sequence for record in catalog.values())
    assert all(record.legal_actions for record in catalog.values())


def test_archive_captures_every_player_update_and_round_trips(leduc_game, tmp_path):
    _solver_instance, archive = _train_archive(leduc_game, iterations=3)
    archive.validate()

    for player in (0, 1):
        assert [entry.iteration for entry in archive.entries_by_player[player]] == [
            1,
            2,
            3,
        ]

    path = archive.save(tmp_path / "sd_cfr_archive.pt")
    loaded = SDCFRArchive.load(path)
    assert loaded.advantage_network_type == archive.advantage_network_type
    assert loaded.advantage_network_layers == (8, 8)
    for player in (0, 1):
        assert [entry.iteration for entry in loaded.entries_by_player[player]] == [
            1,
            2,
            3,
        ]


def test_archiving_does_not_perturb_the_training_trajectory(leduc_game):
    set_seed(91)
    control = _solver(leduc_game, iterations=2)
    control.solve()
    control_policy = {
        key: value.detach().clone()
        for key, value in control._policy_network.state_dict().items()
    }
    control_advantages = [
        {key: value.detach().clone() for key, value in network.state_dict().items()}
        for network in control._advantage_networks
    ]

    set_seed(91)
    instrumented = _solver(leduc_game, iterations=2)
    archive = SDCFRArchive.from_solver(instrumented, game_name="leduc_poker")
    instrumented.solve(post_player_update_callback=archive.capture_from_solver)

    for key, expected in control_policy.items():
        torch.testing.assert_close(
            instrumented._policy_network.state_dict()[key], expected
        )
    for player, expected_state in enumerate(control_advantages):
        for key, expected in expected_state.items():
            torch.testing.assert_close(
                instrumented._advantage_networks[player].state_dict()[key], expected
            )


def test_exact_sd_cfr_prefix_policies_are_valid_and_exploitable(leduc_game):
    _solver_instance, archive = _train_archive(leduc_game, iterations=3)
    policies = exact_average_policies_at_checkpoints(
        leduc_game, archive, [1, 3], weighting="linear"
    )

    assert set(policies) == {1, 3}
    for policy in policies.values():
        tabular = osp_policy.tabular_policy_from_callable(
            leduc_game, policy.action_probabilities
        )
        assert np.isfinite(exploitability.nash_conv(leduc_game, tabular))
        np.testing.assert_allclose(
            np.sum(tabular.action_probability_array, axis=1),
            1.0,
            atol=1e-6,
        )


def test_sampled_sd_cfr_policy_holds_models_fixed_within_episode(leduc_game):
    _solver_instance, archive = _train_archive(leduc_game, iterations=3)
    sampled = SampledSDCFRPolicy(leduc_game, archive, seed=7)
    selected = dict(sampled.selected_iterations)

    state = leduc_game.new_initial_state()
    while state.is_chance_node():
        state = state.child(state.chance_outcomes()[0][0])
    first = sampled.action_probabilities(state)
    second = sampled.action_probabilities(state)

    assert sampled.selected_iterations == selected
    assert first == second
    assert sum(first.values()) == pytest.approx(1.0, abs=1e-6)
    sampled.resample_episode()
    assert set(sampled.selected_iterations) == {0, 1}


def test_exact_average_is_realisation_equivalent_to_historical_mixture(leduc_game):
    _solver_instance, archive = _train_archive(leduc_game, iterations=3)
    exact = exact_average_policies_at_checkpoints(
        leduc_game, archive, [3], weighting="linear"
    )[3]
    uniform = osp_policy.UniformRandomPolicy(leduc_game)
    weights = np.asarray([1.0, 2.0, 3.0]) / 6.0

    exact_player_0_value = expected_game_score.policy_value(
        leduc_game.new_initial_state(), [exact, uniform]
    )[0]
    mixture_player_0_value = 0.0
    for weight, iteration in zip(weights, (1, 2, 3)):
        historical = HistoricalSDCFRPolicy(
            leduc_game, archive, {0: iteration}
        )
        value = expected_game_score.policy_value(
            leduc_game.new_initial_state(), [historical, uniform]
        )[0]
        mixture_player_0_value += float(weight) * float(value)

    assert exact_player_0_value == pytest.approx(
        mixture_player_0_value, abs=1e-6
    )

    exact_player_1_value = expected_game_score.policy_value(
        leduc_game.new_initial_state(), [uniform, exact]
    )[1]
    mixture_player_1_value = 0.0
    for weight, iteration in zip(weights, (1, 2, 3)):
        historical = HistoricalSDCFRPolicy(
            leduc_game, archive, {1: iteration}
        )
        value = expected_game_score.policy_value(
            leduc_game.new_initial_state(), [uniform, historical]
        )[1]
        mixture_player_1_value += float(weight) * float(value)

    assert exact_player_1_value == pytest.approx(
        mixture_player_1_value, abs=1e-6
    )
