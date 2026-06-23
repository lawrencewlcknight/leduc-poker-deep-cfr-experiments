"""Tests for optional MLP architecture variants."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from deep_cfr_poker.networks import build_network, build_shared_trunk_player_heads


@pytest.mark.parametrize(
    "network_type",
    [
        "mlp",
        "dropout_mlp_p05",
        "dropout_mlp_p10",
        "dropout_mlp_p20",
        "residual_mlp",
        "layer_norm_mlp",
        "residual_layer_norm_mlp",
        "centered_advantage_mlp",
        "dueling_mlp",
    ],
)
def test_network_variants_forward_and_reset(network_type):
    net = build_network(
        network_type,
        input_size=6,
        hidden_sizes=(8, 8, 8),
        output_size=3,
    )
    x = torch.randn(4, 6)
    y = net(x)
    assert y.shape == (4, 3)
    net.reset()
    y_after_reset = net(x)
    assert y_after_reset.shape == (4, 3)


@pytest.mark.parametrize("network_type", ["centered_advantage_mlp", "dueling_mlp"])
def test_factorised_advantage_heads_center_action_terms(network_type):
    net = build_network(
        network_type,
        input_size=6,
        hidden_sizes=(8, 8),
        output_size=3,
    )
    x = torch.randn(4, 6)
    y = net(x)
    centred = y - y.mean(dim=-1, keepdim=True)

    if network_type == "centered_advantage_mlp":
        assert torch.allclose(y.mean(dim=-1), torch.zeros(4), atol=1e-6)
    else:
        assert centred.shape == (4, 3)


def test_dropout_mlp_disables_dropout_in_eval_mode():
    net = build_network(
        "dropout_mlp_p20",
        input_size=6,
        hidden_sizes=(8, 8),
        output_size=3,
    )
    x = torch.randn(4, 6)

    net.eval()
    y_eval_1 = net(x)
    y_eval_2 = net(x)
    assert torch.allclose(y_eval_1, y_eval_2)

    net.train()
    y_train_1 = net(x)
    y_train_2 = net(x)
    assert not torch.allclose(y_train_1, y_train_2)


def test_unknown_network_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown network_type"):
        build_network("not_a_network", 6, (8, 8), 3)


def test_shared_trunk_player_heads_share_trunk_and_reset():
    heads = build_shared_trunk_player_heads(
        input_size=6,
        hidden_sizes=(8, 8),
        output_size=3,
        num_players=2,
    )

    assert len(heads) == 2
    assert heads[0].trunk is heads[1].trunk

    x = torch.randn(4, 6)
    assert heads[0](x).shape == (4, 3)
    assert heads[1](x).shape == (4, 3)

    heads[0].trunk.reset()
    heads[0].reset_head()
    heads[1].reset_head()
    assert heads[0](x).shape == (4, 3)
