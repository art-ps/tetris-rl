import pytest
import torch
from torch import nn

from tetris_rl.models import QNetwork


def test_q_network_returns_one_value_per_candidate():
    network = QNetwork(input_dim=23)

    output = network(torch.zeros((8, 23), dtype=torch.float32))

    assert output.shape == (8, 1)


def test_q_network_matches_planned_architecture():
    network = QNetwork(input_dim=23)
    linear_layers = [layer for layer in network.network if isinstance(layer, nn.Linear)]

    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (23, 128),
        (128, 128),
        (128, 64),
        (64, 1),
    ]


def test_q_network_rejects_invalid_input_dimension():
    with pytest.raises(ValueError, match="input_dim"):
        QNetwork(input_dim=0)
