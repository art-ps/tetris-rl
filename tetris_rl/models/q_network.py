"""Q-value network for candidate next-state features."""

from __future__ import annotations

from torch import Tensor, nn


class QNetwork(nn.Module):
    """MLP that maps one candidate feature vector to a scalar Q-value."""

    def __init__(self, input_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be greater than zero")
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)
