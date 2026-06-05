"""Deep Q-Network candidate-placement agent."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tetris_rl.env.features import extract_features
from tetris_rl.env.tetris import Action
from tetris_rl.models import QNetwork


class DQNAgent:
    """Select candidate next states using epsilon-greedy Q-values."""

    def __init__(
        self,
        input_dim: int = 23,
        *,
        epsilon: float = 1.0,
        device: str | torch.device | None = None,
        seed: int | None = None,
    ):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.device = torch.device(device or "cpu")
        self._rng = random.Random(seed)
        self.q_network = QNetwork(input_dim).to(self.device)
        self.target_network = QNetwork(input_dim).to(self.device)
        self.update_target_network()
        self.target_network.eval()

    def candidate_features(
        self,
        valid_actions: Sequence[Action],
        next_piece: str,
    ) -> np.ndarray:
        """Extract one feature vector for every candidate next state."""
        if not valid_actions:
            raise ValueError("cannot evaluate an empty action sequence")
        features = [
            extract_features(
                action.board,
                next_piece,
                completed_lines=action.lines_cleared,
            )
            for action in valid_actions
        ]
        candidates = np.stack(features)
        if candidates.shape[1] != self.input_dim:
            raise ValueError(
                f"candidate feature size {candidates.shape[1]} does not match "
                f"input_dim {self.input_dim}"
            )
        return candidates

    def select_action(
        self,
        valid_actions: Sequence[Action],
        next_piece: str,
        *,
        epsilon: float | None = None,
    ) -> int:
        """Return a random or highest-Q candidate index."""
        if not valid_actions:
            raise ValueError("cannot select an action from an empty sequence")
        effective_epsilon = self.epsilon if epsilon is None else epsilon
        if not 0.0 <= effective_epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        if self._rng.random() < effective_epsilon:
            return self._rng.randrange(len(valid_actions))

        candidates = self.candidate_features(valid_actions, next_piece)
        tensor = torch.from_numpy(candidates).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(tensor).squeeze(-1)
        return int(torch.argmax(q_values).item())

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path: str | Path, **metadata: Any) -> None:
        """Save both networks and agent metadata."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_dim": self.input_dim,
                "epsilon": self.epsilon,
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "metadata": metadata,
            },
            checkpoint_path,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load a checkpoint and return its optional metadata."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint["input_dim"] != self.input_dim:
            raise ValueError(
                f"checkpoint input_dim {checkpoint['input_dim']} does not match "
                f"agent input_dim {self.input_dim}"
            )
        self.epsilon = float(checkpoint["epsilon"])
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        return dict(checkpoint.get("metadata", {}))
