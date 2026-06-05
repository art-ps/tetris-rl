"""Random baseline agent."""

from __future__ import annotations

import random
from collections.abc import Sequence

from tetris_rl.env.tetris import Action


class RandomAgent:
    """Choose uniformly among valid candidate placements."""

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def select_action(self, valid_actions: Sequence[Action]) -> int:
        if not valid_actions:
            raise ValueError("cannot select an action from an empty sequence")
        return self._rng.randrange(len(valid_actions))
