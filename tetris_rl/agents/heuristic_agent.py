"""Hand-written Tetris baseline agent."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tetris_rl.env.features import column_heights, count_holes
from tetris_rl.env.tetris import Action


class HeuristicAgent:
    """Choose the candidate board with the highest hand-written score."""

    def score_board(
        self,
        board: NDArray[np.integer],
        completed_lines: int = 0,
    ) -> float:
        heights = column_heights(board)
        aggregate_height = int(heights.sum())
        holes = count_holes(board)
        bumpiness = int(np.abs(np.diff(heights)).sum())
        return (
            -0.5 * aggregate_height
            - 0.7 * holes
            - 0.3 * bumpiness
            + 1.0 * completed_lines
        )

    def select_action(self, valid_actions: Sequence[Action]) -> int:
        """Return the index of the highest-scoring candidate placement."""
        if not valid_actions:
            raise ValueError("cannot select an action from an empty sequence")
        return max(
            range(len(valid_actions)),
            key=lambda index: self.score_board(
                valid_actions[index].board,
                valid_actions[index].lines_cleared,
            ),
        )
