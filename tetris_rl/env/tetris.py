"""A candidate-placement Tetris environment."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tetris_rl.env.features import column_heights, count_holes
from tetris_rl.env.pieces import PIECES, PIECE_IDS, PIECE_NAMES, Rotation, rotation_size


@dataclass(frozen=True, slots=True)
class Action:
    """A valid final placement and its resulting board candidate."""

    rotation: int
    x: int
    y: int
    lines_cleared: int
    board: NDArray[np.int8] = field(repr=False, compare=False)


class TetrisEnv:
    """Simplified Tetris where each action is a complete hard-drop placement."""

    LINE_SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}

    def __init__(self, width: int = 10, height: int = 20, seed: int | None = None):
        if width < 4 or height < 4:
            raise ValueError("board width and height must both be at least 4")
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self.board = np.zeros((height, width), dtype=np.int8)
        self.current_piece = PIECE_NAMES[0]
        self.next_piece = PIECE_NAMES[0]
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.done = False
        self._valid_actions: list[Action] | None = None
        self.reset()

    def reset(self, seed: int | None = None) -> NDArray[np.int8]:
        """Reset the episode and return a copy of the empty board."""
        if seed is not None:
            self._rng.seed(seed)
        self.board.fill(0)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.done = False
        self.current_piece = self._rng.choice(PIECE_NAMES)
        self.next_piece = self._rng.choice(PIECE_NAMES)
        self._valid_actions = None
        return self.board.copy()

    def _collides(self, rotation: Rotation, x: int, y: int) -> bool:
        for dx, dy in rotation:
            board_x, board_y = x + dx, y + dy
            if (
                board_x < 0
                or board_x >= self.width
                or board_y < 0
                or board_y >= self.height
                or self.board[board_y, board_x]
            ):
                return True
        return False

    def _drop_y(self, rotation: Rotation, x: int) -> int | None:
        if self._collides(rotation, x, 0):
            return None
        y = 0
        while not self._collides(rotation, x, y + 1):
            y += 1
        return y

    @staticmethod
    def _clear_lines(board: NDArray[np.int8]) -> tuple[NDArray[np.int8], int]:
        full_rows = np.all(board != 0, axis=1)
        cleared = int(full_rows.sum())
        if not cleared:
            return board, 0
        remaining = board[~full_rows]
        empty = np.zeros((cleared, board.shape[1]), dtype=np.int8)
        return np.vstack((empty, remaining)), cleared

    def get_valid_actions(self) -> list[Action]:
        """Generate all valid rotations and x positions for the current piece."""
        if self.done:
            return []
        if self._valid_actions is not None:
            return list(self._valid_actions)

        actions: list[Action] = []
        for rotation_index, rotation in enumerate(PIECES[self.current_piece]):
            rotation_width, _ = rotation_size(rotation)
            for x in range(self.width - rotation_width + 1):
                y = self._drop_y(rotation, x)
                if y is None:
                    continue
                candidate = self.board.copy()
                for dx, dy in rotation:
                    candidate[y + dy, x + dx] = PIECE_IDS[self.current_piece]
                candidate, lines = self._clear_lines(candidate)
                candidate.setflags(write=False)
                actions.append(Action(rotation_index, x, y, lines, candidate))
        self._valid_actions = actions
        return list(actions)

    def step(
        self, action_index: int
    ) -> tuple[NDArray[np.int8], float, bool, dict[str, Any]]:
        """Apply one candidate placement."""
        if self.done:
            raise RuntimeError("cannot step a finished episode; call reset()")

        actions = self.get_valid_actions()
        if not actions:
            self.done = True
            raise RuntimeError("no valid actions are available")
        if not 0 <= action_index < len(actions):
            raise IndexError(f"action index {action_index} is out of range")

        previous_heights = column_heights(self.board)
        previous_holes = count_holes(self.board)
        previous_bumpiness = int(np.abs(np.diff(previous_heights)).sum())

        action = actions[action_index]
        self.board = action.board.copy()
        self.pieces_placed += 1
        self.lines_cleared += action.lines_cleared
        self.score += self.LINE_SCORES[action.lines_cleared]

        new_heights = column_heights(self.board)
        holes_created = max(0, count_holes(self.board) - previous_holes)
        aggregate_height_increase = int(new_heights.sum() - previous_heights.sum())
        new_bumpiness = int(np.abs(np.diff(new_heights)).sum())
        bumpiness_increase = new_bumpiness - previous_bumpiness

        reward = action.lines_cleared * 10.0
        if action.lines_cleared == 4:
            reward += 50.0
        reward -= holes_created * 2.0
        reward -= aggregate_height_increase * 0.5
        reward -= bumpiness_increase * 0.2

        self.current_piece = self.next_piece
        self.next_piece = self._rng.choice(PIECE_NAMES)
        self._valid_actions = None
        self.done = not bool(self.get_valid_actions())
        if self.done:
            reward -= 100.0

        info = {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "lines_cleared_step": action.lines_cleared,
            "pieces_placed": self.pieces_placed,
            "max_height": int(new_heights.max(initial=0)),
            "holes": count_holes(self.board),
            "current_piece": self.current_piece,
            "next_piece": self.next_piece,
        }
        return self.board.copy(), reward, self.done, info

    def render_text(self) -> str:
        """Render the locked board as a compact text grid."""
        border = "+" + "-" * self.width + "+"
        rows = ["|" + "".join("#" if cell else "." for cell in row) + "|" for row in self.board]
        return "\n".join((border, *rows, border))

    def clone(self) -> TetrisEnv:
        """Return an independent copy, including random generator state."""
        return copy.deepcopy(self)
