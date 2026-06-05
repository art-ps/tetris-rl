"""Numeric state features used by agents."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tetris_rl.env.pieces import PIECE_NAMES


def column_heights(board: NDArray[np.integer]) -> NDArray[np.int_]:
    """Return the occupied height of every board column."""
    height, width = board.shape
    heights = np.zeros(width, dtype=int)
    for x in range(width):
        occupied = np.flatnonzero(board[:, x])
        if occupied.size:
            heights[x] = height - int(occupied[0])
    return heights


def count_holes(board: NDArray[np.integer]) -> int:
    """Count empty cells with at least one filled cell above them."""
    holes = 0
    for column in board.T:
        occupied = np.flatnonzero(column)
        if occupied.size:
            holes += int(np.count_nonzero(column[int(occupied[0]) :] == 0))
    return holes


def count_wells(board: NDArray[np.integer]) -> int:
    """Return the sum of well depths between columns and board walls."""
    heights = column_heights(board)
    wells = 0
    for x, value in enumerate(heights):
        left = heights[x - 1] if x > 0 else board.shape[0]
        right = heights[x + 1] if x < board.shape[1] - 1 else board.shape[0]
        wells += max(0, min(int(left), int(right)) - int(value))
    return wells


def extract_features(
    board: NDArray[np.integer],
    current_piece: str,
    completed_lines: int = 0,
    *,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """Build board statistics followed by column heights and piece one-hot."""
    if board.ndim != 2:
        raise ValueError("board must be a two-dimensional array")
    if current_piece not in PIECE_NAMES:
        raise ValueError(f"unknown piece: {current_piece}")

    height, width = board.shape
    heights = column_heights(board)
    aggregate_height = int(heights.sum())
    max_height = int(heights.max(initial=0))
    holes = count_holes(board)
    bumpiness = int(np.abs(np.diff(heights)).sum())
    wells = count_wells(board)

    base = np.array(
        [
            aggregate_height,
            max_height,
            holes,
            bumpiness,
            completed_lines,
            wells,
            *heights,
        ],
        dtype=np.float32,
    )
    if normalize:
        base[:6] /= np.array(
            [
                height * width,
                height,
                height * width,
                height * max(1, width - 1),
                4,
                height * width,
            ],
            dtype=np.float32,
        )
        base[6:] /= height

    one_hot = np.zeros(len(PIECE_NAMES), dtype=np.float32)
    one_hot[PIECE_NAMES.index(current_piece)] = 1.0
    return np.concatenate((base, one_hot), dtype=np.float32)
