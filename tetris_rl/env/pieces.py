"""Tetromino definitions."""

from __future__ import annotations

from typing import Final

Cell = tuple[int, int]
Rotation = tuple[Cell, ...]

PIECE_NAMES: Final[tuple[str, ...]] = ("I", "O", "T", "S", "Z", "J", "L")
PIECE_IDS: Final[dict[str, int]] = {
    name: index + 1 for index, name in enumerate(PIECE_NAMES)
}

PIECES: Final[dict[str, tuple[Rotation, ...]]] = {
    "I": (
        ((0, 0), (1, 0), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
    ),
    "O": (
        ((0, 0), (1, 0), (0, 1), (1, 1)),
    ),
    "T": (
        ((0, 0), (1, 0), (2, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (0, 2)),
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "S": (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "Z": (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
    "J": (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 0), (1, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    "L": (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (0, 1)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
}


def rotation_size(rotation: Rotation) -> tuple[int, int]:
    """Return a rotation's width and height."""
    return (
        max(x for x, _ in rotation) + 1,
        max(y for _, y in rotation) + 1,
    )
