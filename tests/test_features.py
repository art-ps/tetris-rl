import numpy as np
import pytest

from tetris_rl.env.features import (
    column_heights,
    count_holes,
    count_wells,
    extract_features,
)
from tetris_rl.env.pieces import PIECE_NAMES


def test_feature_statistics():
    board = np.zeros((4, 4), dtype=np.int8)
    board[-1] = [1, 0, 1, 1]
    board[-2] = [1, 0, 0, 1]
    board[-3, 2] = 1

    assert column_heights(board).tolist() == [2, 0, 3, 2]
    assert count_holes(board) == 1
    assert count_wells(board) == 3


def test_extract_features_shape_values_and_piece_one_hot():
    board = np.zeros((20, 10), dtype=np.int8)
    board[-1, :2] = 1

    features = extract_features(board, "T", completed_lines=2, normalize=False)

    assert features.shape == (6 + 10 + 7,)
    assert features[:6].tolist() == [2, 1, 0, 1, 2, 0]
    assert features[6:16].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert features[16 + PIECE_NAMES.index("T")] == 1
    assert features[16:].sum() == 1


def test_normalized_features_are_bounded_for_regular_board():
    board = np.zeros((20, 10), dtype=np.int8)
    board[10:, ::2] = 1

    features = extract_features(board, "I")

    assert features.dtype == np.float32
    assert np.all(features >= 0)
    assert np.all(features <= 1)


def test_extract_features_rejects_invalid_input():
    with pytest.raises(ValueError, match="two-dimensional"):
        extract_features(np.zeros(10, dtype=np.int8), "I")
    with pytest.raises(ValueError, match="unknown piece"):
        extract_features(np.zeros((20, 10), dtype=np.int8), "X")
