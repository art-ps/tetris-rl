import numpy as np
import pytest

from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.env.pieces import PIECES, PIECE_NAMES
from tetris_rl.env.tetris import TetrisEnv


def test_board_initialization_and_piece_spawning():
    env = TetrisEnv(seed=7)

    state = env.reset(seed=7)

    assert state.shape == (20, 10)
    assert state.dtype == np.int8
    assert np.count_nonzero(state) == 0
    assert env.current_piece in PIECE_NAMES
    assert env.next_piece in PIECE_NAMES
    assert env.score == env.lines_cleared == env.pieces_placed == 0


def test_reset_returns_a_copy():
    env = TetrisEnv(seed=1)

    state = env.reset()
    state[0, 0] = 1

    assert env.board[0, 0] == 0


def test_valid_actions_cover_every_empty_board_o_placement():
    env = TetrisEnv(seed=1)
    env.current_piece = "O"
    env._valid_actions = None

    actions = env.get_valid_actions()

    assert len(actions) == 9
    assert {(action.rotation, action.x, action.y) for action in actions} == {
        (0, x, 18) for x in range(9)
    }
    assert all(np.count_nonzero(action.board) == 4 for action in actions)
    assert all(set(action.board.flat) <= {0, 2} for action in actions)
    assert all(not action.board.flags.writeable for action in actions)


def test_collision_detection():
    env = TetrisEnv(seed=1)
    horizontal_i = PIECES["I"][0]
    env.board[0, 1] = 1

    assert env._collides(horizontal_i, 0, 0)
    assert env._collides(horizontal_i, -1, 0)
    assert not env._collides(horizontal_i, 2, 0)


def test_step_clears_line_and_updates_metrics():
    env = TetrisEnv(seed=1)
    env.board[-1, :6] = 1
    env.current_piece = "I"
    env._valid_actions = None
    action_index = next(
        index
        for index, action in enumerate(env.get_valid_actions())
        if action.rotation == 0 and action.x == 6
    )

    state, reward, done, info = env.step(action_index)

    assert np.count_nonzero(state[-1]) == 0
    assert env.lines_cleared == 1
    assert env.score == 100
    assert env.pieces_placed == 1
    assert info["lines_cleared_step"] == 1
    assert reward > 0
    assert not done


def test_line_clear_accepts_mixed_piece_colors():
    board = np.zeros((4, 7), dtype=np.int8)
    board[-1] = np.arange(1, 8, dtype=np.int8)

    cleared_board, lines = TetrisEnv._clear_lines(board)

    assert lines == 1
    assert np.count_nonzero(cleared_board) == 0


def test_step_promotes_next_piece_and_keeps_piece_colors():
    env = TetrisEnv(seed=1)
    env.current_piece = "T"
    env.next_piece = "L"
    env._valid_actions = None

    state, _, _, info = env.step(0)

    assert env.current_piece == "L"
    assert info["current_piece"] == "L"
    assert info["next_piece"] == env.next_piece
    assert set(state.flat) == {0, 3}


def test_preview_is_always_promoted_to_current_piece():
    env = TetrisEnv(seed=42)

    for _ in range(10):
        preview_before_step = env.next_piece
        env.step(0)
        assert env.current_piece == preview_before_step


def test_game_over_when_piece_cannot_spawn():
    env = TetrisEnv(seed=1)
    env.board[0, :] = 1
    env.current_piece = "O"
    env._valid_actions = None

    assert env.get_valid_actions() == []
    with pytest.raises(RuntimeError, match="no valid actions"):
        env.step(0)
    assert env.done


def test_clone_is_independent_and_preserves_rng_state():
    env = TetrisEnv(seed=42)
    clone = env.clone()

    clone.board[-1, 0] = 1

    assert env.board[-1, 0] == 0
    assert clone.current_piece == env.current_piece
    assert clone.next_piece == env.next_piece
    assert clone._rng.choice(PIECE_NAMES) == env._rng.choice(PIECE_NAMES)


def test_render_text_has_board_and_border():
    env = TetrisEnv(width=4, height=4, seed=1)
    env.board[-1, 0] = 1

    rendered = env.render_text()

    assert rendered.splitlines()[-2] == "|#...|"
    assert rendered.splitlines()[0] == "+----+"


def test_random_agent_can_play_complete_episode():
    env = TetrisEnv(seed=3)
    agent = RandomAgent(seed=3)
    done = False

    for _ in range(1_000):
        action_index = agent.select_action(env.get_valid_actions())
        _, _, done, _ = env.step(action_index)
        if done:
            break

    assert done
    assert env.pieces_placed > 0


def test_random_agent_rejects_empty_action_list():
    with pytest.raises(ValueError, match="empty"):
        RandomAgent().select_action([])
