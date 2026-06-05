import numpy as np
import pytest

from tetris_rl.agents import HeuristicAgent, RandomAgent
from tetris_rl.env import TetrisEnv


def play_episode(agent, seed: int) -> TetrisEnv:
    env = TetrisEnv(seed=seed)
    done = False
    while not done:
        action_index = agent.select_action(env.get_valid_actions())
        _, _, done, _ = env.step(action_index)
    return env


def test_heuristic_agent_prefers_line_clear():
    env = TetrisEnv(seed=1)
    env.board[-1, :6] = 1
    env.current_piece = "I"
    env._valid_actions = None
    actions = env.get_valid_actions()

    selected = actions[HeuristicAgent().select_action(actions)]

    assert selected.rotation == 0
    assert selected.x == 6
    assert selected.lines_cleared == 1


def test_score_board_penalizes_holes_and_height():
    flat = np.zeros((20, 10), dtype=np.int8)
    flat[-1, :4] = 1
    board_with_hole = flat.copy()
    board_with_hole[-2, 0] = 1
    agent = HeuristicAgent()

    assert agent.score_board(flat) > agent.score_board(board_with_hole)


def test_heuristic_agent_rejects_empty_action_list():
    with pytest.raises(ValueError, match="empty"):
        HeuristicAgent().select_action([])


def test_heuristic_agent_outperforms_random_baseline():
    seeds = range(10)
    random_lines = sum(
        play_episode(RandomAgent(seed=seed), seed).lines_cleared for seed in seeds
    )
    heuristic_lines = sum(
        play_episode(HeuristicAgent(), seed).lines_cleared for seed in seeds
    )

    assert heuristic_lines > random_lines
