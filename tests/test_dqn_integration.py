from scripts.evaluate import evaluate_agent
from scripts.play import create_agent, select_action
from tetris_rl.agents import DQNAgent
from tetris_rl.env import TetrisEnv


def test_play_helpers_load_and_use_dqn_checkpoint(tmp_path):
    checkpoint = tmp_path / "agent.pt"
    DQNAgent(epsilon=0.0, seed=42).save(checkpoint)
    agent = create_agent("dqn", 42, str(checkpoint))
    env = TetrisEnv(seed=42)

    action_index = select_action(agent, env)

    assert 0 <= action_index < len(env.get_valid_actions())


def test_evaluate_agent_supports_dqn():
    result = evaluate_agent(
        "dqn",
        lambda _: DQNAgent(epsilon=0.0, seed=42),
        episodes=1,
        seed=42,
        max_steps=2,
    )

    assert result.agent == "dqn"
    assert result.average_pieces == 2
