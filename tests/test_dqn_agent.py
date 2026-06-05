import torch
from torch import nn

from tetris_rl.agents import DQNAgent
from tetris_rl.env import TetrisEnv


class IncreasingQNetwork(nn.Module):
    def forward(self, features):
        return torch.arange(
            features.shape[0],
            dtype=features.dtype,
            device=features.device,
        ).unsqueeze(1)


def test_dqn_agent_extracts_candidate_features():
    env = TetrisEnv(seed=42)
    actions = env.get_valid_actions()
    agent = DQNAgent(epsilon=0, seed=42)

    features = agent.candidate_features(actions, env.next_piece)

    assert features.shape == (len(actions), 23)


def test_dqn_agent_greedy_selection_chooses_highest_q_value():
    env = TetrisEnv(seed=42)
    actions = env.get_valid_actions()
    agent = DQNAgent(epsilon=0, seed=42)
    agent.q_network = IncreasingQNetwork()

    selected = agent.select_action(actions, env.next_piece)

    assert selected == len(actions) - 1


def test_dqn_agent_random_selection_is_seeded():
    env = TetrisEnv(seed=42)
    actions = env.get_valid_actions()
    first = DQNAgent(epsilon=1, seed=7)
    second = DQNAgent(epsilon=1, seed=7)

    assert first.select_action(actions, env.next_piece) == second.select_action(
        actions,
        env.next_piece,
    )


def test_dqn_agent_updates_target_network():
    agent = DQNAgent(seed=42)
    with torch.no_grad():
        next(agent.q_network.parameters()).fill_(3)

    agent.update_target_network()

    assert all(
        torch.equal(online, target)
        for online, target in zip(
            agent.q_network.parameters(),
            agent.target_network.parameters(),
            strict=True,
        )
    )


def test_dqn_agent_saves_and_loads_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.pt"
    original = DQNAgent(epsilon=0.25, seed=42)
    with torch.no_grad():
        next(original.q_network.parameters()).fill_(4)
    original.update_target_network()
    original.save(path, episode=12)

    restored = DQNAgent(epsilon=1.0, seed=1)
    metadata = restored.load(path)

    assert restored.epsilon == 0.25
    assert metadata == {"episode": 12}
    assert all(
        torch.equal(saved, loaded)
        for saved, loaded in zip(
            original.q_network.parameters(),
            restored.q_network.parameters(),
            strict=True,
        )
    )
