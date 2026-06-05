"""DQN training loop for candidate-placement Tetris."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tetris_rl.agents import DQNAgent
from tetris_rl.env import TetrisEnv
from tetris_rl.training.metrics import EpisodeMetrics, MetricsWriter
from tetris_rl.training.replay_buffer import ReplayBuffer
from tetris_rl.utils import TrainingConfig, set_global_seed


class Trainer:
    """Train a DQNAgent and persist metrics and checkpoints."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        run_dir: str | Path,
        checkpoint_dir: str | Path = "checkpoints",
        device: str | torch.device | None = None,
    ):
        config.validate()
        set_global_seed(config.seed)
        self.config = config
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.agent = DQNAgent(
            epsilon=config.epsilon_start,
            device=device,
            seed=config.seed,
        )
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size, seed=config.seed)
        self.optimizer = torch.optim.Adam(
            self.agent.q_network.parameters(),
            lr=config.learning_rate,
        )
        self.loss_function = nn.SmoothL1Loss()
        self.metrics_writer = MetricsWriter(self.run_dir / "metrics.csv")
        self.total_steps = 0
        self.best_eval_score = float("-inf")

    def train(
        self,
        on_episode_end: Callable[[EpisodeMetrics], None] | None = None,
    ) -> list[EpisodeMetrics]:
        """Run all configured episodes."""
        history: list[EpisodeMetrics] = []
        recent_rewards: deque[float] = deque(maxlen=100)
        recent_lines: deque[int] = deque(maxlen=100)

        for episode in range(1, self.config.episodes + 1):
            metrics = self.train_episode(episode)
            recent_rewards.append(metrics.reward)
            recent_lines.append(metrics.lines)
            metrics = EpisodeMetrics(
                episode=metrics.episode,
                reward=metrics.reward,
                score=metrics.score,
                lines=metrics.lines,
                pieces=metrics.pieces,
                epsilon=metrics.epsilon,
                loss=metrics.loss,
                avg_reward_100=sum(recent_rewards) / len(recent_rewards),
                avg_lines_100=sum(recent_lines) / len(recent_lines),
            )
            history.append(metrics)
            self.metrics_writer.append(metrics)
            if on_episode_end is not None:
                on_episode_end(metrics)

            if episode % self.config.checkpoint_every == 0:
                self.agent.save(
                    self.checkpoint_dir / f"episode_{episode}.pt",
                    episode=episode,
                    total_steps=self.total_steps,
                )
            if episode % self.config.eval_every == 0:
                evaluation_score = self.evaluate(self.config.eval_episodes)
                if evaluation_score > self.best_eval_score:
                    self.best_eval_score = evaluation_score
                    self.agent.save(
                        self.checkpoint_dir / "best.pt",
                        episode=episode,
                        total_steps=self.total_steps,
                        evaluation_score=evaluation_score,
                    )

        self.agent.save(
            self.checkpoint_dir / "final.pt",
            episode=self.config.episodes,
            total_steps=self.total_steps,
        )
        return history

    def train_episode(self, episode: int) -> EpisodeMetrics:
        """Play and optimize one episode."""
        env = TetrisEnv(seed=self.config.seed + episode)
        total_reward = 0.0
        losses: list[float] = []

        for step_index in range(self.config.max_steps_per_episode):
            actions = env.get_valid_actions()
            action_index = self.agent.select_action(actions, env.next_piece)
            state_features = self.agent.candidate_features(actions, env.next_piece)[
                action_index
            ]
            _, reward, done, _ = env.step(action_index)
            total_reward += reward
            transition_done = done or step_index == self.config.max_steps_per_episode - 1
            next_features = self._target_next_features(env, transition_done)
            self.replay_buffer.push(
                state_features,
                reward,
                next_features,
                transition_done,
            )
            self.total_steps += 1

            if len(self.replay_buffer) >= max(
                self.config.min_replay_size,
                self.config.batch_size,
            ):
                losses.append(self.optimize_step())
            if self.total_steps % self.config.target_update_every == 0:
                self.agent.update_target_network()
            if transition_done:
                break

        self.agent.epsilon = max(
            self.config.epsilon_end,
            self.agent.epsilon * self.config.epsilon_decay,
        )
        return EpisodeMetrics(
            episode=episode,
            reward=total_reward,
            score=env.score,
            lines=env.lines_cleared,
            pieces=env.pieces_placed,
            epsilon=self.agent.epsilon,
            loss=sum(losses) / len(losses) if losses else 0.0,
            avg_reward_100=0.0,
            avg_lines_100=0.0,
        )

    def optimize_step(self) -> float:
        """Perform one gradient update from replay memory."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        device = self.agent.device
        states = torch.from_numpy(batch.states).to(device)
        rewards = torch.from_numpy(batch.rewards).to(device)
        next_states = torch.from_numpy(batch.next_states).to(device)
        dones = torch.from_numpy(batch.dones).to(device)

        self.agent.q_network.train()
        predicted = self.agent.q_network(states).squeeze(-1)
        with torch.no_grad():
            next_values = self.agent.target_network(next_states).squeeze(-1)
            targets = rewards + self.config.gamma * next_values * (~dones)
        loss = self.loss_function(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def evaluate(self, episodes: int) -> float:
        """Return average greedy score without changing training epsilon."""
        scores: list[int] = []
        for offset in range(episodes):
            env = TetrisEnv(seed=self.config.seed + 100_000 + offset)
            for _ in range(self.config.max_steps_per_episode):
                actions = env.get_valid_actions()
                action_index = self.agent.select_action(
                    actions,
                    env.next_piece,
                    epsilon=0.0,
                )
                _, _, done, _ = env.step(action_index)
                if done:
                    break
            scores.append(env.score)
        return sum(scores) / len(scores)

    def _target_next_features(self, env: TetrisEnv, done: bool) -> np.ndarray:
        if done:
            return np.zeros(self.agent.input_dim, dtype=np.float32)
        candidates = self.agent.candidate_features(env.get_valid_actions(), env.next_piece)
        tensor = torch.from_numpy(candidates).to(self.agent.device)
        with torch.no_grad():
            values = self.agent.target_network(tensor).squeeze(-1)
        return candidates[int(torch.argmax(values).item())]
