"""Evaluate baseline agents and an optional trained DQN."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tetris_rl.agents import DQNAgent, HeuristicAgent, RandomAgent
from tetris_rl.env import Action, TetrisEnv


class Agent(Protocol):
    def select_action(self, valid_actions: Sequence[Action]) -> int: ...


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    agent: str
    episodes: int
    average_score: float
    average_lines: float
    average_pieces: float
    average_reward: float
    best_score: int
    best_lines: int


def evaluate_agent(
    name: str,
    agent_factory: Callable[[int], Agent],
    episodes: int,
    seed: int,
    max_steps: int = 5_000,
) -> EvaluationResult:
    """Play deterministic episode seeds and aggregate their metrics."""
    scores: list[int] = []
    lines: list[int] = []
    pieces: list[int] = []
    rewards: list[float] = []

    for offset in range(episodes):
        episode_seed = seed + offset
        env = TetrisEnv(seed=episode_seed)
        agent = agent_factory(episode_seed)
        done = False
        total_reward = 0.0

        for _ in range(max_steps):
            actions = env.get_valid_actions()
            if isinstance(agent, DQNAgent):
                action_index = agent.select_action(actions, env.next_piece, epsilon=0.0)
            else:
                action_index = agent.select_action(actions)
            _, reward, done, _ = env.step(action_index)
            total_reward += reward
            if done:
                break

        scores.append(env.score)
        lines.append(env.lines_cleared)
        pieces.append(env.pieces_placed)
        rewards.append(total_reward)

    return EvaluationResult(
        agent=name,
        episodes=episodes,
        average_score=sum(scores) / episodes,
        average_lines=sum(lines) / episodes,
        average_pieces=sum(pieces) / episodes,
        average_reward=sum(rewards) / episodes,
        best_score=max(scores),
        best_lines=max(lines),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare RandomAgent, HeuristicAgent, and an optional DQN.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="number of episodes per agent (default: 30)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="first environment seed (default: 42)",
    )
    parser.add_argument(
        "--checkpoint",
        help="include a trained DQN checkpoint in the comparison",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5_000,
        help="maximum placements per episode (default: 5000)",
    )
    parser.add_argument("--device", help="PyTorch device for DQN evaluation")
    return parser


def print_results(results: Sequence[EvaluationResult]) -> None:
    header = (
        f"{'agent':<12} {'avg score':>10} {'avg lines':>10} "
        f"{'avg pieces':>11} {'avg reward':>11} {'best score':>11} {'best lines':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.agent:<12} "
            f"{result.average_score:>10.1f} "
            f"{result.average_lines:>10.2f} "
            f"{result.average_pieces:>11.1f} "
            f"{result.average_reward:>11.1f} "
            f"{result.best_score:>11} "
            f"{result.best_lines:>10}"
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes <= 0 or args.max_steps <= 0:
        raise SystemExit("--episodes and --max-steps must be greater than zero")

    results = [
        evaluate_agent(
            "random",
            lambda episode_seed: RandomAgent(seed=episode_seed),
            args.episodes,
            args.seed,
            args.max_steps,
        ),
        evaluate_agent(
            "heuristic",
            lambda _: HeuristicAgent(),
            args.episodes,
            args.seed,
            args.max_steps,
        ),
    ]
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint does not exist: {checkpoint}")

        def dqn_factory(_: int) -> DQNAgent:
            agent = DQNAgent(epsilon=0.0, device=args.device)
            agent.load(checkpoint)
            return agent

        results.append(
            evaluate_agent(
                "dqn",
                dqn_factory,
                args.episodes,
                args.seed,
                args.max_steps,
            )
        )
    print_results(results)


if __name__ == "__main__":
    main()
