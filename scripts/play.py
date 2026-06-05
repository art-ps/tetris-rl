"""Visually play an episode with an implemented baseline agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from tetris_rl.agents import DQNAgent, HeuristicAgent, RandomAgent
from tetris_rl.env import TetrisEnv
from tetris_rl.visualization import FallingPiece, PygameViewer, ViewerEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize a Tetris RL baseline agent.")
    parser.add_argument(
        "--agent",
        choices=("heuristic", "random", "dqn"),
        default="heuristic",
        help="agent to play (default: heuristic)",
    )
    parser.add_argument("--seed", type=int, default=42, help="episode seed (default: 42)")
    parser.add_argument(
        "--speed",
        type=float,
        default=30.0,
        help="initial falling speed in rows per second (default: 30)",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=30,
        help="board cell size in pixels (default: 30)",
    )
    parser.add_argument("--checkpoint", help="trained DQN checkpoint")
    parser.add_argument("--device", help="PyTorch device for DQN playback")
    return parser


def create_agent(
    name: str,
    seed: int,
    checkpoint: str | None = None,
    device: str | None = None,
):
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "dqn":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for the DQN agent")
        agent = DQNAgent(epsilon=0.0, device=device, seed=seed)
        agent.load(checkpoint)
        return agent
    return HeuristicAgent()


def select_action(agent, env: TetrisEnv) -> int:
    actions = env.get_valid_actions()
    if isinstance(agent, DQNAgent):
        return agent.select_action(actions, env.next_piece, epsilon=0.0)
    return agent.select_action(actions)


def main() -> None:
    args = build_parser().parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be greater than zero")
    if args.agent == "dqn":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required when --agent dqn")
        if not Path(args.checkpoint).exists():
            raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")

    env = TetrisEnv(seed=args.seed)
    agent = create_agent(args.agent, args.seed, args.checkpoint, args.device)
    viewer = PygameViewer(cell_size=args.cell_size, caption=f"Tetris RL - {args.agent}")
    speed = args.speed
    paused = False
    total_reward = 0.0
    action_index: int | None = None
    falling_piece: FallingPiece | None = None
    running = True

    try:
        while running:
            elapsed_ms = viewer.tick()
            for event in viewer.poll_events():
                if event is ViewerEvent.QUIT:
                    running = False
                elif event is ViewerEvent.TOGGLE_PAUSE:
                    paused = not paused
                elif event is ViewerEvent.RESTART:
                    env.reset(seed=args.seed)
                    agent = create_agent(args.agent, args.seed, args.checkpoint, args.device)
                    total_reward = 0.0
                    action_index = None
                    falling_piece = None
                elif event is ViewerEvent.SPEED_UP:
                    speed = min(120.0, speed * 1.5)
                elif event is ViewerEvent.SPEED_DOWN:
                    speed = max(1.0, speed / 1.5)

            if running and not paused and not env.done and falling_piece is None:
                action_index = select_action(agent, env)
                action = env.get_valid_actions()[action_index]
                falling_piece = FallingPiece.from_action(env.current_piece, action)

            if running and not paused and falling_piece is not None:
                if falling_piece.advance(elapsed_ms, speed):
                    assert action_index is not None
                    _, reward, _, _ = env.step(action_index)
                    total_reward += reward
                    action_index = None
                    falling_piece = None

            viewer.draw(
                env,
                agent_name=f"{args.agent.title()}Agent",
                speed=speed,
                total_reward=total_reward,
                paused=paused,
                falling_piece=falling_piece,
            )
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
