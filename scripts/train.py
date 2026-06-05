"""Train the candidate-state DQN agent."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from tetris_rl.training import Trainer
from tetris_rl.utils import load_training_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Tetris DQN agent.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="training YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--run-dir",
        help="output run directory (default: runs/<timestamp>)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="checkpoint directory (default: checkpoints)",
    )
    parser.add_argument("--device", help="PyTorch device, for example cpu or mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_training_config(args.config)
    run_dir = Path(args.run_dir or f"runs/{datetime.now():%Y%m%d_%H%M%S}")
    trainer = Trainer(
        config,
        run_dir=run_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    progress = tqdm(total=config.episodes, desc="training", unit="episode")

    def update_progress(metrics) -> None:
        progress.update()
        progress.set_postfix(
            reward=f"{metrics.reward:.1f}",
            lines=metrics.lines,
            epsilon=f"{metrics.epsilon:.3f}",
            loss=f"{metrics.loss:.4f}",
        )

    trainer.train(update_progress)
    progress.close()
    print(f"metrics: {run_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
