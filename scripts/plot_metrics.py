"""Plot reward, cleared lines, and loss from a training run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_metrics(run_dir: str | Path) -> dict[str, list[float]]:
    path = Path(run_dir) / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"metrics file does not exist: {path}")
    with path.open(encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    if not rows:
        raise ValueError(f"metrics file is empty: {path}")
    return {
        key: [float(row[key]) for row in rows]
        for key in rows[0]
    }


def save_plot(
    episodes: list[float],
    raw: list[float],
    average: list[float] | None,
    *,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(episodes, raw, alpha=0.3, linewidth=0.8, label=ylabel)
    if average is not None:
        axis.plot(episodes, average, linewidth=1.8, label="rolling average 100")
    axis.set_xlabel("Episode")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_run(run_dir: str | Path) -> list[Path]:
    run_path = Path(run_dir)
    metrics = load_metrics(run_path)
    plots = (
        ("reward", "avg_reward_100", "Reward", "Training reward", "reward.png"),
        ("lines", "avg_lines_100", "Cleared lines", "Cleared lines", "lines.png"),
        ("loss", None, "Loss", "Training loss", "loss.png"),
    )
    paths: list[Path] = []
    for raw, average, ylabel, title, filename in plots:
        output = run_path / filename
        save_plot(
            metrics["episode"],
            metrics[raw],
            metrics[average] if average else None,
            ylabel=ylabel,
            title=title,
            path=output,
        )
        paths.append(output)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DQN training metrics.")
    parser.add_argument("--run", required=True, help="run directory containing metrics.csv")
    args = parser.parse_args()
    for path in plot_run(args.run):
        print(path)


if __name__ == "__main__":
    main()
