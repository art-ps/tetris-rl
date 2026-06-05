"""Training configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    episodes: int = 3_000
    gamma: float = 0.99
    learning_rate: float = 0.001
    batch_size: int = 128
    replay_buffer_size: int = 50_000
    min_replay_size: int = 1_000
    target_update_every: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    checkpoint_every: int = 100
    eval_every: int = 100
    eval_episodes: int = 10
    max_steps_per_episode: int = 5_000
    seed: int = 42

    def validate(self) -> None:
        positive_ints = (
            "episodes",
            "batch_size",
            "replay_buffer_size",
            "min_replay_size",
            "target_update_every",
            "checkpoint_every",
            "eval_every",
            "eval_episodes",
            "max_steps_per_episode",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.min_replay_size > self.replay_buffer_size:
            raise ValueError("min_replay_size cannot exceed replay_buffer_size")
        if self.batch_size > self.replay_buffer_size:
            raise ValueError("batch_size cannot exceed replay_buffer_size")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be between zero and one")


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a YAML config while rejecting unknown keys."""
    with Path(path).open(encoding="utf-8") as config_file:
        raw: dict[str, Any] = yaml.safe_load(config_file) or {}
    valid_names = {item.name for item in fields(TrainingConfig)}
    unknown = set(raw) - valid_names
    if unknown:
        raise ValueError(f"unknown training config keys: {', '.join(sorted(unknown))}")
    config = TrainingConfig(**raw)
    config.validate()
    return config
