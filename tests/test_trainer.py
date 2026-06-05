import csv

import numpy as np

from tetris_rl.training import Trainer
from tetris_rl.utils import TrainingConfig


def smoke_config() -> TrainingConfig:
    return TrainingConfig(
        episodes=2,
        batch_size=2,
        replay_buffer_size=50,
        min_replay_size=2,
        target_update_every=2,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.5,
        checkpoint_every=1,
        eval_every=1,
        eval_episodes=1,
        max_steps_per_episode=3,
        seed=7,
    )


def test_trainer_optimizes_batch(tmp_path):
    trainer = Trainer(
        smoke_config(),
        run_dir=tmp_path / "run",
        checkpoint_dir=tmp_path / "checkpoints",
    )
    for index in range(2):
        state = np.full(23, index, dtype=np.float32)
        trainer.replay_buffer.push(state, 1.0, state + 1, False)

    loss = trainer.optimize_step()

    assert loss >= 0


def test_trainer_smoke_run_writes_metrics_and_checkpoints(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = Trainer(
        smoke_config(),
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
    )

    history = trainer.train()

    assert len(history) == 2
    assert history[-1].epsilon == 0.25
    assert (run_dir / "metrics.csv").exists()
    assert (checkpoint_dir / "episode_1.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "final.pt").exists()
    with (run_dir / "metrics.csv").open() as metrics_file:
        assert len(list(csv.DictReader(metrics_file))) == 2
