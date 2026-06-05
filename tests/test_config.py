import pytest

from tetris_rl.utils import TrainingConfig, load_training_config


def test_load_training_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("episodes: 5\nbatch_size: 4\nmin_replay_size: 4\n")

    config = load_training_config(path)

    assert config.episodes == 5
    assert config.batch_size == 4
    assert config.gamma == 0.99


def test_training_config_rejects_invalid_values_and_unknown_keys(tmp_path):
    with pytest.raises(ValueError, match="min_replay_size"):
        TrainingConfig(replay_buffer_size=10, min_replay_size=11).validate()

    path = tmp_path / "config.yaml"
    path.write_text("mystery_option: true\n")
    with pytest.raises(ValueError, match="unknown"):
        load_training_config(path)
