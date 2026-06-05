import numpy as np
import pytest

from tetris_rl.training import ReplayBuffer


def test_replay_buffer_push_and_sample():
    buffer = ReplayBuffer(capacity=10, seed=42)
    for index in range(5):
        state = np.full(23, index, dtype=np.float32)
        buffer.push(state, index * 0.5, state + 1, index == 4)

    batch = buffer.sample(3)

    assert len(buffer) == 5
    assert batch.states.shape == (3, 23)
    assert batch.next_states.shape == (3, 23)
    assert batch.rewards.shape == (3,)
    assert batch.dones.shape == (3,)
    assert batch.states.dtype == np.float32
    assert batch.dones.dtype == np.bool_


def test_replay_buffer_copies_inputs_and_evicts_oldest():
    buffer = ReplayBuffer(capacity=2, seed=42)
    state = np.zeros(3, dtype=np.float32)
    buffer.push(state, 0, state, False)
    state[:] = 99
    buffer.push(np.ones(3), 1, np.ones(3), False)
    buffer.push(np.full(3, 2), 2, np.full(3, 2), True)

    batch = buffer.sample(2)

    assert len(buffer) == 2
    assert {float(row[0]) for row in batch.states} == {1.0, 2.0}


def test_replay_buffer_validates_transitions_and_batch_size():
    buffer = ReplayBuffer(capacity=2)

    with pytest.raises(ValueError, match="one-dimensional"):
        buffer.push(np.zeros((2, 2)), 0, np.zeros((2, 2)), False)
    with pytest.raises(ValueError, match="matching shapes"):
        buffer.push(np.zeros(2), 0, np.zeros(3), False)
    with pytest.raises(ValueError, match="exceed"):
        buffer.sample(1)
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=0)
