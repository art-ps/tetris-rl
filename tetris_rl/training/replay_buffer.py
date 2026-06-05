"""Fixed-capacity experience replay buffer."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    states: NDArray[np.float32]
    rewards: NDArray[np.float32]
    next_states: NDArray[np.float32]
    dones: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Transition:
    state: NDArray[np.float32]
    reward: float
    next_state: NDArray[np.float32]
    done: bool


class ReplayBuffer:
    """Store and uniformly sample immutable transition copies."""

    def __init__(self, capacity: int = 50_000, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        self.capacity = capacity
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: NDArray[np.floating],
        reward: float,
        next_state: NDArray[np.floating],
        done: bool,
    ) -> None:
        """Append a transition, copying feature arrays from the caller."""
        state_copy = np.asarray(state, dtype=np.float32).copy()
        next_state_copy = np.asarray(next_state, dtype=np.float32).copy()
        if state_copy.ndim != 1 or next_state_copy.ndim != 1:
            raise ValueError("state and next_state must be one-dimensional")
        if state_copy.shape != next_state_copy.shape:
            raise ValueError("state and next_state must have matching shapes")
        self._buffer.append(
            Transition(state_copy, float(reward), next_state_copy, bool(done))
        )

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample transitions without replacement."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if batch_size > len(self._buffer):
            raise ValueError("batch_size cannot exceed the number of transitions")
        transitions = self._rng.sample(list(self._buffer), batch_size)
        return ReplayBatch(
            states=np.stack([item.state for item in transitions]),
            rewards=np.asarray([item.reward for item in transitions], dtype=np.float32),
            next_states=np.stack([item.next_state for item in transitions]),
            dones=np.asarray([item.done for item in transitions], dtype=np.bool_),
        )
