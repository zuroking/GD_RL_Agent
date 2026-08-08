"""Uniform experience replay buffer.

Stores (obs, action, reward, next_obs, terminal) transitions.
Sampling is uniform random — no priority weighting in Phase 4.
"""

import numpy as np
from numpy.typing import NDArray


class ReplayBuffer:
    """Circular buffer for DQN transitions.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Dimension of the observation vector.
    """

    def __init__(self, capacity: int, obs_dim: int = 4) -> None:
        self._cap = capacity
        self._obs_dim = obs_dim
        self._ptr = 0
        self._size = 0

        self._obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._terminals = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        obs: NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: NDArray[np.float32],
        terminal: bool,
    ) -> None:
        """Store one transition."""
        i = self._ptr
        self._obs[i]       = obs
        self._next_obs[i]  = next_obs
        self._actions[i]   = action
        self._rewards[i]   = reward
        self._terminals[i] = float(terminal)

        self._ptr = (i + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size: int) -> dict[str, NDArray]:
        """Sample a random batch.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict with keys: obs, action, reward, next_obs, terminal.

        Raises:
            ValueError: If the buffer has fewer transitions than batch_size.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer has {self._size} transitions, need {batch_size}."
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs":      self._obs[idx],
            "action":   self._actions[idx],
            "reward":   self._rewards[idx],
            "next_obs": self._next_obs[idx],
            "terminal": self._terminals[idx],
        }

    def __len__(self) -> int:
        return self._size
