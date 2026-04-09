from __future__ import annotations

import numpy as np
import torch

from snake_rl.config import Config


class ReplayBuffer:
    """Fixed-size circular replay buffer backed by pre-allocated numpy arrays."""

    def __init__(self, config: Config | None = None):
        cfg = config or Config()
        cap = cfg.buffer_size
        gs = cfg.grid_size
        nc = cfg.num_channels

        self.capacity = cap
        self.pos = 0
        self.size = 0

        self.states = np.zeros((cap, nc, gs, gs), dtype=np.float32)
        self.actions = np.zeros(cap, dtype=np.int64)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.next_states = np.zeros((cap, nc, gs, gs), dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Insert N transitions at once. Arrays have leading dim N."""
        n = states.shape[0]
        if self.pos + n <= self.capacity:
            s = slice(self.pos, self.pos + n)
            self.states[s] = states
            self.actions[s] = actions
            self.rewards[s] = rewards
            self.next_states[s] = next_states
            self.dones[s] = dones
        else:
            # Wraps around the end of the buffer
            first = self.capacity - self.pos
            self.states[self.pos :] = states[:first]
            self.actions[self.pos :] = actions[:first]
            self.rewards[self.pos :] = rewards[:first]
            self.next_states[self.pos :] = next_states[:first]
            self.dones[self.pos :] = dones[:first]

            rest = n - first
            self.states[:rest] = states[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_states[:rest] = next_states[first:]
            self.dones[:rest] = dones[first:]

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]).to(device),
            torch.from_numpy(self.actions[indices]).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.next_states[indices]).to(device),
            torch.from_numpy(self.dones[indices]).to(device),
        )

    def __len__(self) -> int:
        return self.size
