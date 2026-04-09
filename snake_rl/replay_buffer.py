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
