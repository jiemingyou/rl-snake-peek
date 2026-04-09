from __future__ import annotations

import numpy as np
import torch

from snake_rl.config import Config


class ReplayBuffer:
    """Fixed-size circular replay buffer with pinned-memory staging for fast GPU transfer."""

    def __init__(self, config: Config | None = None, device: torch.device | str = "cpu"):
        cfg = config or Config()
        cap = cfg.buffer_size
        gs = cfg.grid_size
        nc = cfg.num_channels

        self.capacity = cap
        self.pos = 0
        self.size = 0
        self.device = torch.device(device)

        self.states = np.zeros((cap, nc, gs, gs), dtype=np.float32)
        self.actions = np.zeros(cap, dtype=np.int64)
        self.rewards = np.zeros(cap, dtype=np.float32)
        self.next_states = np.zeros((cap, nc, gs, gs), dtype=np.float32)
        self.dones = np.zeros(cap, dtype=np.float32)

        # Pre-allocated pinned tensors avoid a page-lock on every sample() call.
        # Data is copied into these, then transferred to GPU with non_blocking=True.
        bs = cfg.batch_size
        use_pin = self.device.type == "cuda"
        self._pin_states = torch.zeros(bs, nc, gs, gs, pin_memory=use_pin)
        self._pin_next_states = torch.zeros(bs, nc, gs, gs, pin_memory=use_pin)
        self._pin_actions = torch.zeros(bs, dtype=torch.int64, pin_memory=use_pin)
        self._pin_rewards = torch.zeros(bs, pin_memory=use_pin)
        self._pin_dones = torch.zeros(bs, pin_memory=use_pin)

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
        self, batch_size: int, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = torch.device(device) if device is not None else self.device
        indices = np.random.randint(0, self.size, size=batch_size)

        # Copy into pre-pinned staging tensors, then async-transfer to GPU
        self._pin_states.copy_(torch.from_numpy(self.states[indices]))
        self._pin_next_states.copy_(torch.from_numpy(self.next_states[indices]))
        self._pin_actions.copy_(torch.from_numpy(self.actions[indices]))
        self._pin_rewards.copy_(torch.from_numpy(self.rewards[indices]))
        self._pin_dones.copy_(torch.from_numpy(self.dones[indices]))

        nb = dev.type == "cuda"
        return (
            self._pin_states.to(dev, non_blocking=nb),
            self._pin_actions.to(dev, non_blocking=nb),
            self._pin_rewards.to(dev, non_blocking=nb),
            self._pin_next_states.to(dev, non_blocking=nb),
            self._pin_dones.to(dev, non_blocking=nb),
        )

    def __len__(self) -> int:
        return self.size
