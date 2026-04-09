from __future__ import annotations

import torch
import torch.nn as nn

from snake_rl.config import Config


class DQN(nn.Module):
    """CNN-based Deep Q-Network.

    Uses padding=1 with no pooling so feature maps stay at grid_size x grid_size.
    This means Grad-CAM heatmaps are natively grid-aligned with no upsampling needed.
    """

    def __init__(self, config: Config | None = None):
        super().__init__()
        cfg = config or Config()
        gs = cfg.grid_size

        self.conv1 = nn.Conv2d(cfg.num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Grad-CAM target

        self.fc1 = nn.Linear(64 * gs * gs, 256)
        self.fc2 = nn.Linear(256, cfg.num_actions)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
