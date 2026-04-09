from __future__ import annotations

import torch
import torch.nn as nn

from snake_rl.config import Config

POOL_SIZE = 8


class DQN(nn.Module):
    """CNN-based Deep Q-Network.

    Conv layers use padding=1, no stride, so feature maps stay at full grid
    resolution (grid_size x grid_size). This gives Grad-CAM heatmaps that are
    natively grid-aligned with no upsampling needed.

    An adaptive average pool sits between the last conv and the FC head so that
    the parameter count stays tractable regardless of grid size.
    """

    def __init__(self, config: Config | None = None):
        super().__init__()
        cfg = config or Config()

        self.conv1 = nn.Conv2d(cfg.num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Grad-CAM target

        # Reduce spatial dims before FC; Grad-CAM hooks conv3 output at full resolution
        self.pool = nn.AdaptiveAvgPool2d(POOL_SIZE)

        self.fc1 = nn.Linear(64 * POOL_SIZE * POOL_SIZE, 512)
        self.fc2 = nn.Linear(512, cfg.num_actions)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
