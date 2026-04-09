from __future__ import annotations

import torch
import torch.nn as nn

from snake_rl.config import Config

POOL_SIZE = 8


class DQN(nn.Module):
    """CNN-based Deep Q-Network.

    6 conv layers (3x3, padding=1, no stride) give a 13x13 receptive field while
    keeping feature maps at full grid resolution. This means Grad-CAM heatmaps
    map 1:1 to grid cells with no upsampling artifacts.

    An adaptive pool between the last conv and the FC head keeps the parameter
    count tractable regardless of grid size.
    """

    def __init__(self, config: Config | None = None):
        super().__init__()
        cfg = config or Config()

        self.conv1 = nn.Conv2d(cfg.num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # Grad-CAM target

        self.pool = nn.AdaptiveAvgPool2d(POOL_SIZE)

        self.fc1 = nn.Linear(64 * POOL_SIZE * POOL_SIZE, 512)
        self.fc2 = nn.Linear(512, cfg.num_actions)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
