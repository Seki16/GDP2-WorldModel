from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class DQNConfig:
    obs_type: Literal["pixel", "latent"] = "pixel"
    obs_size: int = 64
    latent_dim: int = 384
    n_actions: int = 4
    hidden_dim: int = 128

    # Pixel encoder architecture
    conv1_channels: int = 32
    conv2_channels: int = 64
    conv3_channels: int = 64
    conv_kernel_1: int = 5
    conv_kernel_2: int = 5
    conv_kernel_3: int = 3
    conv_stride_1: int = 2
    conv_stride_2: int = 2
    conv_stride_3: int = 2


class PixelEncoder(nn.Module):
    """
    CNN encoder for raw RGB maze observations.

    Input:
        x: (B, 3, H, W), float32 in [0, 1]

    Output:
        features: (B, feature_dim)
    """

    def __init__(self, config: DQNConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, config.conv1_channels, kernel_size=config.conv_kernel_1, stride=config.conv_stride_1),
            nn.ReLU(),
            nn.Conv2d(config.conv1_channels, config.conv2_channels, kernel_size=config.conv_kernel_2, stride=config.conv_stride_2),
            nn.ReLU(),
            nn.Conv2d(config.conv2_channels, config.conv3_channels, kernel_size=config.conv_kernel_3, stride=config.conv_stride_3),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.obs_size, config.obs_size)
            self.output_dim = self.net(dummy).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PixelDQN(nn.Module):
    """
    DQN for pixel observations.

    Input:
        x: (B, 3, H, W)

    Output:
        q_values: (B, n_actions)
    """

    def __init__(self, config: DQNConfig):
        super().__init__()
        self.config = config
        self.encoder = PixelEncoder(config)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


class LatentDQN(nn.Module):
    """
    DQN for latent observations.

    Input:
        z: (B, latent_dim)

    Output:
        q_values: (B, n_actions)
    """

    def __init__(self, config: DQNConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.n_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)