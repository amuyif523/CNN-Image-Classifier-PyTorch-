from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Utility Conv2d -> BatchNorm -> ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """A compact CNN that performs well on CIFAR-like datasets."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, dropout=0.05),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64, dropout=0.1),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, dropout=0.15),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes: int) -> SimpleCNN:
    """Factory that hides concrete architecture from callers."""
    return SimpleCNN(num_classes=num_classes)
