"""Compact U-Net used by the learned segmentation engine.

Outputs three sigmoid maps per pixel (all in [0, 1]):

0. foreground probability,
1. cell-boundary probability (pixels on the edge of a cell, including the
   seam between two touching cells),
2. normalised distance: each cell's Euclidean distance transform divided by
   its own maximum, so every cell peaks at 1 on its mid-line and falls to 0 at
   its edge. Seeds for the watershed are taken where this map is high.

Only operators supported by OpenCV's DNN module are used (Conv, BatchNorm,
ReLU, MaxPool, bilinear Resize, Concat, Sigmoid) so the exported ONNX model
runs without PyTorch or onnxruntime.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Four-level U-Net. Input height and width must be multiples of 16."""

    def __init__(self, base: int = 24, in_channels: int = 1, out_channels: int = 3) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList(
            [ConvBlock(in_channels, c[0])] + [ConvBlock(c[i], c[i + 1]) for i in range(4)]
        )
        self.dec = nn.ModuleList([ConvBlock(c[i + 1] + c[i], c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for block in self.dec:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return torch.sigmoid(self.head(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
