"""U-Net for 4-channel SAR -> 4-class layover/shadow segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, cout), cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, cout), cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """
    Four downsampling levels. GroupNorm rather than BatchNorm so small batches
    behave, which matters at 512x512 where batch size is memory-bound.
    """

    def __init__(self, cin: int = 4, classes: int = 4, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.down = nn.ModuleList(
            [ConvBlock(cin if i == 0 else c[i - 1], c[i]) for i in range(4)]
        )
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(c[3], c[4])
        self.up = nn.ModuleList(
            [ConvBlock(c[i] + c[i + 1], c[i]) for i in range(3, -1, -1)]
        )
        self.head = nn.Conv2d(c[0], classes, 1)

    def forward(self, x):
        skips = []
        for conv in self.down:
            x = conv(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for conv, skip in zip(self.up, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([x, skip], dim=1))
        return self.head(x)
