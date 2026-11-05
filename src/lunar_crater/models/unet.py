"""A plain U-Net that maps a DEM tile to per-pixel rim logits."""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net with ``depth`` downsamplings, starting at ``base`` channels.

    Upsampling is bilinear followed by convolution, which avoids the checkerboard
    artefacts of transposed convolutions on thin rims. Input sides must be divisible by
    ``2 ** depth``.
    """

    def __init__(self, in_channels: int = 1, base: int = 32, depth: int = 4):
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_channels, widths[0])])
        self.down.extend(_block(widths[i], widths[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(
            _block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(depth))
        )
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()
        for block in self.up:
            skip = skips.pop()
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear")
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


def rim_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Binary cross-entropy plus soft Dice; Dice keeps the sparse rims from being ignored."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(-2, -1))
    total = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1 - (2 * inter + eps) / (total + eps)
    return bce + dice.mean()
