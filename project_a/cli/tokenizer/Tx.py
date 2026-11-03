"""Tx: feature tokenizer.

Turns the baseline's feature map f [B, C, h, w] into a sequence of tokens
[B, N, d] (N = h * w) that the transformer core can attend over. Each token
gets a fixed 2D sin/cos position code so the core knows where it sits in the image.
"""
import torch
from torch import nn
from torch.nn import functional as F


def pos2d(h, w, d, device=None):
    """Fixed 2D sinusoidal position encoding for an h x w grid.

    Returns a [h * w, d] tensor in row-major order (same order as
    flatten(2) on a [B, d, h, w] map). The first d/2 channels encode the
    column (x), the last d/2 encode the row (y). No learned parameters, so it
    works for any grid size.
    """
    assert d % 4 == 0  # d/4 frequencies each for sin(x), cos(x), sin(y), cos(y)
    # Row and column index of every grid cell, row-major.
    y = torch.arange(h, device=device).repeat_interleave(w)
    x = torch.arange(w, device=device).repeat(h)
    # Frequencies from 1 down to 1/10000, geometrically spaced.
    omega = 1.0 / (10000 ** (torch.arange(d // 4, device=device) / (d // 4)))
    x = x[:, None] * omega[None]  # [h*w, d/4]
    y = y[:, None] * omega[None]  # [h*w, d/4]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)  # [h*w, d]


class Tx(nn.Module):
    """Feature map [B, c_in, h, w] -> tokens [B, h*w, d]."""

    def __init__(self, c_in, d=256) -> None:
        super().__init__()
        assert d % 4 == 0  # required by pos2d
        # 1x1 conv: projects each feature vector to width d, keeps resolution.
        self.proj = nn.Conv2d(c_in, d, kernel_size=1)
        self.norm = nn.LayerNorm(d)

    def forward(self, f):
        x = self.proj(f)                       # [B, d, h, w]
        B, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)       # [B, h*w, d], one token per location
        x = x + pos2d(h, w, d, x.device)       # add position (broadcast over B)
        x = self.norm(x)                       # keep token scale stable for the core
        return x


if __name__ == "__main__":
    # Shape check: 8x10 feature map -> 80 tokens of width 256.
    model = Tx(c_in=64, d=256)
    f = torch.randn(2, 64, 8, 10)
    out = model(f)
    print("input :", tuple(f.shape))
    print("output:", tuple(out.shape))
    print("pos2d :", tuple(pos2d(8, 10, 256).shape))
