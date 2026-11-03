"""Ty: mask tokenizer.

Turns the current answer map y_t [B, 1, H, W] (logits at full image
resolution) into tokens [B, N, d] on the same grid as the feature tokens
from Tx, so the two can be fused token by token.
"""
import torch
from torch import nn
from tokenizer.Tx import pos2d


class Ty(nn.Module):
    """Mask [B, 1, H, W] -> tokens [B, (H/patch)*(W/patch), d]."""

    def __init__(self, d=256, patch=8) -> None:
        super().__init__()
        assert d % 4 == 0  # required by pos2d
        self.patch = patch
        # Non-overlapping patch embedding: each patch x patch block of the
        # mask becomes one d-dim token. patch should equal the baseline's
        # downsampling factor so Ty's grid matches Tx's grid.
        self.proj = nn.Conv2d(1, d, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d)

    def forward(self, y):
        x = self.proj(y)                       # [B, d, H/patch, W/patch]
        B, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)       # [B, h*w, d], row-major like Tx
        x = x + pos2d(h, w, d, x.device)       # same position code as Tx
        x = self.norm(x)
        return x


if __name__ == "__main__":
    # Shape check: 64x80 mask with patch 8 -> 8x10 grid -> 80 tokens.
    model = Ty(d=256, patch=8)
    y = torch.randn(2, 1, 64, 80)
    out = model(y)
    print("input :", tuple(y.shape))
    print("output:", tuple(out.shape))
    print("params:", sum(p.numel() for p in model.parameters()))
