"""Head: decodes tokens into a full-resolution logit correction dy.

The refiner returns y_(t+1) = y_t + dy, so the head only predicts a change to the
current mask, not a whole new mask.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """Tokens [B, h*w, d] on an h x w grid -> correction [B, 1, H, W]."""

    def __init__(self, d=256, patch=16) -> None:
        super().__init__()
        # Each token is decoded into a patch x patch block of pixels.
        # Must equal the patch size used by Ty so the grids line up.
        self.patch = patch
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, patch * patch)  # one logit per pixel in the block
        # Zero init: dy = 0 before training, so step() starts by returning y_t
        # unchanged and the refiner begins exactly at the baseline.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens, h, w, out_size=None):
        x = self.proj(self.norm(tokens))                       # [B, h*w, patch^2]
        B = x.shape[0]
        # Back to a channels-first grid: [B, patch^2, h, w].
        x = x.transpose(1, 2).reshape(B, self.patch * self.patch, h, w)
        # Rearrange the patch^2 channels into space: [B, 1, h*patch, w*patch].
        x = F.pixel_shuffle(x, self.patch)
        # If the image size is not a multiple of patch, resize to the exact
        # original H x W (predict.py requires the original resolution).
        if out_size is not None and tuple(x.shape[-2:]) != tuple(out_size):
            x = F.interpolate(x, size=tuple(out_size), mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    # Shape check: 8x10 grid of tokens with patch 8 -> 64x80 correction map.
    head = Head(d=256, patch=8)
    tokens = torch.randn(2, 80, 256)
    dy = head(tokens, 8, 10, out_size=(64, 80))
    print("tokens:", tuple(tokens.shape))
    print("dy    :", tuple(dy.shape))
