"""Output heads (doc stages 12 CNN, 13 head): grid tokens -> full-resolution correction dy.

The current head sees only the core's output and the current answer (state-conditioned detail
recovery, which the handout allows). Because heads run inside the loop (in every step()), they may
also take the image or features as input (TA, 2026-11-18); using the image on the final prediction
after the loop is not allowed. See docs/ChangingTheArchitecture.md.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PixelShuffleHead(nn.Module):
    """grid tokens [B, h*w, d] -> dy [B, 1, H, W].

    1. Each token -> a patch x patch block of logits (pixel_shuffle): coarse dy.
    2. A dilated conv stack on [coarse dy, current probability, ctx channels] sharpens
       edges and thin roads at full resolution. Dilations (1, 2, 4, 8, 1) give each
       output pixel ~31 px of context.
    ctx > 0: a few channels of the core's output tokens, upsampled, so the stack knows
             WHAT the core decided around each pixel, not only the 1-channel dy.
    gate:    two outputs r, g and dy = (coarse + r) * sigmoid(g + 2), so a step can leave
             already-correct pixels alone. Without it: dy = coarse + r.
    """

    def __init__(self, d, patch, cfg) -> None:
        super().__init__()
        self.patch = patch
        self.gate = cfg.gate
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, patch * patch)
        # Small, not zero: the untrained refiner stays near y_0 but every step still
        # changes the mask (the staff tests check that updates change the prediction).
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        self.ctx = nn.Linear(d, cfg.ctx) if cfg.ctx > 0 else None
        layers, c = [], 2 + cfg.ctx
        for dil in cfg.dilations:
            layers += [nn.Conv2d(c, cfg.hidden, 3, padding=dil, dilation=dil), nn.GELU()]
            c = cfg.hidden
        layers.append(nn.Conv2d(c, 2 if cfg.gate else 1, 3, padding=1))
        self.refine = nn.Sequential(*layers)
        # Last conv starts at zero: r = 0 (and gate = sigmoid(2) ~ 0.88) at initialization.
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, tokens, h, w, logits):
        H, W = logits.shape[-2:]
        B = tokens.shape[0]
        t = self.norm(tokens)
        x = self.proj(t)                                                     # [B, h*w, patch^2]
        x = x.transpose(1, 2).reshape(B, self.patch * self.patch, h, w)
        dy = F.pixel_shuffle(x, self.patch)                                  # [B, 1, h*patch, w*patch]
        if tuple(dy.shape[-2:]) != (H, W):
            dy = F.interpolate(dy, size=(H, W), mode="bilinear", align_corners=False)
        inputs = [dy, logits.sigmoid().to(dy.dtype)]
        if self.ctx is not None:
            c = self.ctx(t).transpose(1, 2).reshape(B, -1, h, w)             # [B, ctx, h, w]
            c = F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)
            inputs.append(c.to(dy.dtype))
        out = self.refine(torch.cat(inputs, dim=1))
        if not self.gate:
            return dy + out
        r, g = out[:, :1], out[:, 1:]
        return (dy + r) * torch.sigmoid(g + 2.0)


HEADS = {"pixelshuffle": PixelShuffleHead}
