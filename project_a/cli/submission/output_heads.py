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


class SpatialHead(nn.Module):
    """(core tokens, y_t, z_t, image edges E) -> (dy, z_t+1) (attempt10).

    1. Each token -> a patch x patch block of logits (coarse dy), plus ctx channels of the core
       output, as in PixelShuffleHead.
    2. A dilated conv stack at full resolution on [coarse dy, p, ctx, E, up(z)]: E lets it put the
       road boundary on the actual pixels; z brings what earlier steps noted.
    3. Two outputs from the stack's features:
         dy   = (coarse + r) * sigmoid(g + 2)         gated correction (as attempt7's gate)
         z'   = (1 - u) * z + u * tanh(c)             GRU-style update at z's resolution; bounded for any K
    """

    def __init__(self, d, patch, cfg, c_edges, cz) -> None:
        super().__init__()
        self.patch = patch
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, patch * patch)
        # Small, not zero: the untrained refiner stays near y_0 but every step still changes the mask.
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        self.ctx = nn.Linear(d, cfg.ctx) if cfg.ctx > 0 else None
        layers, c = [], 2 + cfg.ctx + c_edges + cz
        for dil in cfg.dilations:
            layers += [nn.Conv2d(c, cfg.hidden, 3, padding=dil, dilation=dil), nn.GELU()]
            c = cfg.hidden
        self.stack = nn.Sequential(*layers)
        self.dy_out = nn.Conv2d(cfg.hidden, 2, 3, padding=1)                   # r, g
        nn.init.zeros_(self.dy_out.weight)                                     # r = 0, gate ~ 0.88 at init
        nn.init.zeros_(self.dy_out.bias)
        self.z_out = nn.Conv2d(cfg.hidden + cz, 2 * cz, 3, padding=1)          # u, c

    def forward(self, tokens, h, w, logits, z, edges=None):
        H, W = logits.shape[-2:]
        B = tokens.shape[0]
        t = self.norm(tokens)
        x = self.proj(t).transpose(1, 2).reshape(B, self.patch * self.patch, h, w)
        coarse = F.pixel_shuffle(x, self.patch)                              # [B, 1, h*patch, w*patch]
        if tuple(coarse.shape[-2:]) != (H, W):
            coarse = F.interpolate(coarse, size=(H, W), mode="bilinear", align_corners=False)
        inputs = [coarse, logits.sigmoid().to(coarse.dtype)]
        if self.ctx is not None:
            c = self.ctx(t).transpose(1, 2).reshape(B, -1, h, w)
            inputs.append(F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False).to(coarse.dtype))
        if edges is not None:
            inputs.append(edges.to(coarse.dtype))
        inputs.append(F.interpolate(z, size=(H, W), mode="bilinear", align_corners=False).to(coarse.dtype))
        feats = self.stack(torch.cat(inputs, dim=1))                         # [B, hidden, H, W]

        r, g = self.dy_out(feats).chunk(2, dim=1)
        dy = (coarse + r) * torch.sigmoid(g + 2.0)

        pooled = F.adaptive_avg_pool2d(feats, z.shape[-2:])                  # to z's resolution
        u, c = self.z_out(torch.cat([pooled, z.to(pooled.dtype)], dim=1)).chunk(2, dim=1)
        u = torch.sigmoid(u)
        return dy, (1 - u) * z + u * torch.tanh(c)


HEADS = {"pixelshuffle": PixelShuffleHead}
