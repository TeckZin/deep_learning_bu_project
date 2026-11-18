"""Building blocks of the refiner (everything except the shared core).

    pos2d          fixed 2D sin/cos position code, with an optional per-image offset
    GridTokenizer  features f + current answer y_t -> ONE token per patch (fused)
    Initializer    nz learned memory tokens read the grid once -> z_0
    Head           grid tokens -> full-resolution logit correction dy

All of this lives inside submission/ because the staff runner only ships and
hashes submission/*.py; code imported from other folders would be missing.
"""
import torch
from torch import nn
from torch.nn import functional as F


def pos2d(h, w, d, device=None, offset=None):
    """Fixed 2D sinusoidal position code for an h x w grid, row-major.

    offset=None      -> [h*w, d]      (grid starts at row 0, col 0: a full tile)
    offset [B, 2]    -> [B, h*w, d]   (row, col) of this grid's top-left cell
                                       inside the full tile, in grid cells.
    train.py passes the crop position here, so a 512 crop that came from the
    bottom-right of a tile gets the same codes it will have at full-tile inference.
    First d/2 channels encode the column, last d/2 the row.
    """
    assert d % 4 == 0, "d must be divisible by 4"
    rows = torch.arange(h, device=device, dtype=torch.float32).repeat_interleave(w)   # [h*w]
    cols = torch.arange(w, device=device, dtype=torch.float32).repeat(h)               # [h*w]
    if offset is not None:
        offset = offset.to(device=device, dtype=torch.float32)
        rows = rows[None] + offset[:, :1]                                              # [B, h*w]
        cols = cols[None] + offset[:, 1:]
    omega = 1.0 / (10000 ** (torch.arange(d // 4, device=device, dtype=torch.float32) / (d // 4)))
    r = rows[..., None] * omega
    c = cols[..., None] * omega
    return torch.cat([c.sin(), c.cos(), r.sin(), r.cos()], dim=-1)


class GridTokenizer(nn.Module):
    """(features [B, C, h_f, w_f], logits y_t [B, 1, H, W]) -> tokens [B, N, d], N = (H/patch)(W/patch).

    Image evidence and the current answer for the same patch are summed into ONE
    token, so every answer token directly sees the image at its own location
    (in the old design they were separate tokens and, in cross mode, could only
    meet through the 32 memory slots).
    """

    def __init__(self, c_in, d=256, patch=16) -> None:
        super().__init__()
        self.patch = patch
        # attempt7: normalize the frozen features per location first. Raw DINOv2 tokens
        # have a large norm and swamped the answer branch, so the core barely saw y_t.
        self.feat_norm = nn.LayerNorm(c_in)
        # Image branch: per-location MLP on the (pooled) baseline features.
        self.feat = nn.Sequential(nn.Conv2d(c_in, d, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        # Answer branch: patch embedding of [probability, scaled clamped logit, uncertainty].
        # attempt7: uncertainty 4p(1-p) is 1 where the answer is unsure (edges, gaps), 0 where sure.
        self.ans = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d)

    def forward(self, features, logits, offset=None):
        H, W = logits.shape[-2:]
        h, w = H // self.patch, W // self.patch
        if h == 0 or w == 0:
            raise ValueError(f"Image {H}x{W} is smaller than patch {self.patch}")
        # Resample features onto the patch grid (any baseline stride works).
        f = F.adaptive_avg_pool2d(features, (h, w))                        # [B, C, h, w]
        f = self.feat_norm(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)       # LayerNorm over C
        p = logits.sigmoid()
        y = torch.cat([p, logits.clamp(-10, 10) / 10, 4 * p * (1 - p)], dim=1)
        x = self.feat(f) + self.ans(y)[..., :h, :w]                         # [B, d, h, w]
        x = x.flatten(2).transpose(1, 2)                                    # [B, N, d]
        x = x + pos2d(h, w, x.shape[-1], x.device, offset)
        return self.norm(x), h, w


class Initializer(nn.Module):
    """grid tokens [B, N, d] -> z_0 [B, nz, d]. Runs in initialize(); never calls the core."""

    def __init__(self, d=256, nz=32, nhead=8) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, nz, d) * 0.02)          # learned memory slots
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, grid):
        q = self.norm_q(self.queries.expand(grid.shape[0], -1, -1))         # [B, nz, d]
        kv = self.norm_kv(grid)
        z, _ = self.attn(q, kv, kv, need_weights=False)                     # slots read the grid
        return self.norm_out(q + z)


class Head(nn.Module):
    """grid tokens [B, h*w, d] -> correction dy [B, 1, H, W].

    1. Each token -> a patch x patch block of logits (pixel_shuffle): coarse dy.
    2. A dilated conv stack on [coarse dy, current probability] sharpens edges and
       thin roads at full resolution. It never sees the image, only the core's
       correction and the current answer (state-conditioned detail recovery,
       which the handout allows), so it is not a segmentation bypass.

    attempt4: the refine stack is wider (64 channels) and dilated (1, 2, 4, 8, 1), so
    each output pixel sees ~31 px of context instead of ~7 px. That is enough to
    straighten an edge that a 1/16-resolution baseline blurred over 16-px blocks.
    """

    def __init__(self, d=256, patch=16, hidden=64, dilations=(1, 2, 4, 8, 1), ctx=16) -> None:
        super().__init__()
        self.patch = patch
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, patch * patch)
        # Small, not zero: the untrained refiner stays near y_0 but every step still
        # changes the mask (the staff tests check that updates change the prediction).
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        # attempt7: a few channels of the core's output tokens, upsampled, so the
        # full-resolution stack knows WHAT the core decided around each pixel (road
        # direction, "this gap should be connected"), not only the 1-channel dy.
        # This is state (core output), never the image: still not a segmentation bypass.
        self.ctx = nn.Linear(d, ctx)
        layers, c = [], 2 + ctx
        for dil in dilations:
            layers += [nn.Conv2d(c, hidden, 3, padding=dil, dilation=dil), nn.GELU()]
            c = hidden
        # attempt7: two outputs, a residual r and a gate g: dy = (coarse + r) * sigmoid(g + 2).
        # The gate lets a step leave already-correct pixels alone, which keeps K=1 from
        # dropping below K=0 and keeps K=6/8 stable.
        layers.append(nn.Conv2d(hidden, 2, 3, padding=1))
        self.refine = nn.Sequential(*layers)
        # Last conv starts at zero: r = 0, gate = sigmoid(2) ~ 0.88 at initialization.
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
        c = self.ctx(t).transpose(1, 2).reshape(B, -1, h, w)                 # [B, ctx, h, w]
        c = F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)
        out = self.refine(torch.cat([dy, logits.sigmoid().to(dy.dtype), c.to(dy.dtype)], dim=1))
        r, g = out[:, :1], out[:, 1:]
        return (dy + r) * torch.sigmoid(g + 2.0)
