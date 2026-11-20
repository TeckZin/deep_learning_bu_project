"""Tokenizers (doc stages 5 patchify, 7 tokenize, 8 fuse x/y): (f, y_t) -> grid tokens.

Add a variant by writing a class with the same constructor/forward signature and
registering it in TOKENIZERS; select it with tokenizer.kind in submission/config.py.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .pos import pos2d


class FusedTokenizer(nn.Module):
    """(features [B, C, h_f, w_f], logits y_t [B, 1, H, W]) -> tokens [B, N, d], N = (H/patch)(W/patch).

    Image evidence and the current answer for the same patch become ONE token, so every
    answer token directly sees the image at its own location.
      fuse="sum"     token = feat(f) + ans(y)
      fuse="concat"  token = Linear([feat(f), ans(y)])
    """

    def __init__(self, c_in, d, patch, cfg) -> None:
        super().__init__()
        if cfg.fuse not in ("sum", "concat"):
            raise ValueError(f"tokenizer.fuse must be 'sum' or 'concat', got {cfg.fuse!r}")
        self.patch = patch
        self.uncertainty = cfg.uncertainty
        # Raw DINOv2 tokens have a large norm and can swamp the answer branch (attempt7).
        self.feat_norm = nn.LayerNorm(c_in) if cfg.feat_norm else None
        # Image branch: per-location MLP on the (pooled) frozen features.
        self.feat = nn.Sequential(nn.Conv2d(c_in, d, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        # Answer branch: patch embedding of [probability, scaled clamped logit(, uncertainty)].
        # Uncertainty 4p(1-p) is 1 where the answer is unsure (edges, gaps), 0 where sure.
        self.ans = nn.Conv2d(3 if cfg.uncertainty else 2, d, kernel_size=patch, stride=patch)
        self.fuse = nn.Linear(2 * d, d) if cfg.fuse == "concat" else None
        self.norm = nn.LayerNorm(d)

    def forward(self, features, logits, offset=None):
        H, W = logits.shape[-2:]
        h, w = H // self.patch, W // self.patch
        if h == 0 or w == 0:
            raise ValueError(f"Image {H}x{W} is smaller than patch {self.patch}")
        f = F.adaptive_avg_pool2d(features, (h, w))                        # [B, C, h, w]
        if self.feat_norm is not None:
            f = self.feat_norm(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # LayerNorm over C
        p = logits.sigmoid()
        channels = [p, logits.clamp(-10, 10) / 10]
        if self.uncertainty:
            channels.append(4 * p * (1 - p))
        a = self.ans(torch.cat(channels, dim=1))[..., :h, :w]              # [B, d, h, w]
        if self.fuse is None:
            x = (self.feat(f) + a).flatten(2).transpose(1, 2)               # [B, N, d]
        else:
            x = self.fuse(torch.cat([self.feat(f), a], dim=1).flatten(2).transpose(1, 2))
        x = x + pos2d(h, w, x.shape[-1], x.device, offset)
        return self.norm(x), h, w


TOKENIZERS = {"fused": FusedTokenizer}
