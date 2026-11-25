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


class SplitTokenizer(nn.Module):
    """(x, E, y_t, z_t) -> tokens [B, N, d]: one tokenizer per input, then one fusion (attempt10).

    TokX: frozen features x (+ image edge features E pooled to the patch) -> per-location MLP
    TokY: the current answer, patch embedding of [p, logit/10, 4p(1-p)] (as FusedTokenizer)
    TokZ: the spatial memory, strided conv from each patch's (patch/s x patch/s) cells of z
    Each token is LayerNormed before fusing, so no source dominates (raw DINO features have a large norm):
      fuse="concat"  token = Linear([Tx ; Ty ; Tz])
      fuse="mlp"     token = MLP([Tx ; Ty ; Tz])    x, y, z can interact before the first attention
    """

    def __init__(self, c_x, c_edges, cz, z_stride, d, patch, cfg, add_pos) -> None:
        super().__init__()
        if cfg.fuse not in ("concat", "mlp"):
            raise ValueError(f"split tokenizer: fuse must be 'concat' or 'mlp', got {cfg.fuse!r}")
        if patch % z_stride:
            raise ValueError(f"patch {patch} must be a multiple of memory.stride {z_stride}")
        self.patch = patch
        self.add_pos = add_pos
        self.x_norm = nn.LayerNorm(c_x)
        self.tok_x = nn.Sequential(nn.Conv2d(c_x + c_edges, d, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        self.tok_y = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        self.tok_z = nn.Conv2d(cz, d, kernel_size=patch // z_stride, stride=patch // z_stride)
        self.source_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])   # one each for x, y, z
        if cfg.fuse == "concat":
            self.fuse = nn.Linear(3 * d, d)
        else:
            self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x, edges, logits, z, offset=None):
        H, W = logits.shape[-2:]
        h, w = H // self.patch, W // self.patch
        if h == 0 or w == 0:
            raise ValueError(f"Image {H}x{W} is smaller than patch {self.patch}")
        x = F.adaptive_avg_pool2d(x, (h, w))
        x = self.x_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)          # LayerNorm over channels
        if edges is not None:
            x = torch.cat([x, F.adaptive_avg_pool2d(edges, (h, w))], dim=1)
        p = logits.sigmoid()
        tx = self.tok_x(x)
        ty = self.tok_y(torch.cat([p, logits.clamp(-10, 10) / 10, 4 * p * (1 - p)], dim=1))[..., :h, :w]
        tz = self.tok_z(z)[..., :h, :w]
        parts = [norm(t.flatten(2).transpose(1, 2)) for norm, t in zip(self.source_norms, (tx, ty, tz))]
        tokens = self.fuse(torch.cat(parts, dim=-1))                         # [B, N, d]
        if self.add_pos:
            tokens = tokens + pos2d(h, w, tokens.shape[-1], tokens.device, offset)
        return self.norm(tokens), h, w


TOKENIZERS = {"fused": FusedTokenizer}
