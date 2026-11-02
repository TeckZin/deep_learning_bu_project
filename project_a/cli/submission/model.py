# """UNTRAINED interface demonstration, not a competitive baseline or solution.
#
# Replace this package with your refiner. Keep the three-method interface and
# expose one nn.Module as `core`. The staff runner, not this class, owns K.
# State can be a tensor or a nested structure. All trainable modules must be
# registered on the refiner before checkpoint loading. No lazy K-dependent layers.
#
# Handout notation: `state` is z_t, a [B,N,d] bank of latent vector tokens;
# `previous_logits` is y_t, a [B,1,H,W] dense answer map (NOT ground truth).
# This example embeds image features and downsampled sigmoid(y_t), then fuses
# them into token context. The core consumes [B,N,d], not full-resolution pixels.
# The readout decodes z_(t+1) to a logit correction and returns y_(t+1).
# """
# import torch
# from torch import nn
# from torch.nn import functional as F
#
#
# class Refiner(nn.Module):
#     def __init__(self, feature_channels=5, width=32):
#         super().__init__()
#         self.context = nn.Conv2d(feature_channels + 1, width, 1)
#         # ONE instance, reused at every refinement step.
#         self.core = nn.TransformerEncoderLayer(
#             d_model=width, nhead=4, dim_feedforward=2 * width,
#             dropout=0.0, batch_first=True, norm_first=True,
#         )
#         self.readout = nn.Linear(width, 1)
#         self.width = width
#
#     def initialize(self, features, initial_logits):
#         b, _, h, w = features.shape
#         return features.new_zeros(b, h * w, self.width)
#
#     def step(self, features, state, previous_logits):
#         h, w = features.shape[-2:]
#         mask = F.interpolate(previous_logits.sigmoid(), size=(h, w),
#                              mode="bilinear", align_corners=False)
#         context = self.context(torch.cat([features, mask], dim=1))
#         context = context.flatten(2).transpose(1, 2)
#         next_state = self.core(state + context)
#         delta = self.readout(next_state).transpose(1, 2).reshape(-1, 1, h, w)
#         delta = F.interpolate(delta, size=previous_logits.shape[-2:],
#                               mode="bilinear", align_corners=False)
#         return next_state, previous_logits + 0.1 * delta
#
#
# def build_refiner(feature_channels):
#     return Refiner(feature_channels=feature_channels)



import torch
from torch import nn
from torch.nn import functional as F


def token_grid(h, w, max_side):
    if max(h, w) <= max_side:
        return h, w
    scale = max_side / max(h, w)
    return max(1, round(h * scale)), max(1, round(w * scale))


def sincos_2d(gh, gw, dim, device, dtype):
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter))
    ys = torch.arange(gh, device=device, dtype=torch.float32)[:, None] * omega
    xs = torch.arange(gw, device=device, dtype=torch.float32)[:, None] * omega
    pe_y = torch.cat([ys.sin(), ys.cos()], dim=1)[:, None, :].expand(gh, gw, -1)
    pe_x = torch.cat([xs.sin(), xs.cos()], dim=1)[None, :, :].expand(gh, gw, -1)
    return torch.cat([pe_y, pe_x], dim=-1).reshape(1, gh * gw, dim).to(dtype)


class FeatureTokenizer(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.dim = dim

    def forward(self, features, grid):
        x = F.adaptive_avg_pool2d(features, grid)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + sincos_2d(*grid, self.dim, x.device, x.dtype)


class MaskTokenizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(2, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, logits, grid):
        p = F.adaptive_avg_pool2d(logits.sigmoid(), grid)
        x = torch.cat([p, 4 * p * (1 - p)], dim=1)
        return self.proj(x).flatten(2).transpose(1, 2)


class Fuser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(3 * dim)
        self.proj = nn.Linear(3 * dim, dim)

    def forward(self, state, x_tokens, m_tokens):
        update = self.proj(self.norm(torch.cat([state, x_tokens, m_tokens], dim=-1)))
        return state + update


class Core(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=4 * dim,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return self.norm(z)


class Memory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.init = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, x_tokens):
        return self.init(x_tokens)


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, state, grid, out_size):
        delta = self.mlp(state).transpose(1, 2).reshape(-1, 1, *grid)
        return F.interpolate(delta, size=out_size, mode="bilinear", align_corners=False)


class Refiner(nn.Module):
    def __init__(self, feature_channels=5, dim=64, depth=2, heads=4, max_tokens_side=32):
        super().__init__()
        if dim % 4 or dim % heads:
            raise ValueError("dim must be divisible by 4 and by heads")
        self.tx = FeatureTokenizer(feature_channels, dim)
        self.ty = MaskTokenizer(dim)
        self.encoder = Fuser(dim)
        self.core = Core(dim, depth, heads)
        self.memory = Memory(dim)
        self.head = Head(dim)
        self.max_tokens_side = max_tokens_side

    def _grid(self, features):
        return token_grid(*features.shape[-2:], self.max_tokens_side)

    def initialize(self, features, initial_logits):
        grid = self._grid(features)
        return self.memory(self.tx(features, grid))

    def step(self, features, state, previous_logits):
        grid = self._grid(features)
        x_tokens = self.tx(features, grid)
        m_tokens = self.ty(previous_logits, grid)
        fused = self.encoder(state, x_tokens, m_tokens)
        next_state = self.core(fused)
        delta = self.head(next_state, grid, previous_logits.shape[-2:])
        return next_state, previous_logits + delta


def build_refiner(feature_channels):
    return Refiner(feature_channels=feature_channels)
