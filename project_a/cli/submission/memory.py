"""State initializers (doc stage 6): grid tokens -> z_0. Must never call the core."""
import torch
from torch import nn
from torch.nn import functional as F


class MemoryInitializer(nn.Module):
    """grid tokens [B, N, d] -> z_0 [B, nz, d]: nz learned memory slots read the grid once."""

    def __init__(self, d, nhead, cfg) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, cfg.nz, d) * 0.02)      # learned memory slots
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, grid):
        q = self.norm_q(self.queries.expand(grid.shape[0], -1, -1))         # [B, nz, d]
        kv = self.norm_kv(grid)
        z, _ = self.attn(q, kv, kv, need_weights=False)                     # slots read the grid
        return self.norm_out(q + z)


class SpatialMemory(nn.Module):
    """(frozen features x, y_0) -> z_0 [B, Cz, H/s, W/s]: memory maps "parallel to y" (attempt10).

        z_0 = tanh(Conv3x3([up(Conv1x1(LN(x))), p_0]))
    Made from the encoder's features and H0's answer, so step 1 starts from an informed memory
    (as RAFT starts its hidden state from its context network). tanh keeps it in the range the
    head's GRU update produces.
    """

    def __init__(self, c_x, cfg) -> None:
        super().__init__()
        self.stride = cfg.stride
        self.norm = nn.LayerNorm(c_x)
        self.proj = nn.Conv2d(c_x, cfg.channels, 1)
        self.mix = nn.Conv2d(cfg.channels + 1, cfg.channels, 3, padding=1)

    def forward(self, x, logits):
        H, W = logits.shape[-2:]
        size = (H // self.stride, W // self.stride)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)          # LayerNorm over channels
        x = F.interpolate(self.proj(x), size=size, mode="bilinear", align_corners=False)
        p = F.adaptive_avg_pool2d(logits.sigmoid(), size)
        # float32 even under autocast: z is carried across K steps, like y
        return torch.tanh(self.mix(torch.cat([x, p], dim=1))).float()


MEMORIES = {"slots": MemoryInitializer}
