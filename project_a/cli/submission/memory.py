"""State initializers (doc stage 6): grid tokens -> z_0. Must never call the core."""
import torch
from torch import nn


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


MEMORIES = {"slots": MemoryInitializer}
