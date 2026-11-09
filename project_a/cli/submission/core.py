"""Core: the shared recurrent transformer, exposed as model.core.

Built only from PyTorch's own transformer layers. The same Core instance is
applied once per refinement step, so its weights are reused K times;
predict.py checks it is called exactly once per step().

Input and output are the same sequence model.py builds:
    tokens = [grid tokens (N), memory tokens (n_mem)]      [B, N + n_mem, d]
(one fused image+answer token per patch, then the memory z_t).

mode="self"  (default, recommended)
    nn.TransformerEncoder: every token attends to every token, so each patch sees
    its neighbours, far-away road pieces and the memory directly.
    Cost per layer ~ (N + n_mem)^2: a full 1024x1024 tile at patch 16 is
    4096 + 32 tokens, which fits easily on one L40S with PyTorch's fused attention.

mode="cross" (cheaper, weaker)
    memory self-attends and reads the grid (nn.TransformerDecoderLayer, num_layers//2 times),
    then the grid reads the memory back (nn.MultiheadAttention + feed-forward).
    Grid tokens never attend to each other directly, only through the memory.
"""
import torch
from torch import nn


class Core(nn.Module):
    def __init__(self, d=256, nhead=8, num_layers=4, dim_ff=None, dropout=0.05,
                 mode="self", n_mem=None) -> None:
        super().__init__()
        if mode not in ("self", "cross"):
            raise ValueError("mode must be 'self' or 'cross'")
        if mode == "cross" and not n_mem:
            raise ValueError("mode='cross' needs n_mem (number of memory tokens at the end)")
        self.mode = mode
        self.n_mem = n_mem
        dim_ff = dim_ff or 4 * d
        common = dict(d_model=d, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout,
                      activation="gelu", batch_first=True, norm_first=True)

        if mode == "self":
            # num_layers independent layers inside ONE core (a fixed multi-layer core
            # is allowed); the whole stack is reused at every step.
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(**common), num_layers=num_layers,
                enable_nested_tensor=False)
        else:
            pairs = max(1, num_layers // 2)
            self.grid_norm = nn.LayerNorm(d)          # DecoderLayer does not normalize `memory`
            self.think_read = nn.ModuleList(
                [nn.TransformerDecoderLayer(**common) for _ in range(pairs)])
            self.write_norm_q = nn.LayerNorm(d)
            self.write_norm_kv = nn.LayerNorm(d)
            self.write_attn = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
            self.write_drop = nn.Dropout(dropout)
            self.write_norm_ff = nn.LayerNorm(d)
            self.write_ff = nn.Sequential(
                nn.Linear(d, dim_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim_ff, d), nn.Dropout(dropout))
        # Final norm keeps the carried state bounded however many steps are unrolled.
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens):
        if self.mode == "self":
            return self.norm(self.encoder(tokens))

        grid = tokens[:, :-self.n_mem]                   # [B, N, d]
        mem = tokens[:, -self.n_mem:]                    # [B, n_mem, d]
        grid_kv = self.grid_norm(grid)
        for layer in self.think_read:
            mem = layer(tgt=mem, memory=grid_kv)         # memory: self-attn, then reads the grid
        q, kv = self.write_norm_q(grid), self.write_norm_kv(mem)
        attn, _ = self.write_attn(q, kv, kv, need_weights=False)
        grid = grid + self.write_drop(attn)              # grid reads the memory back
        grid = grid + self.write_ff(self.write_norm_ff(grid))
        return self.norm(torch.cat([grid, mem], dim=1))


if __name__ == "__main__":
    # From the cli folder:  python -m submission.core
    tokens = torch.randn(2, 64 + 32, 256)
    for mode in ("self", "cross"):
        core = Core(d=256, mode=mode, n_mem=32).eval()
        print(f"{mode:5s}: {tuple(tokens.shape)} -> {tuple(core(tokens).shape)}, "
              f"params {sum(p.numel() for p in core.parameters()):,}")
