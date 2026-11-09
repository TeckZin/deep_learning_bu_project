"""Core: the shared recurrent transformer (exposed as model.core).

The same Core instance is applied once per refinement step, so its weights are
reused K times. predict.py checks it is called exactly once per step().

Input and output are the same joint sequence model.py builds:
    tokens = [image tokens (N), answer tokens (N), memory tokens (n_mem)]   [B, 2N + n_mem, d]

Two modes (same input/output shape, so model.py's split works for both):

mode="self"  (default) -- joint self-attention
    num_layers SelfBlocks over the whole sequence. Every token attends to every
    token: answer<->image, answer<->answer, memory<->everything.
    Cost per layer ~ (2N + n_mem)^2.

mode="cross" -- memory bottleneck (TRM / Perceiver style)
    read : memory cross-attends to the grid tokens   (CrossBlock, n_mem x 2N)
    think: memory self-attends                        (SelfBlock,  n_mem x n_mem)
    ... repeated num_layers // 2 times, then
    write: grid tokens cross-attend to the memory     (CrossBlock, 2N x n_mem)
    Much cheaper; answer tokens only talk to each other through the memory.
"""
import torch
from torch import nn

from .cross import CrossBlock
from .self import SelfBlock


class Core(nn.Module):
    def __init__(self, d=256, nhead=8, num_layers=4, dim_ff=None, dropout=0.1,
                 mode="self", n_mem=None) -> None:
        super().__init__()
        if mode not in ("self", "cross"):
            raise ValueError("mode must be 'self' or 'cross'")
        if mode == "cross" and not n_mem:
            raise ValueError("mode='cross' needs n_mem (number of memory tokens at the end of the sequence)")
        self.mode = mode
        self.n_mem = n_mem
        if mode == "self":
            self.blocks = nn.ModuleList(
                [SelfBlock(d, nhead, dim_ff, dropout) for _ in range(num_layers)])
        else:
            pairs = max(1, num_layers // 2)     # 4 layers -> 2 (read, think) pairs + 1 write
            self.read = nn.ModuleList([CrossBlock(d, nhead, dim_ff, dropout) for _ in range(pairs)])
            self.think = nn.ModuleList([SelfBlock(d, nhead, dim_ff, dropout) for _ in range(pairs)])
            self.write = CrossBlock(d, nhead, dim_ff, dropout)
        # Final norm keeps the state bounded no matter how many steps are unrolled.
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens):
        if self.mode == "self":
            for block in self.blocks:
                tokens = block(tokens)                  # everyone attends to everyone
            return self.norm(tokens)

        grid = tokens[:, :-self.n_mem]                  # image + answer tokens  [B, 2N, d]
        mem = tokens[:, -self.n_mem:]                   # memory tokens          [B, n_mem, d]
        for read, think in zip(self.read, self.think):
            mem = read(mem, grid)                       # memory reads the grid (cross)
            mem = think(mem)                            # memory reasons alone (self)
        grid = self.write(grid, mem)                    # grid reads the memory back (cross)
        return self.norm(torch.cat([grid, mem], dim=1)) # same layout as the input


if __name__ == "__main__":
    # Run from project_a/cli with:  python -m transformer.core
    tokens = torch.randn(2, 2 * 64 + 32, 256)           # [image 64, answer 64, memory 32]
    for mode in ("self", "cross"):
        core = Core(d=256, mode=mode, n_mem=32)
        out = core(tokens)
        print(f"{mode:5s}: in {tuple(tokens.shape)} -> out {tuple(out.shape)}, "
              f"params {sum(p.numel() for p in core.parameters()):,}")
