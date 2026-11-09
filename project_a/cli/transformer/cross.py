"""CrossBlock: one pre-norm transformer block with CROSS-attention.

Tokens x ask questions of a DIFFERENT sequence `context`
(Q from x, K and V from context). The output has x's length, not context's.

    x -> LayerNorm(x), LayerNorm(context) -> cross-attention -> + x
      -> LayerNorm -> feed-forward                           -> + x

Input  x [B, Nq, d], context [B, Nk, d]  ->  output [B, Nq, d]
Run the shape check from project_a/cli with:  python -m transformer.cross
"""
import torch
from torch import nn

from .attention import FeedForward, MultiHeadAttention


class CrossBlock(nn.Module):
    def __init__(self, d=256, nhead=8, dim_ff=None, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d)          # normalizes the asking tokens
        self.norm_kv = nn.LayerNorm(d)         # normalizes the tokens being read
        self.attn = MultiHeadAttention(d, nhead, dropout)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ff = FeedForward(d, dim_ff, dropout)

    def forward(self, x, context):
        x = x + self.drop(self.attn(self.norm_q(x), self.norm_kv(context)))   # cross: Q from x, K/V from context
        x = x + self.ff(self.norm2(x))
        return x


if __name__ == "__main__":
    block = CrossBlock(d=256, nhead=8)
    mem = torch.randn(2, 32, 256)              # 32 memory tokens ask...
    grid = torch.randn(2, 2048, 256)           # ...2048 image + answer tokens
    print("x      :", tuple(mem.shape))        # (2, 32, 256)
    print("context:", tuple(grid.shape))       # (2, 2048, 256)
    print("out    :", tuple(block(mem, grid).shape))   # (2, 32, 256)
    print("params :", sum(p.numel() for p in block.parameters()))
