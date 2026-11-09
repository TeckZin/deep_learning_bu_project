"""SelfBlock: one pre-norm transformer block with SELF-attention.

Every token attends to every token in the SAME sequence (Q, K, V all from x).

    x -> LayerNorm -> self-attention -> + x
      -> LayerNorm -> feed-forward   -> + x

Input  x [B, N, d]  ->  output [B, N, d]
Run the shape check from project_a/cli with:  python -m transformer.self
"""
import torch
from torch import nn

from .attention import FeedForward, MultiHeadAttention


class SelfBlock(nn.Module):
    def __init__(self, d=256, nhead=8, dim_ff=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, nhead, dropout)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ff = FeedForward(d, dim_ff, dropout)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.drop(self.attn(h, h))     # self: query and context are the same tokens
        x = x + self.ff(self.norm2(x))
        return x


if __name__ == "__main__":
    block = SelfBlock(d=256, nhead=8)
    x = torch.randn(2, 100, 256)
    print("in :", tuple(x.shape))              # (2, 100, 256)
    print("out:", tuple(block(x).shape))       # (2, 100, 256)
    print("params:", sum(p.numel() for p in block.parameters()))
