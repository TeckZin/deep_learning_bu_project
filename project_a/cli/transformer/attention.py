"""Shared building pieces for self.py and cross.py.

MultiHeadAttention: the attention itself, written out (Q, K, V, split into heads).
FeedForward:        the per-token MLP that follows attention in every block.
"""
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_h)) V, split across `nhead` heads.

    query   [B, Nq, d]  the tokens asking (Q comes from here)
    context [B, Nk, d]  the tokens being read (K and V come from here)
    returns [B, Nq, d]  one output per query token

    Self-attention : forward(x, x)        -> Q, K, V all from x
    Cross-attention: forward(x, context)  -> Q from x, K and V from context
    """

    def __init__(self, d, nhead, dropout=0.0):
        super().__init__()
        assert d % nhead == 0, "d must be divisible by nhead"
        self.nhead = nhead
        self.dh = d // nhead              # channels per head, e.g. 256 / 8 = 32
        self.q = nn.Linear(d, d)          # W_Q
        self.k = nn.Linear(d, d)          # W_K
        self.v = nn.Linear(d, d)          # W_V
        self.out = nn.Linear(d, d)        # mixes the heads back together
        self.dropout = dropout

    def _split(self, x):
        # [B, N, d] -> [B, nhead, N, dh]: each head gets its own dh channels
        B, N, _ = x.shape
        return x.view(B, N, self.nhead, self.dh).transpose(1, 2)

    def forward(self, query, context):
        q = self._split(self.q(query))    # [B, h, Nq, dh]
        k = self._split(self.k(context))  # [B, h, Nk, dh]
        v = self._split(self.v(context))  # [B, h, Nk, dh]
        # softmax(q k^T / sqrt(dh)) v for every head at once (memory-efficient kernel).
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)   # [B, h, Nq, dh]
        B, h, Nq, dh = out.shape
        out = out.transpose(1, 2).reshape(B, Nq, h * dh)                   # heads side by side: [B, Nq, d]
        return self.out(out)


class FeedForward(nn.Module):
    """Per-token MLP: d -> dim_ff -> d. No mixing between tokens."""

    def __init__(self, d, dim_ff=None, dropout=0.0):
        super().__init__()
        dim_ff = dim_ff or 4 * d
        self.net = nn.Sequential(
            nn.Linear(d, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
