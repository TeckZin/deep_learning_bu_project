"""Initializer: builds the initial latent state z_0.

The state is a small bank of nz learned latent tokens [B, nz, d]. They read the
image tokens (from Tx) and the initial-answer tokens (from Ty) with
cross-attention, so z_0 already summarizes the image and the baseline's guess.
This runs inside model.initialize(), which must NOT call the core.
"""
import torch
from torch import nn


class Initializer(nn.Module):
    """(img_tokens [B, N, d], ans_tokens [B, M, d]) -> z_0 [B, nz, d]."""

    def __init__(self, d=256, nz=32, nhead=8) -> None:
        super().__init__()
        self.nz = nz
        # nz learned query vectors shared across images (small init for stability).
        self.queries = nn.Parameter(torch.randn(1, nz, d) * 0.02)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, img_tokens, ans_tokens):
        B = img_tokens.shape[0]
        # Same queries for every image in the batch.
        q = self.norm_q(self.queries.expand(B, -1, -1))                   # [B, nz, d]
        # Keys/values: all image tokens and all answer tokens together.
        kv = self.norm_kv(torch.cat([img_tokens, ans_tokens], dim=1))     # [B, N+M, d]
        # Each latent attends over the whole image + current answer.
        z, _ = self.attn(q, kv, kv, need_weights=False)                  # [B, nz, d]
        # Residual + norm: z_0 = norm(q + read-out).
        return self.norm_out(q + z)


if __name__ == "__main__":
    # Shape check: 80 image + 80 answer tokens -> 32 latent tokens.
    init = Initializer(d=256, nz=32)
    img = torch.randn(2, 80, 256)
    ans = torch.randn(2, 80, 256)
    z0 = init(img, ans)
    print("img tokens:", tuple(img.shape))
    print("ans tokens:", tuple(ans.shape))
    print("z0        :", tuple(z0.shape))
