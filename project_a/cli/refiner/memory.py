import torch
from torch import nn


class Initializer(nn.Module):

    def __init__(self, d=256, nz=32, nhead=8) -> None:
        super().__init__()
        self.nz = nz
        self.queries = nn.Parameter(torch.randn(1, nz, d) * 0.02)
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, img_tokens, ans_tokens):
        B = img_tokens.shape[0]
        q = self.norm_q(self.queries.expand(B, -1, -1))
        kv = self.norm_kv(torch.cat([img_tokens, ans_tokens], dim=1))
        z, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm_out(q + z)


if __name__ == "__main__":
    init = Initializer(d=256, nz=32)
    img = torch.randn(2, 80, 256)
    ans = torch.randn(2, 80, 256)
    z0 = init(img, ans)
    print("img tokens:", tuple(img.shape))
    print("ans tokens:", tuple(ans.shape))
    print("z0        :", tuple(z0.shape))
