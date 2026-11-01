import torch
from torch import nn

from tokenizer.Tx import Tx
from tokenizer.Ty import Ty
from transformer.core import Core
from refiner.memory import Initializer
from refiner.head import Head


class Refiner(nn.Module):

    def __init__(self, c_in, d=256, patch=16, nz=32, nhead=8, num_layers=4, dropout=0.1) -> None:
        super().__init__()
        self.patch = patch
        self.tx = Tx(c_in, d)
        self.ty = Ty(d, patch)
        self.init = Initializer(d, nz, nhead)
        self.core = Core(d, nhead, num_layers, dropout=dropout)
        self.head = Head(d, patch)

    def forward(self, f, y0, K, return_all=False):
        if K == 0:
            return [] if return_all else y0
        H, W = y0.shape[-2:]
        h, w = H // self.patch, W // self.patch
        img = self.tx(f)
        N = img.shape[1]
        assert N == h * w, f"Tx grid {tuple(f.shape[-2:])} does not match Ty grid {(h, w)}; set patch = H / h"
        yt = y0
        zt = self.init(img, self.ty(y0))
        Nz = zt.shape[1]
        ys = []
        for _ in range(K):
            ans = self.ty(yt)
            out = self.core(torch.cat([img, ans, zt], dim=1))
            _, ans_out, zt = out.split([N, N, Nz], dim=1)
            yt = yt + self.head(ans_out, h, w, out_size=(H, W))
            ys.append(yt)
        return ys if return_all else yt


if __name__ == "__main__":
    model = Refiner(c_in=64, d=256, patch=8).eval()
    f = torch.randn(2, 64, 8, 10)
    y0 = torch.randn(2, 1, 64, 80)
    with torch.no_grad():
        y0_out = model(f, y0, 0)
        y2 = model(f, y0, 2)
        y4_all = model(f, y0, 4, return_all=True)
    print("f      :", tuple(f.shape))
    print("y0     :", tuple(y0.shape))
    print("K=0 identity  :", torch.equal(y0_out, y0))
    print("K=2 output    :", tuple(y2.shape))
    print("K=4 steps     :", len(y4_all))
    print("prefix K2==K4 :", torch.allclose(y2, y4_all[1], atol=1e-6))
    print("params        :", sum(p.numel() for p in model.parameters()))
