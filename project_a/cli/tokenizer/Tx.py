import torch
from torch import nn
from torch.nn import functional as F


def pos2d(h, w, d, device=None):
    assert d % 4 == 0
    y = torch.arange(h, device=device).repeat_interleave(w)
    x = torch.arange(w, device=device).repeat(h)
    omega = 1.0 / (10000 ** (torch.arange(d // 4, device=device) / (d // 4)))
    x = x[:, None] * omega[None]
    y = y[:, None] * omega[None]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)

class Tx(nn.Module):

    def __init__(self, c_in, d=256) -> None:
        super().__init__()
        assert d % 4 == 0
        self.proj = nn.Conv2d(c_in, d, kernel_size=1)
        self.norm = nn.LayerNorm(d)

    def forward(self, f):
        x = self.proj(f)
        B, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + pos2d(h, w, d, x.device)
        x = self.norm(x)
        return x


if __name__ == "__main__":
    model = Tx(c_in=64, d=256)
    f = torch.randn(2, 64, 8, 10)
    out = model(f)
    print("input :", tuple(f.shape))
    print("output:", tuple(out.shape))
    print("pos2d :", tuple(pos2d(8, 10, 256).shape))
