import torch
from torch import nn
from tokenizer.Tx import pos2d

class Ty(nn.Module):

    def __init__(self, d=256, patch=8) -> None:
        super().__init__()
        assert d % 4 == 0
        self.patch = patch
        self.proj = nn.Conv2d(1, d, kernel_size=patch, stride=patch)
        self.norm = nn.LayerNorm(d)

    def forward(self, y):
        x = self.proj(y)
        B, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + pos2d(h, w, d, x.device)
        x = self.norm(x)
        return x


if __name__ == "__main__":
    model = Ty(d=256, patch=8)
    y = torch.randn(2, 1, 64, 80)
    out = model(y)
    print("input :", tuple(y.shape))
    print("output:", tuple(out.shape))
    print("params:", sum(p.numel() for p in model.parameters()))
