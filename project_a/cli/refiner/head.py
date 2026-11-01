import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):

    def __init__(self, d=256, patch=16) -> None:
        super().__init__()
        self.patch = patch
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, patch * patch)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens, h, w, out_size=None):
        x = self.proj(self.norm(tokens))
        B = x.shape[0]
        x = x.transpose(1, 2).reshape(B, self.patch * self.patch, h, w)
        x = F.pixel_shuffle(x, self.patch)
        if out_size is not None and tuple(x.shape[-2:]) != tuple(out_size):
            x = F.interpolate(x, size=tuple(out_size), mode="bilinear", align_corners=False)
        return x


if __name__ == "__main__":
    head = Head(d=256, patch=8)
    tokens = torch.randn(2, 80, 256)
    dy = head(tokens, 8, 10, out_size=(64, 80))
    print("tokens:", tuple(tokens.shape))
    print("dy    :", tuple(dy.shape))
