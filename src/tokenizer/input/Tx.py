import torch
from torch import nn
from torch.nn import functional as F


class Tx(nn.Module):

    def __init__(self, c_in, d=256) -> None:
        super().__init__()
        assert d % 4 == 0
        self.proj = nn.Conv2d(c_in, d, kernel_size=1)
        self.norm = nn.LayerNorm(d)


    def forward(self, f):
        x = self.proj(f)
        B, d, h, w = x.shape()
        x = x.flatten(2).tranpose(1,2)
        print("forward loop")



