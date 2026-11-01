import torch
from torch import nn

from Tx import Tx


class Ty(nn.Module):

    def __init__(self, c_in, d=256, nhead=8, num_layers=2, dim_ff=None, dropout=0.1) -> None:
        super().__init__()
        assert d % nhead == 0
        self.tx = Tx(c_in, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=dim_ff or 4 * d,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, f):
        B, _, h, w = f.shape
        x = self.tx(f)
        x = self.encoder(x)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(B, -1, h, w)


if __name__ == "__main__":
    model = Ty(c_in=64, d=256)
    f = torch.randn(2, 64, 8, 10)
    out = model(f)
    print("input :", tuple(f.shape))
    print("tokens:", tuple(model.tx(f).shape))
    print("output:", tuple(out.shape))
    print("params:", sum(p.numel() for p in model.parameters()))
