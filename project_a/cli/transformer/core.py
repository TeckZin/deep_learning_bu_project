import torch
from torch import nn


class Core(nn.Module):

    def __init__(self, d=256, nhead=8, num_layers=4, dim_ff=None, dropout=0.1) -> None:
        super().__init__()
        assert d % nhead == 0
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

    def forward(self, tokens):
        return self.norm(self.encoder(tokens))


if __name__ == "__main__":
    core = Core(d=256)
    tokens = torch.randn(2, 192, 256)
    out = core(tokens)
    print("input :", tuple(tokens.shape))
    print("output:", tuple(out.shape))
