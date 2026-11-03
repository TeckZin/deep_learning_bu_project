"""Core: the shared recurrent transformer (exposed as model.core).

The same Core instance is applied once per refinement step, so its weights are
reused K times. predict.py checks it is called exactly once per step().
"""
import torch
from torch import nn


class Core(nn.Module):
    """Tokens [B, N, d] -> updated tokens [B, N, d]."""

    def __init__(self, d=256, nhead=8, num_layers=4, dim_ff=None, dropout=0.1) -> None:
        super().__init__()
        assert d % nhead == 0  # each attention head gets d / nhead channels
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=dim_ff or 4 * d,
            dropout=dropout,          # active in training only; predict.py runs model.eval()
            activation="gelu",
            batch_first=True,         # inputs are [B, N, d]
            norm_first=True,          # pre-norm: more stable when stacked / unrolled
        )
        # num_layers deep copies of `layer`, run in sequence. Wrapping them in one
        # module means the forward hook in predict.py counts one call per step.
        # Nested tensors are disabled so outputs are plain dense tensors.
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        # Final norm keeps the state bounded no matter how many steps are unrolled.
        self.norm = nn.LayerNorm(d)

    def forward(self, tokens):
        return self.norm(self.encoder(tokens))


if __name__ == "__main__":
    # Shape check: token count and width are unchanged.
    core = Core(d=256)
    tokens = torch.randn(2, 192, 256)
    out = core(tokens)
    print("input :", tuple(tokens.shape))
    print("output:", tuple(out.shape))
