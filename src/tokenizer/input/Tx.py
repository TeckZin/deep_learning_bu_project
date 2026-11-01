import torch
from torch import nn


def pos2d(h, w, d, device=None):
    """Return deterministic sine/cosine positions in row-major grid order."""
    assert d % 4 == 0
    y = torch.arange(h, device=device).repeat_interleave(w)
    x = torch.arange(w, device=device).repeat(h)
    omega = 1.0 / (10000 ** (torch.arange(d // 4, device=device) / (d // 4)))
    x = x[:, None] * omega[None]
    y = y[:, None] * omega[None]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)

class Tx(nn.Module):
    """Teck's image tokenizer: BCHW features to B(HW)d spatial tokens."""

    def __init__(self, c_in, d=256) -> None:
        super().__init__()
        assert d % 4 == 0
        self.proj = nn.Conv2d(c_in, d, kernel_size=1)
        self.norm = nn.LayerNorm(d)

    def forward(self, f):
        x = self.proj(f)
        _, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        # Avoid promoting AMP activations back to float32 throughout the core.
        x = x + pos2d(h, w, d, x.device).to(dtype=x.dtype)
        x = self.norm(x)
        return x
