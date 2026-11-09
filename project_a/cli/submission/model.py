"""Recurrent token refiner for EC523 Project A (road segmentation).

Everything the model needs is inside submission/ (parts.py, core.py, this file):
the staff runner only ships and hashes submission/*.py.

    state = model.initialize(features, y_0)                # z_0 = I(f, y_0)
    repeat K times (the runner owns the loop, the model never sees K):
        state, y = model.step(features, state, y)          # (z_t+1, y_t+1) = R(f, z_t, y_t)

One step:
    grid  = Tokenize(f, y_t)                               # one fused image+answer token per patch
    [grid', z_t+1] = core([grid, z_t])                     # ONE core call
    y_t+1 = y_t + Head(grid', y_t)                         # a correction, not a new mask

Rules required by predict.py / the tests:
  * build_refiner(feature_channels) returns the model; every layer is created in __init__.
  * model.core is one nn.Module, called exactly once per step(); initialize() never calls it.
  * pos_offset is a plain attribute (not a parameter or buffer, not in the checkpoint).
    train.py sets it to the crop position; predict.py never touches it, so it stays
    None = full tile at evaluation.
"""
import torch
from torch import nn

from .core import Core
from .parts import GridTokenizer, Head, Initializer


class Refiner(nn.Module):
    def __init__(self, c_in, d=256, patch=16, nz=32, nhead=8, num_layers=4, dropout=0.05,
                 core_mode="self") -> None:
        super().__init__()
        self.patch = patch
        self.tok = GridTokenizer(c_in, d, patch)
        self.init = Initializer(d, nz, nhead)
        self.core = Core(d, nhead, num_layers, dropout=dropout, mode=core_mode, n_mem=nz)
        self.head = Head(d, patch)
        self.pos_offset = None

    def initialize(self, features, initial_logits):
        """z_0 = I(f, y_0). Called once before step 1; does not call the core."""
        grid, _, _ = self.tok(features, initial_logits, self.pos_offset)
        return self.init(grid)                                       # [B, nz, d]

    def step(self, features, state, previous_logits):
        """One refinement step: (z_t, y_t) -> (z_t+1, y_t+1)."""
        grid, h, w = self.tok(features, previous_logits, self.pos_offset)   # [B, N, d]
        n, nz = grid.shape[1], state.shape[1]
        out = self.core(torch.cat([grid, state], dim=1))                    # ONE core call
        grid_out, next_state = out.split([n, nz], dim=1)
        dy = self.head(grid_out, h, w, previous_logits)                     # [B, 1, H, W]
        return next_state, previous_logits + dy


# "self" (recommended) or "cross". Changing it, or any size above, changes the
# checkpoint layout: retrain after changing.
CORE_MODE = "self"


def build_refiner(feature_channels):
    """Entry point used by predict.py, train.py and make_demo.py."""
    return Refiner(c_in=feature_channels, core_mode=CORE_MODE)


if __name__ == "__main__":
    # From the cli folder:  python -m submission.model
    model = build_refiner(5).eval()
    f = torch.randn(2, 5, 4, 5)
    y0 = torch.randn(2, 1, 64, 80)
    with torch.no_grad():
        z, y, ys = model.initialize(f, y0), y0, []
        for _ in range(4):
            z, y = model.step(f, z, y)
            ys.append(y)
        z2, y2 = model.initialize(f, y0), y0
        for _ in range(2):
            z2, y2 = model.step(f, z2, y2)
    print("y4 shape      :", tuple(ys[-1].shape))
    print("prefix K2==K4 :", torch.allclose(y2, ys[1], atol=1e-6))
    print("changes y     :", not torch.equal(ys[0], y0))
    print(f"params        : {sum(p.numel() for p in model.parameters()):,}"
          f"  (core {sum(p.numel() for p in model.core.parameters()):,})")
