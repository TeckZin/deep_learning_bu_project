"""Recurrent token refiner for EC523 Project A (road segmentation).

Everything the refiner needs is inside submission/ (shipped as one package):
    config.py        architecture presets (ACTIVE = the one build_refiner() uses by default)
    tokenizers.py    (f, y_t) -> one fused token per patch            doc stages 5, 7, 8
    memory.py        grid -> z_0 (memory initialization)              doc stage 6
    core.py          the ONE shared transformer, model.core           doc stages 9-11
    output_heads.py  grid tokens -> full-resolution correction dy     doc stages 12-13
    pos.py           2D position code

    state = model.initialize(features, y_0)                # z_0 = I(f, y_0)
    repeat K times (the caller owns the loop, the model never sees K):
        state, y = model.step(features, state, y)          # (z_t+1, y_t+1) = R(f, z_t, y_t)

One step:
    grid  = Tokenize(f, y_t)
    [grid', z_t+1] = core([grid, z_t])                     # ONE core call
    y_t+1 = y_t + Head(grid', y_t)                         # a correction, not a new mask

Rules required by predict.py / the tests:
  * build_refiner(feature_channels) returns the model; every layer is created in __init__.
  * model.core is one nn.Module, called exactly once per step(); initialize() never calls it.
  * pos_offset is a plain attribute (not a parameter or buffer, not in the checkpoint).
    train.py sets it to the crop position; inference never touches it (None = full tile).
"""
import torch
from torch import nn

from .config import config_from_metadata, get_config
from .core import Core
from .memory import MEMORIES
from .output_heads import HEADS
from .tokenizers import TOKENIZERS


class Refiner(nn.Module):
    def __init__(self, c_in, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = cfg.patch
        c = cfg.core
        # Attribute names (tok, init, core, head) are the checkpoint key prefixes: keep them.
        self.tok = TOKENIZERS[cfg.tokenizer.kind](c_in, cfg.d, cfg.patch, cfg.tokenizer)
        self.init = MEMORIES[cfg.memory.kind](cfg.d, c.heads, cfg.memory)
        self.core = Core(cfg.d, c.heads, c.layers, dim_ff=c.ff_mult * cfg.d, dropout=c.dropout,
                         mode=c.mode, n_mem=cfg.memory.nz)
        self.head = HEADS[cfg.head.kind](cfg.d, cfg.patch, cfg.head)
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


def build_refiner(feature_channels, cfg=None):
    """Entry point used by predict.py, infer.py, train.py and make_demo.py.

    cfg=None -> the ACTIVE preset in submission/config.py (the staff runner passes nothing).
    """
    return Refiner(c_in=feature_channels, cfg=cfg or get_config())


def load_refiner(path, feature_channels, device="cpu", prefix=""):
    """Build the refiner a checkpoint was trained with (config from its metadata) and load it.

    prefix: key prefix inside a bundled file (infer.py's submission weights use "refiner.").
    Checkpoints without stored config fall back to the ACTIVE preset.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata() or {}
    model = build_refiner(feature_channels, config_from_metadata(meta))
    state = load_file(str(path), device=str(device))
    if prefix:
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    return model.to(device)


if __name__ == "__main__":
    # From the cli folder:  python -m submission.model
    from .config import PRESETS

    for name, cfg in PRESETS.items():
        model = build_refiner(5, cfg).eval()
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
        big = build_refiner(384, cfg)
        print(f"{name:10s} y4 {tuple(ys[-1].shape)}  prefix K2==K4 {torch.allclose(y2, ys[1], atol=1e-6)}"
              f"  changes y {not torch.equal(ys[0], y0)}"
              f"  params@384 {sum(p.numel() for p in big.parameters()):,}"
              f" (core {sum(p.numel() for p in big.core.parameters()):,})")
