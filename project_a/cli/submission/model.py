"""Recurrent token refiner for EC523 Project A (road segmentation).

Uses our own modules (in project_a/cli/):
    tokenizer/Tx.py       Tx: features -> image tokens (+ 2D position)
    tokenizer/Ty.py       Ty: mask y_t -> answer tokens (patch conv + 2D position)
    refiner/memory.py     Initializer: nz learned memory tokens read image + y_0 -> z_0
    transformer/core.py   Core: shared transformer (model.core), mode "self" or "cross"
    refiner/head.py       Head: answer tokens -> full-resolution correction dy

This is refiner/refiner.py rewritten for the staff runner: predict.py owns the
loop and never passes K, so the setup becomes initialize() and the loop body
becomes step().

    state = model.initialize(features, y_0)                # z_0 = I(f, y_0)
    repeat K times:
        state, y = model.step(features, state, y)          # (z_t+1, y_t+1) = R(f, z_t, y_t)

One step:
    [image tokens, answer tokens, memory z_t] --core (ONE call)--> [_, answer', z_t+1]
    y_t+1 = y_t + Head(answer')

core_mode="self"  : joint self-attention over all tokens (default)
core_mode="cross" : memory reads the grid (cross), thinks (self), grid reads memory back (cross)

Rules required by predict.py / the tests:
  * build_refiner(feature_channels) returns the model.
  * model.core is one nn.Module, called exactly once per step().
  * initialize() never calls core.
  * The model never sees K, and every layer is created in __init__.
"""
import torch
from torch import nn
from torch.nn import functional as F

from tokenizer.Tx import Tx
from tokenizer.Ty import Ty
from transformer.core import Core
from refiner.memory import Initializer
from refiner.head import Head


class Refiner(nn.Module):
    def __init__(self, c_in, d=256, patch=32, nz=32, nhead=4, num_layers=4, dropout=0.1,
                 core_mode="self") -> None:
        super().__init__()
        # patch = pixels per token side. Ty and Head both use it, and the image
        # tokens are pooled to the same grid, so all three always line up.
        # 1024x1024 tile, patch 32 -> 32x32 grid -> 1024 image + 1024 answer tokens.
        self.patch = patch
        self.tx = Tx(c_in, d)
        self.ty = Ty(d, patch)
        self.init = Initializer(d, nz, nhead)
        # The ONE shared core. n_mem=nz: the last nz tokens of the sequence are memory
        # (step() concatenates [image, answer, memory]); only used by core_mode="cross".
        self.core = Core(d, nhead, num_layers, dropout=dropout, mode=core_mode, n_mem=nz)
        self.head = Head(d, patch)

    def _image_tokens(self, features, y):
        """Pool features to Ty's grid (H/patch x W/patch), then tokenize with Tx.

        Works for any baseline stride: the features are resized to match the
        mask's patch grid instead of requiring patch = baseline stride.
        """
        H, W = y.shape[-2:]
        h, w = H // self.patch, W // self.patch
        f = F.adaptive_avg_pool2d(features, (h, w))    # [B, C, h, w]
        return self.tx(f), h, w                         # [B, h*w, d]

    def initialize(self, features, initial_logits):
        """z_0 = I(f, y_0). Called once before step 1; does not call the core."""
        img, _, _ = self._image_tokens(features, initial_logits)
        ans = self.ty(initial_logits)                   # [B, h*w, d]
        return self.init(img, ans)                      # [B, nz, d]

    def step(self, features, state, previous_logits):
        """One refinement step: (z_t, y_t) -> (z_t+1, y_t+1)."""
        H, W = previous_logits.shape[-2:]
        img, h, w = self._image_tokens(features, previous_logits)   # [B, N, d]
        ans = self.ty(previous_logits)                              # [B, N, d]
        n, nz = img.shape[1], state.shape[1]
        if ans.shape[1] != n:
            raise ValueError(f"Tx grid {n} tokens != Ty grid {ans.shape[1]} tokens")
        out = self.core(torch.cat([img, ans, state], dim=1))       # [B, 2N+nz, d], one core call
        _, ans_out, next_state = out.split([n, n, nz], dim=1)
        dy = self.head(ans_out, h, w, out_size=(H, W))              # [B, 1, H, W]
        return next_state, previous_logits + dy         # correction, not a new mask


CORE_MODE = "cross"   # "self" or "cross"; changing it changes the checkpoint layout, so retrain


def build_refiner(feature_channels):
    """Entry point used by predict.py and train.py."""
    return Refiner(c_in=feature_channels, core_mode=CORE_MODE)
