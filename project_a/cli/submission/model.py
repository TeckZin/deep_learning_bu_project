# """UNTRAINED interface demonstration, not a competitive baseline or solution.
#
# Replace this package with your refiner. Keep the three-method interface and
# expose one nn.Module as `core`. The staff runner, not this class, owns K.
# State can be a tensor or a nested structure. All trainable modules must be
# registered on the refiner before checkpoint loading. No lazy K-dependent layers.
#
# Handout notation: `state` is z_t, a [B,N,d] bank of latent vector tokens;
# `previous_logits` is y_t, a [B,1,H,W] dense answer map (NOT ground truth).
# This example embeds image features and downsampled sigmoid(y_t), then fuses
# them into token context. The core consumes [B,N,d], not full-resolution pixels.
# The readout decodes z_(t+1) to a logit correction and returns y_(t+1).
# """
# import torch
# from torch import nn
# from torch.nn import functional as F
#
#
# class Refiner(nn.Module):
#     def __init__(self, feature_channels=5, width=32):
#         super().__init__()
#         self.context = nn.Conv2d(feature_channels + 1, width, 1)
#         # ONE instance, reused at every refinement step.
#         self.core = nn.TransformerEncoderLayer(
#             d_model=width, nhead=4, dim_feedforward=2 * width,
#             dropout=0.0, batch_first=True, norm_first=True,
#         )
#         self.readout = nn.Linear(width, 1)
#         self.width = width
#
#     def initialize(self, features, initial_logits):
#         b, _, h, w = features.shape
#         return features.new_zeros(b, h * w, self.width)
#
#     def step(self, features, state, previous_logits):
#         h, w = features.shape[-2:]
#         mask = F.interpolate(previous_logits.sigmoid(), size=(h, w),
#                              mode="bilinear", align_corners=False)
#         context = self.context(torch.cat([features, mask], dim=1))
#         context = context.flatten(2).transpose(1, 2)
#         next_state = self.core(state + context)
#         delta = self.readout(next_state).transpose(1, 2).reshape(-1, 1, h, w)
#         delta = F.interpolate(delta, size=previous_logits.shape[-2:],
#                               mode="bilinear", align_corners=False)
#         return next_state, previous_logits + 0.1 * delta
#
#
# def build_refiner(feature_channels):
#     return Refiner(feature_channels=feature_channels)


"""Recurrent token refiner for EC523 Project A (road segmentation).

Uses our own modules (in project_a/cli/):
    tokenizer/Tx.py       Tx: features -> image tokens (+ 2D position)
    tokenizer/Ty.py       Ty: mask y_t -> answer tokens (patch conv + 2D position)
    refiner/memory.py     Initializer: nz learned memory tokens read image + y_0 -> z_0
    transformer/core.py   Core: shared transformer (model.core)
    refiner/head.py       Head: answer tokens -> full-resolution correction dy

This is refiner/refiner.py rewritten for the staff runner: predict.py owns the
loop and never passes K, so the setup becomes initialize() and the loop body
becomes step().

    state = model.initialize(features, y_0)                # z_0 = I(f, y_0)
    repeat K times:
        state, y = model.step(features, state, y)          # (z_t+1, y_t+1) = R(f, z_t, y_t)

One step (joint attention):
    [image tokens, answer tokens, memory z_t] --core (ONE call)--> [_, answer', z_t+1]
    y_t+1 = y_t + Head(answer')

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
    def __init__(self, c_in, d=256, patch=32, nz=32, nhead=8, num_layers=4, dropout=0.1) -> None:
        super().__init__()
        # patch = pixels per token side. Ty and Head both use it, and the image
        # tokens are pooled to the same grid, so all three always line up.
        # 1024x1024 tile, patch 32 -> 32x32 grid -> 1024 image + 1024 answer tokens.
        self.patch = patch
        self.tx = Tx(c_in, d)
        self.ty = Ty(d, patch)
        self.init = Initializer(d, nz, nhead)
        self.core = Core(d, nhead, num_layers, dropout=dropout)   # the ONE shared core
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


def build_refiner(feature_channels):
    """Entry point used by predict.py and train.py."""
    return Refiner(c_in=feature_channels)
