"""UNTRAINED interface demonstration, not a competitive baseline or solution.

Replace this package with your refiner. Keep the three-method interface and
expose one nn.Module as `core`. The staff runner, not this class, owns K.
State can be a tensor or a nested structure. All trainable modules must be
registered on the refiner before checkpoint loading. No lazy K-dependent layers.

Handout notation: `state` is z_t, a [B,N,d] bank of latent vector tokens;
`previous_logits` is y_t, a [B,1,H,W] dense answer map (NOT ground truth).
This example embeds image features and downsampled sigmoid(y_t), then fuses
them into token context. The core consumes [B,N,d], not full-resolution pixels.
The readout decodes z_(t+1) to a logit correction and returns y_(t+1).
"""
import torch
from torch import nn
from torch.nn import functional as F


class Refiner(nn.Module):
    def __init__(self, feature_channels=5, width=32):
        super().__init__()
        self.context = nn.Conv2d(feature_channels + 1, width, 1)
        # ONE instance, reused at every refinement step.
        self.core = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.readout = nn.Linear(width, 1)
        self.width = width

    def initialize(self, features, initial_logits):
        b, _, h, w = features.shape
        return features.new_zeros(b, h * w, self.width)

    def step(self, features, state, previous_logits):
        h, w = features.shape[-2:]
        mask = F.interpolate(previous_logits.sigmoid(), size=(h, w),
                             mode="bilinear", align_corners=False)
        context = self.context(torch.cat([features, mask], dim=1))
        context = context.flatten(2).transpose(1, 2)
        next_state = self.core(state + context)
        delta = self.readout(next_state).transpose(1, 2).reshape(-1, 1, h, w)
        delta = F.interpolate(delta, size=previous_logits.shape[-2:],
                              mode="bilinear", align_corners=False)
        return next_state, previous_logits + 0.1 * delta


def build_refiner(feature_channels):
    return Refiner(feature_channels=feature_channels)
