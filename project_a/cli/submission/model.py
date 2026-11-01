"""Shared recurrent refiner using Teck's image tokenizer.

The baseline is frozen and external to this module. Image tokens are computed
once in initialize(); only memory and the dense answer change across steps.
The runner owns K. All modules are registered before loading weights.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.tokenizer.input.Tx import Tx


class Refiner(nn.Module):
    def __init__(self, feature_channels=256, width=256):
        super().__init__()
        self.image_tokenizer = Tx(feature_channels, d=width)
        self.mask_projection = nn.Conv2d(1, width, 1)
        self.mask_norm = nn.LayerNorm(width)
        self.core = nn.TransformerEncoderLayer(
            d_model=width, nhead=8, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.readout = nn.Linear(width, 1)
        self.width = width

    def initialize(self, features, initial_logits):
        image_tokens = self.image_tokenizer(features)
        return {"memory": torch.zeros_like(image_tokens), "image_tokens": image_tokens}

    def step(self, features, state, previous_logits):
        h, w = features.shape[-2:]
        probability = F.interpolate(previous_logits.sigmoid(), size=(h, w),
                                    mode="bilinear", align_corners=False)
        answer = self.mask_projection(probability).flatten(2).transpose(1, 2)
        answer = self.mask_norm(answer)
        memory = self.core(state["memory"] + state["image_tokens"] + answer)
        delta = self.readout(memory).transpose(1, 2).reshape(-1, 1, h, w)
        delta = F.interpolate(delta, size=previous_logits.shape[-2:],
                              mode="bilinear", align_corners=False)
        return {"memory": memory, "image_tokens": state["image_tokens"]}, previous_logits + 0.1 * delta


def build_refiner(feature_channels):
    return Refiner(feature_channels=feature_channels)
