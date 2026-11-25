"""The full-resolution image inside the loop (TA ruling 2026-11-18: any CNN inside step() may see it).

The staff interface only hands step() the features f, so the image travels inside f:
    pack:    f [B, C, h, w] + image [B, 3, H, W]  ->  [B, C + 3*s*s, h, w]    (pixel_unshuffle at stride s)
    unpack:  [B, C + 3*s*s, h, w]  ->  f [B, C, h, w], image [B, 3, h*s, w*s]   (pixel_shuffle)
Crops on the patch grid keep the packed image aligned with the features.
"""
import torch
from torch import nn
from torch.nn import functional as F

# ImageNet normalization, as the DINOv2 encoder uses.
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def packed_channels(cfg):
    """How many channels of f hold the packed image for this RefinerConfig (0 = no image)."""
    return 3 * cfg.image.stride ** 2 if cfg.image.use else 0


def pack_image(features, image, stride):
    """Append the normalized RGB image [B, 3, H, W] in [0, 1] to features [B, C, h, w] as 3*stride^2 channels."""
    h, w = features.shape[-2:]
    x = (image - MEAN.to(image)) / STD.to(image)
    if tuple(x.shape[-2:]) != (h * stride, w * stride):      # image not a multiple of the stride
        x = F.interpolate(x, size=(h * stride, w * stride), mode="bilinear", align_corners=False)
    return torch.cat([features, F.pixel_unshuffle(x, stride).to(features.dtype)], dim=1)


def unpack_image(features, stride):
    """Packed features -> (encoder features [B, C, h, w], normalized image [B, 3, h*stride, w*stride])."""
    n = 3 * stride ** 2
    return features[:, :-n], F.pixel_shuffle(features[:, -n:], stride)


class ImagePackedBase(nn.Module):
    """Wraps a frozen base so base(image) returns (features with the image packed in, y0)."""

    def __init__(self, base, stride) -> None:
        super().__init__()
        self.base = base
        self.stride = stride
        self.feature_channels = base.feature_channels + 3 * stride ** 2

    def forward(self, image, which=None):
        features, logits = self.base(image) if which is None else self.base(image, which)
        return pack_image(features, image, self.stride), logits


class ImageCNN(nn.Module):
    """Normalized image [B, 3, H, W] -> edge features E [B, hidden, H, W].

    A few 3x3 convs at full resolution and no downsampling, so E keeps pixel-exact road boundaries.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        layers, c = [], 3
        for _ in range(cfg.layers):
            layers += [nn.Conv2d(c, cfg.hidden, 3, padding=1), nn.GELU()]
            c = cfg.hidden
        self.net = nn.Sequential(*layers)

    def forward(self, image):
        return self.net(image)
