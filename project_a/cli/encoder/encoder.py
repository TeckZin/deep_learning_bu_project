"""Encoder: frozen ImageNet-pretrained ResNet-34 feature extractor.

Maps an RGB image [B, 3, H, W] to a feature map [B, 256, H/16, W/16] using the
ResNet-34 stem plus layer1-layer3 (layer4 and the classifier are dropped).
All weights are frozen; it is never trained.
"""
import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights


class Encoder(nn.Module):
    """Image [B, 3, H, W] -> features [B, 256, H/16, W/16]."""

    def __init__(self) -> None:
        super().__init__()
        # Downloads ImageNet weights on first use (needs internet).
        r = resnet34(weights=ResNet34_Weights.DEFAULT)
        # Downsampling by stage: conv1 /2, maxpool /4, layer1 /4, layer2 /8, layer3 /16.
        # layer3 outputs 256 channels.
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        # Freeze: no gradients for any encoder weight.
        for p in self.parameters():
            p.requires_grad = False
        # Put BatchNorm in inference mode (use stored ImageNet statistics).
        self.eval()

    @torch.no_grad()  # never build a graph through the frozen encoder
    def forward(self, x):
        return self.body(x)


if __name__ == "__main__":
    # Shape check: 256x320 image -> 16x20 feature grid with 256 channels.
    enc = Encoder()
    x = torch.randn(2, 3, 256, 320)
    f = enc(x)
    print("image:", tuple(x.shape))
    print("f    :", tuple(f.shape))
