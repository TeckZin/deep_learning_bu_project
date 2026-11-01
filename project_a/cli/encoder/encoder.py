import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights


class Encoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        r = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.body(x)


if __name__ == "__main__":
    enc = Encoder()
    x = torch.randn(2, 3, 256, 320)
    f = enc(x)
    print("image:", tuple(x.shape))
    print("f    :", tuple(f.shape))
