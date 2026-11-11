import torch
from torch import nn
from torchvision import models


class FrozenEncoder(nn.Module):
    def __init__(self, body, feature_channels, output_stride, mean, std):
        super().__init__()
        self.body = body
        self.feature_channels = feature_channels
        self.output_stride = output_stride

        self.register_buffer(
            "mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Keep the encoder in evaluation mode while a parent model trains.
        return super().train(False)

    @torch.no_grad()
    def forward(self, images):
        """Map RGB floats in [0, 1] from [B, 3, H, W] to [B, C, h, w]."""
        normalized_images = (images - self.mean) / self.std
        return self.body(normalized_images)

class DinoPatchGrid(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, images):
        batch_size, _, height, width = images.shape
        if height % 16 or width % 16:
            raise ValueError("DINO ViT-S/16 needs height and width divisible by 16")

        tokens = self.backbone.get_intermediate_layers(images, n=1)[0]
        patch_tokens = tokens[:, 1:]  # Keep the spatial tokens.
        grid = patch_tokens.reshape(batch_size, height // 16, width // 16, -1)
        return grid.permute(0, 3, 1, 2).contiguous()

def build_encoder(name):
    if name == "dino_vits16":
        backbone = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vits16",
            trust_repo=True,
        )
        return FrozenEncoder(
            DinoPatchGrid(backbone),
            feature_channels=backbone.embed_dim,
            output_stride=16,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    options = {
        "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 256),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 1024),
    }
    make_backbone, weights, channels = options[name]
    backbone = make_backbone(weights=weights)

    body = nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
    )

    preprocessing = weights.transforms()
    return FrozenEncoder(
        body,
        feature_channels=channels,
        output_stride=16,
        mean=preprocessing.mean,
        std=preprocessing.std,
    )