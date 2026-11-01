"""Temporary frozen ResNet18/U-Net baseline; not a staff-provided checkpoint."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


def pad_image(image):
    h, w = image.shape[-2:]
    return F.pad(image, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")


class TemporaryBaseline(nn.Module):
    feature_channels = 256

    def __init__(self):
        super().__init__()
        import segmentation_models_pytorch as smp
        # prepare() owns downloads; inference loads complete local weights.
        self.segmenter = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                                  in_channels=3, classes=1, activation=None)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        # requires_grad_(False) does not freeze BatchNorm running statistics.
        self.segmenter.encoder.eval()
        return self

    def forward(self, image):
        features = self.segmenter.encoder((image - self.mean) / self.std)
        decoded = self.segmenter.decoder(features)
        logits = self.segmenter.segmentation_head(decoded)
        return features[-2], logits

    def freeze_encoder(self):
        self.segmenter.encoder.requires_grad_(False)
        self.segmenter.encoder.eval()


def prepare_encoder(path):
    from torchvision.models import ResNet18_Weights, resnet18
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"Encoder weights already exist: {path}")
    encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    temporary = path.with_name(path.name + ".tmp")
    save_file({k: v.contiguous() for k, v in encoder.state_dict().items()}, str(temporary),
              metadata={"source": "torchvision ResNet18 IMAGENET1K_V1"})
    temporary.replace(path)


def initialize_encoder(model, path):
    weights = load_file(str(path), device="cpu")
    weights = {k: v for k, v in weights.items() if not k.startswith("fc.")}
    model.segmenter.encoder.load_state_dict(weights, strict=True)


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(temporary))
    temporary.replace(path)


def load_weights(model, path, device="cpu"):
    model.load_state_dict(load_file(str(path), device=str(device)), strict=True)
    return model
