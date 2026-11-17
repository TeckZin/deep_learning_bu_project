from torch import nn
from torch.nn import functional as F


class EncoderBaseline(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.feature_channels = encoder.feature_channels
        self.head = nn.Conv2d(self.feature_channels, 1, kernel_size=1)

    def initial_logits(self, features, image_size):
        coarse_logits = self.head(features)
        return F.interpolate(
            coarse_logits,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, images):
        features = self.encoder(images)
        logits = self.initial_logits(features, images.shape[-2:])
        return features, logits