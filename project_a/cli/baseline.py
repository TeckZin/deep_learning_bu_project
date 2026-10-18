"""STAFF-OWNED adapter. No trained course-data baseline is included in this kit.

Staff must implement load_official_baseline and publish its checkpoint hash.
The model must return (features, initial_logits) for RGB float32 [B,3,H,W]
in [0,1], and expose feature_channels. Logits must be [B,1,H,W].
All baseline parameters AND normalization statistics remain frozen.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DemoBaseline(nn.Module):
    """Deterministic brightness heuristic for testing only; not learned."""
    feature_channels = 5

    def forward(self, image):
        h, w = image.shape[-2:]
        small = F.adaptive_avg_pool2d(image, (max(1, h // 16), max(1, w // 16)))
        sh, sw = small.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, sh, device=image.device),
            torch.linspace(-1, 1, sw, device=image.device), indexing="ij")
        pos = torch.stack([yy, xx])[None].expand(image.shape[0], -1, -1, -1)
        return torch.cat([small, pos], 1), 8.0 * (image.mean(1, keepdim=True) - 0.5)


def load_official_baseline(checkpoint, device):
    raise RuntimeError(
        "Official frozen course baseline is not bundled. Staff must release "
        "the adapter/checkpoint. Use --demo ONLY for interface smoke tests.")


def load_baseline(checkpoint, device, demo=False):
    if demo:
        if checkpoint is not None:
            raise ValueError("--demo cannot be combined with --baseline")
        model = DemoBaseline()
    else:
        if checkpoint is None:
            raise ValueError("Official mode requires --baseline CHECKPOINT")
        model = load_official_baseline(checkpoint, device)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model
