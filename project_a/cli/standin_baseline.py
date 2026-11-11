"""Stand-in frozen baseline for development: DINOv2 encoder (frozen) + our own H0 head.

NOT the staff baseline. Use it only while the official baseline is unreleased, so the
refiner trains on realistic features f and initial logits y0. Same interface as baseline.py:

    features, initial_logits = base(image)    # image: RGB float32 [B, 3, H, W] in [0, 1]
    base.feature_channels                      # 384 for ViT-S/14

    image -> DINOv2 (frozen) -> f [B, C, H/16, W/16] -> H0 (trained once, then frozen) -> y0 [B, 1, H, W]

DINOv2 uses 14-pixel patches, so the image is resized to a multiple of 14 (1024 -> 1022),
and its patch features are resampled to an (H/16, W/16) grid. The refiner and train.py
then see exactly the same shapes as with a stride-16 encoder.

DINOv2 weights come from torch.hub. Download them once on the login node (internet):
    python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)"
After that the cached copy is used, also on compute nodes without internet.
"""
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file

DINO_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}
PATCH = 14      # DINOv2 patch size
STRIDE = 16     # feature grid we hand to the refiner: H/16 x W/16


class DinoEncoder(nn.Module):
    """Frozen DINOv2. image [B,3,H,W] in [0,1] -> features [B, C, out_h, out_w]."""

    def __init__(self, name="dinov2_vits14") -> None:
        super().__init__()
        if name not in DINO_DIMS:
            raise ValueError(f"Unknown DINOv2 model {name!r}; choose from {sorted(DINO_DIMS)}")
        self.name = name
        self.channels = DINO_DIMS[name]
        self.vit = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
        self.vit.requires_grad_(False)
        # ImageNet normalization; not saved in any checkpoint.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)          # always frozen, whatever the caller asks

    @torch.no_grad()
    def forward(self, image, out_hw):
        B, _, H, W = image.shape
        hr, wr = max(1, round(H / PATCH)) * PATCH, max(1, round(W / PATCH)) * PATCH
        x = (image - self.mean) / self.std
        if (hr, wr) != (H, W):
            x = F.interpolate(x, size=(hr, wr), mode="bilinear", align_corners=False)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            tokens = self.vit.forward_features(x)["x_norm_patchtokens"]      # [B, (hr/14)(wr/14), C]
        f = tokens.float().transpose(1, 2).reshape(B, self.channels, hr // PATCH, wr // PATCH)
        return F.interpolate(f, size=out_hw, mode="bilinear", align_corners=False)


class H0Head(nn.Module):
    """Our initial-guess head: features [B, C, h, w] -> road logits [B, 1, H, W].

    A small conv net on the 1/16 grid, bilinearly upsampled. Deliberately simple, like a
    typical frozen baseline head: blurry at road edges, which leaves the refiner real work.
    """

    def __init__(self, c_in, hidden=128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1))

    def forward(self, f, out_hw):
        return F.interpolate(self.net(f), size=out_hw, mode="bilinear", align_corners=False)


class StandinBaseline(nn.Module):
    """E + H0 wrapped like the staff baseline: forward(image) -> (features, initial_logits)."""

    def __init__(self, encoder, head) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.feature_channels = encoder.channels

    def forward(self, image):
        H, W = image.shape[-2:]
        f = self.encoder(image, (max(1, H // STRIDE), max(1, W // STRIDE)))
        return f, self.head(f, (H, W))


def load_standin(checkpoint, device):
    """Frozen stand-in baseline from an H0 checkpoint written by train_h0.py."""
    with safe_open(str(checkpoint), framework="pt") as fh:
        meta = fh.metadata() or {}
    encoder = DinoEncoder(meta.get("dino", "dinov2_vits14"))
    head = H0Head(encoder.channels, int(meta.get("hidden", 128)))
    head.load_state_dict(load_file(str(checkpoint)), strict=True)
    base = StandinBaseline(encoder, head).to(device).eval()
    base.requires_grad_(False)
    return base
