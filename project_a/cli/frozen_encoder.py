"""Frozen encoder + H0 head (doc stages 2 encoder + 4 prediction head): image -> (features f, logits y0).

The handout calls this stage the frozen (initial) segmenter E + H0.

    image -> Encoder (frozen) -> f [B, C, H/16, W/16] -> H0 head (trained once, then frozen) -> y0 [B, 1, H, W]

Same interface as the staff baseline.py, so the refiner, train.py and infer.py do not care
which one they get:
    features, initial_logits = base(image)    # image: RGB float32 [B, 3, H, W] in [0, 1]
    base.feature_channels

FrozenEncoder can hold several H0 heads that share one encoder. Out-of-fold training uses this:
base(image, which=[...]) gives each image y0 from the head that never trained on it.
With which=None the first head is used (validation / inference).

Encoders (ENCODERS):
    dinov2_vits14 / vitb14 / vitl14   frozen DINOv2 from torch.hub (facebookresearch/dinov2).
                                      Download once with internet:
        python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)"
    pool                              5-channel pooled RGB + coordinates; no download.
                                      For tests and demos only.
"""
import os

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

STRIDE = 16     # feature grid handed to the refiner: H/16 x W/16
DINO_DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024}
DINO_PATCH = 14


class DinoEncoder(nn.Module):
    """Frozen DINOv2. image [B,3,H,W] in [0,1] -> features [B, C, out_h, out_w].

    DINOv2 uses 14-pixel patches, so the image is resized to a multiple of 14 (1024 -> 1022)
    and its patch features are resampled to the (H/16, W/16) grid.
    """

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
        hr, wr = max(1, round(H / DINO_PATCH)) * DINO_PATCH, max(1, round(W / DINO_PATCH)) * DINO_PATCH
        x = (image - self.mean) / self.std
        if (hr, wr) != (H, W):
            x = F.interpolate(x, size=(hr, wr), mode="bilinear", align_corners=False)
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"):
            tokens = self.vit.forward_features(x)["x_norm_patchtokens"]      # [B, (hr/14)(wr/14), C]
        f = tokens.float().transpose(1, 2).reshape(B, self.channels, hr // DINO_PATCH, wr // DINO_PATCH)
        return F.interpolate(f, size=out_hw, mode="bilinear", align_corners=False)


class PoolEncoder(nn.Module):
    """Tests / demos only: average-pooled RGB + 2 coordinate channels (like the staff DemoBaseline)."""
    name = "pool"
    channels = 5

    @torch.no_grad()
    def forward(self, image, out_hw):
        small = F.adaptive_avg_pool2d(image, out_hw)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, out_hw[0], device=image.device),
                                torch.linspace(-1, 1, out_hw[1], device=image.device), indexing="ij")
        pos = torch.stack([yy, xx])[None].expand(image.shape[0], -1, -1, -1)
        return torch.cat([small, pos], 1)


ENCODERS = {**{name: (lambda n=name: DinoEncoder(n)) for name in DINO_DIMS}, "pool": PoolEncoder}


def build_encoder(name):
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    return ENCODERS[name]()


class H0Head(nn.Module):
    """Initial-guess head: features [B, C, h, w] -> road logits [B, 1, H, W].

    A small conv net on the 1/16 grid, bilinearly upsampled. Deliberately simple:
    blurry at road edges, which leaves the refiner real work.
    """

    def __init__(self, c_in, hidden=128) -> None:
        super().__init__()
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 1, 1))

    def forward(self, f, out_hw):
        return F.interpolate(self.net(f), size=out_hw, mode="bilinear", align_corners=False)


class FrozenEncoder(nn.Module):
    """Encoder + one or more H0 heads: forward(image, which=None) -> (features, initial_logits)."""

    def __init__(self, encoder, heads) -> None:
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList(heads)
        self.feature_channels = encoder.channels

    def features(self, image):
        H, W = image.shape[-2:]
        return self.encoder(image, (max(1, H // STRIDE), max(1, W // STRIDE)))

    def forward(self, image, which=None):
        """which: None (head 0 for all) or a list / LongTensor [B] of head indices per image."""
        f = self.features(image)
        out_hw = image.shape[-2:]
        if which is None or len(self.heads) == 1:
            return f, self.heads[0](f, out_hw)
        which = [int(i) for i in which]
        y = torch.empty(f.shape[0], 1, *out_hw, device=f.device, dtype=f.dtype)
        for k in set(which):
            idx = [i for i, w in enumerate(which) if w == k]
            y[idx] = self.heads[k](f[idx], out_hw)
        return f, y


def save_head(head, path, encoder_name, metadata=None):
    """H0 weights + the metadata needed to rebuild it (encoder name, hidden size)."""
    meta = {"kind": "h0", "encoder": encoder_name, "hidden": str(head.hidden), **(metadata or {})}
    state = {k: v.detach().contiguous().cpu() for k, v in head.state_dict().items()}
    save_file(state, str(path), metadata=meta)
    return path


def head_metadata(path):
    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata() or {}
    # Older (attempt6) H0 files stored the encoder under "dino".
    meta.setdefault("encoder", meta.get("dino", "dinov2_vits14"))
    return meta


def load_frozen_encoder(head_paths, device, encoder=None):
    """Frozen FrozenEncoder from one or more H0 files (all must use the same encoder)."""
    head_paths = [head_paths] if isinstance(head_paths, (str, os.PathLike)) else list(head_paths)
    metas = [head_metadata(p) for p in head_paths]
    names = {m["encoder"] for m in metas}
    if len(names) != 1:
        raise ValueError(f"H0 heads use different encoders: {sorted(names)}")
    encoder = encoder or build_encoder(names.pop())
    heads = []
    for path, meta in zip(head_paths, metas):
        head = H0Head(encoder.channels, int(meta.get("hidden", 128)))
        head.load_state_dict(load_file(str(path)), strict=True)
        heads.append(head)
    base = FrozenEncoder(encoder, heads).to(device).eval()
    base.requires_grad_(False)
    return base
