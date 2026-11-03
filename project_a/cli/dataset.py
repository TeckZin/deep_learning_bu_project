from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

ROAD = 3
IGNORE = 0
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_image(path):
    # Same preprocessing as predict.py: RGB, float32 in [0, 1], shape (3, H, W).
    with Image.open(path) as im:
        array = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def load_mask(path):
    # LoveDA labels: 0 = no-data, 1..7 = classes, 3 = road. No clipping.
    with Image.open(path) as im:
        raw = torch.from_numpy(np.asarray(im, dtype=np.int64).copy())
    target = (raw == ROAD).float()[None]
    valid = (raw != IGNORE)[None]
    return target, valid


class RoadDataset(Dataset):
    """Pairs images and masks by file name; returns (image, target, valid, id)."""

    def __init__(self, image_dir, mask_dir, ids=None, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        images = {p.stem: p for p in sorted(self.image_dir.iterdir())
                  if p.suffix.lower() in IMAGE_SUFFIXES}
        masks = {p.stem: p for p in sorted(self.mask_dir.iterdir())
                 if p.suffix.lower() in IMAGE_SUFFIXES}
        self.ids = sorted(ids) if ids is not None else sorted(images.keys() & masks.keys())
        missing = [i for i in self.ids if i not in images or i not in masks]
        if missing:
            raise FileNotFoundError(f"{len(missing)} ids lack an image or mask, e.g. {missing[:3]}")
        if not self.ids:
            raise ValueError(f"No matching image/mask pairs in {image_dir} and {mask_dir}")
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = load_image(self.images[image_id])
        target, valid = load_mask(self.masks[image_id])
        if image.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"{image_id}: image {tuple(image.shape)} vs mask {tuple(target.shape)}")
        if self.transform is not None:
            # A transform must apply the same geometric change to all three tensors.
            image, target, valid = self.transform(image, target, valid)
        return image, target, valid, image_id


def masked_bce(logits, target, valid):
    # Binary cross-entropy over valid pixels only (no-data pixels are excluded).
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid = valid.float()
    return (loss * valid).sum() / valid.sum().clamp_min(1.0)
