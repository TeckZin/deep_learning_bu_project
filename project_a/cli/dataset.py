"""LoveDA road dataset + the public stress recipe as a training augmentation."""
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from make_stress import corrupt   # the staff's public corruption recipe (same ranges as the stress set)

ROAD = 3
IGNORE = 0
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_image(path):
    # Same preprocessing as predict.py: RGB, float32 in [0, 1], shape (3, H, W).
    with Image.open(path) as im:
        array = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def load_mask(path):
    # Raw course labels: 0 = no-data (ignored), 3 = road, 1,2,4,5,6,7 = non-road.
    with Image.open(path) as im:
        raw = torch.from_numpy(np.asarray(im, dtype=np.int64).copy())
    target = (raw == ROAD).float()[None]
    valid = (raw != IGNORE)[None]
    return target, valid


class StressAugment:
    """With probability p, apply one shadow / occlusion / blur corruption to the image.

    Uses make_stress.corrupt, so the corruptions match the leaderboard's stress set
    (20% of the score). Labels are untouched, exactly like the real stress set.
    Training images only; never apply this to validation or evaluation data.
    """
    KINDS = ("shadow", "occlusion", "blur")

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, image, target, valid):
        if random.random() >= self.p:
            return image, target, valid
        rng = np.random.default_rng(random.getrandbits(63))
        kind = self.KINDS[int(rng.integers(len(self.KINDS)))]
        pil = Image.fromarray(np.rint(image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        out, _ = corrupt(pil, kind, rng)
        image = torch.from_numpy(np.asarray(out, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()
        return image, target, valid


class RoadDataset(Dataset):
    """Pairs images and masks by file name; returns (image, target, valid, id) as full tiles.

    Cropping happens in train.py AFTER the frozen baseline has run on the full tile,
    so the refiner trains on exactly the features it will see at inference.
    """

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
            image, target, valid = self.transform(image, target, valid)
        return image, target, valid, image_id
