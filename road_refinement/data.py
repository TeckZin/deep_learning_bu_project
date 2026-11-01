"""Course-layout discovery and aligned binary-road training crops."""
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Tile:
    id: str
    image: Path
    label: Path
    domain: str

    def record(self):
        return {"id": self.id, "image": str(self.image), "label": str(self.label), "domain": self.domain}


def discover(root, split, per_domain):
    if per_domain < 0:
        raise ValueError("per-domain limits must be nonnegative (0 means all)")
    tiles = []
    for domain in ("Urban", "Rural"):
        folder = Path(root).resolve() / split / domain
        images = sorted((folder / "images_png").glob("*.png"))
        if not images:
            raise ValueError(f"No PNG images in {folder / 'images_png'}")
        if per_domain:
            images = images[:per_domain]
        for image in images:
            label = folder / "masks_png" / image.name
            if not label.is_file():
                raise ValueError(f"Missing paired mask: {label}")
            tiles.append(Tile(f"{domain.lower()}/{image.stem}", image, label, domain.lower()))
    return tiles


def read_tile(tile):
    with Image.open(tile.image) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    with Image.open(tile.label) as mask:
        labels = np.asarray(mask).copy()
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected a raw integer mask, not a color mask: {tile.label}")
    if labels.shape != rgb.shape[:2] or (labels < 0).any() or (labels > 7).any():
        raise ValueError(f"Image/mask dimensions or label IDs invalid: {tile.id}")
    return rgb, labels


def image_tensor(rgb):
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)


class RoadCrops(Dataset):
    """Deterministic by (seed, sample index), including after resume.

    Consecutive groups of len(tiles) samples visit every selected tile. Small
    images are padded with ignored pixels; labels are never resized.
    """
    def __init__(self, tiles, crop_size, samples, seed):
        self.tiles = tiles
        self.crop_size = crop_size
        self.samples = samples
        self.seed = seed

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        rng = random.Random(self.seed + index)
        for attempt in range(20):
            tile = self.tiles[(index + attempt) % len(self.tiles)]
            rgb, labels = read_tile(tile)
            pad_h = max(0, self.crop_size - labels.shape[0])
            pad_w = max(0, self.crop_size - labels.shape[1])
            if pad_h or pad_w:
                rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                labels = np.pad(labels, ((0, pad_h), (0, pad_w)), constant_values=0)
            y = rng.randrange(labels.shape[0] - self.crop_size + 1)
            x = rng.randrange(labels.shape[1] - self.crop_size + 1)
            mask = labels[y:y+self.crop_size, x:x+self.crop_size]
            if (mask != 0).any():
                return (image_tensor(rgb[y:y+self.crop_size, x:x+self.crop_size]),
                        torch.from_numpy(mask.copy()).long().unsqueeze(0))
        raise ValueError("Could not sample a valid crop after 20 attempts; inspect masks or increase crop size")


def masked_bce(logits, labels):
    valid = labels != 0
    if not valid.any():
        raise ValueError("All-ignore batch has no training loss")
    target = (labels == 3).to(logits.dtype)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return loss[valid].mean()
