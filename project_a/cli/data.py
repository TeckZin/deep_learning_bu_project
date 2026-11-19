"""Data (doc stage 1): tile loading, training augmentation, crops, y0 degradation.

Rows come from splits.py: dicts with "id", "domain", "image" and "label" (absolute paths).
Full tiles are loaded; cropping happens in train.py AFTER the frozen encoder + H0 have run
on the full tile, so the refiner trains on exactly the features it sees at inference.
"""
import math
import random

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from make_stress import corrupt   # the staff's public corruption recipe (same ranges as the stress set)

ROAD = 3
IGNORE = 0


def load_image(path):
    """RGB float32 in [0, 1], shape (3, H, W). Same preprocessing as predict.py / infer.py."""
    with Image.open(path) as im:
        array = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def load_mask(path):
    """Raw course labels: 0 = no-data (ignored), 3 = road, 1,2,4,5,6,7 = non-road."""
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


class TileDataset(Dataset):
    """Full tiles from split rows -> (image [3,H,W], target [1,H,W], valid [1,H,W], id)."""

    def __init__(self, rows, transform=None):
        if not rows:
            raise ValueError("TileDataset needs at least one row")
        self.rows = list(rows)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_image(row["image"])
        target, valid = load_mask(row["label"])
        if image.shape[-2:] != target.shape[-2:]:
            raise ValueError(f"{row['id']}: image {tuple(image.shape)} vs mask {tuple(target.shape)}")
        if self.transform is not None:
            image, target, valid = self.transform(image, target, valid)
        return image, target, valid, row["id"]


class DemoDataset(Dataset):
    """Random tiles for --demo: checks the code path only."""

    def __init__(self, n=8, size=(64, 80)):
        g = torch.Generator().manual_seed(0)
        self.images = torch.rand(n, 3, *size, generator=g)
        self.targets = (torch.rand(n, 1, *size, generator=g) > 0.8).float()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], self.targets[i], torch.ones_like(self.targets[i], dtype=torch.bool), f"demo/{i}"


def subset_rows(rows, fraction=1.0, max_images=None, seed=0):
    """Random subset for quick runs, fixed by the seed so the same rows are picked every run."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = max(1, round(len(rows) * fraction))
    if max_images is not None:
        n = min(n, max_images)
    if n >= len(rows):
        return list(rows)
    return random.Random(seed).sample(list(rows), n)


def augment_tiles(image, target, valid):
    """Random horizontal flip + 90-degree rotation, independently per tile, before the baseline."""
    square = image.shape[-2] == image.shape[-1]
    out = ([], [], [])
    for i in range(image.shape[0]):
        parts = [image[i], target[i], valid[i]]
        if random.random() < 0.5:
            parts = [x.flip(-1) for x in parts]
        k = random.randrange(4) if square else 2 * random.randrange(2)
        if k:
            parts = [torch.rot90(x, k, dims=(-2, -1)) for x in parts]
        for collected, x in zip(out, parts):
            collected.append(x)
    return tuple(torch.stack(c) for c in out)


def random_crops(features, logits, target, valid, crop, patch):
    """One random crop per tile, aligned so it maps exactly onto the feature grid and patch grid.

    Returns the cropped tensors and offsets [B, 2] = crop's top-left (row, col) in patch cells.
    """
    B, _, H, W = logits.shape
    fh, fw = features.shape[-2:]
    if H % fh or W % fw:
        raise ValueError(f"Feature grid {fh}x{fw} does not evenly divide the image {H}x{W}")
    sy, sx = H // fh, W // fw                             # baseline feature stride
    align = math.lcm(patch, sy, sx)
    ch, cw = min(crop, H) // align * align, min(crop, W) // align * align
    if ch == 0 or cw == 0:
        raise ValueError(f"--crop {crop} is smaller than the required alignment {align}")
    fs, ls, ts, vs, offsets = [], [], [], [], []
    for i in range(B):
        y = random.randrange((H - ch) // align + 1) * align
        x = random.randrange((W - cw) // align + 1) * align
        fs.append(features[i, :, y // sy:(y + ch) // sy, x // sx:(x + cw) // sx])
        ls.append(logits[i, :, y:y + ch, x:x + cw])
        ts.append(target[i, :, y:y + ch, x:x + cw])
        vs.append(valid[i, :, y:y + ch, x:x + cw])
        offsets.append((y // patch, x // patch))
    return (torch.stack(fs), torch.stack(ls), torch.stack(ts), torch.stack(vs),
            torch.tensor(offsets, device=logits.device))


def degrade_logits(logits, prob, patch=16):
    """Randomly make y0 worse for the refiner's INPUT only (labels unchanged).

    Per tile, with probability `prob`, one of:
      coarse  - average-pool to 2x or 4x the patch size, upsample back (blurrier edges)
      holes   - 10-25% of patch cells pushed to 'unsure' (logit * 0.1), so road
                pieces go missing and the refiner must reconnect them
      flips   - 3-10% of patch cells get their logit sign flipped (false road / missed road)
      temp    - logit scale times U(0.3, 1.5) (over/under-confident baseline)
    The refiner is then trained to correct errors larger than the ones a training-set
    y0 makes. Features are left untouched, so the image evidence is still there.
    """
    if prob <= 0:
        return logits
    B, _, H, W = logits.shape
    out = logits.clone()
    h, w = max(1, H // patch), max(1, W // patch)
    for i in range(B):
        if random.random() >= prob:
            continue
        y = out[i:i + 1]
        kind = random.randrange(4)
        if kind == 0:
            k = patch * random.choice((2, 4))
            if H >= k and W >= k:
                y = F.interpolate(F.avg_pool2d(y, k, stride=k, ceil_mode=True), size=(H, W),
                                  mode="bilinear", align_corners=False)
        elif kind in (1, 2):
            frac = random.uniform(0.10, 0.25) if kind == 1 else random.uniform(0.03, 0.10)
            cells = (torch.rand(1, 1, h, w, device=y.device) < frac).float()
            cells = F.interpolate(cells, size=(H, W), mode="nearest")
            factor = 0.1 if kind == 1 else -1.0
            y = y * (1 - cells) + factor * y * cells
        else:
            y = y * random.uniform(0.3, 1.5)
        out[i:i + 1] = y
    return out
