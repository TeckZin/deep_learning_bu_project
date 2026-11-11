from pathlib import Path

import torch

from dataset import RoadDataset
from encoder import build_encoder
from encoder_baseline import EncoderBaseline
from train import seg_loss

torch.set_num_threads(1)
torch.manual_seed(0)

root = Path("/projectnb/ec523kb/projects/RoadSeg/dataset/Train/Urban")
tiles = RoadDataset(root / "images_png", root / "masks_png")

for image, target, valid, tile_id in tiles:
    road_pixels = target[0].nonzero()
    if road_pixels.numel() > 0:
        break
else:
    raise RuntimeError("Found no road pixels in the Urban training tiles")

size = 256
row, col = road_pixels[len(road_pixels) // 2].tolist()
top = (row // size) * size
left = (col // size) * size
crop = (..., slice(top, top + size), slice(left, left + size))

images = image[crop].unsqueeze(0)
targets = target[crop].unsqueeze(0)
valid_pixels = valid[crop].unsqueeze(0)

baseline = EncoderBaseline(build_encoder("resnet34"))
baseline.train()
optimizer = torch.optim.AdamW(baseline.head.parameters(), lr=1e-3)

with torch.no_grad():
    features, logits = baseline(images)
    before = seg_loss(logits, targets, valid_pixels).item()

assert logits.shape == targets.shape

for _ in range(25):
    optimizer.zero_grad(set_to_none=True)
    logits = baseline.initial_logits(features, images.shape[-2:])
    loss = seg_loss(logits, targets, valid_pixels)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    logits = baseline.initial_logits(features, images.shape[-2:])
    after = seg_loss(logits, targets, valid_pixels).item()

print(f"Tile: {tile_id}")
print(f"Features: {tuple(features.shape)}")
print(f"Logits: {tuple(logits.shape)}")
print(f"Loss: {before:.4f} -> {after:.4f}")