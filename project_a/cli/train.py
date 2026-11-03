import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import save_file

from baseline import load_baseline
from dataset import RoadDataset
from submission.model import build_refiner


def seg_loss(logits, target, valid):
    # BCE + Dice over valid pixels only (LoveDA no-data pixels are excluded).
    valid = valid.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    p = logits.sigmoid() * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


def unroll(model, features, initial_logits, steps):
    # Same loop as predict.py: initialize once, then call step() K times.
    state = model.initialize(features, initial_logits)
    current = initial_logits
    outputs = []
    for _ in range(steps):
        state, current = model.step(features, state, current)
        outputs.append(current)
    return outputs


def random_flip(image, target, valid):
    if random.random() < 0.5:
        image, target, valid = image.flip(-1), target.flip(-1), valid.flip(-1)
    if random.random() < 0.5:
        image, target, valid = image.flip(-2), target.flip(-2), valid.flip(-2)
    return image, target, valid


def demo_dataset():
    images = torch.rand(8, 3, 64, 80)
    targets = (torch.rand(8, 1, 64, 80) > 0.8).float()
    valid = torch.ones_like(targets, dtype=torch.bool)
    return TensorDataset(images, targets, valid)


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, str(path))


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the refiner in submission/model.py")
    p.add_argument("--images", type=Path, help="Folder of training images")
    p.add_argument("--masks", type=Path, help="Folder of LoveDA masks (3 = road, 0 = no-data)")
    p.add_argument("--baseline", type=Path, help="Frozen baseline checkpoint")
    p.add_argument("--output", type=Path, default=Path("refiner.safetensors"))
    p.add_argument("--steps", type=int, default=4, choices=range(1, 9), metavar="K")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo", action="store_true", help="Random data + demo baseline, checks the code only")
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    base = load_baseline(args.baseline, device, demo=args.demo)
    if isinstance(base, nn.Module):
        base.eval()
        base.requires_grad_(False)

    if args.demo:
        dataset = demo_dataset()
        workers = 0
    else:
        if args.images is None or args.masks is None or args.baseline is None:
            p.error("--images, --masks and --baseline are required unless --demo")
        dataset = RoadDataset(args.images, args.masks)
        workers = args.workers
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=workers, pin_memory=device.type == "cuda", drop_last=False)

    model = build_refiner(base.feature_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            image, target, valid = (t.to(device, non_blocking=True) for t in batch[:3])
            image, target, valid = random_flip(image, target, valid)
            with torch.no_grad():
                features, initial_logits = base(image)
            # Loss on every step so the mask is good at any K the runner uses.
            outputs = unroll(model, features, initial_logits, args.steps)
            loss = sum(seg_loss(y, target, valid) for y in outputs) / len(outputs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / max(len(loader), 1):.4f}")
        save_checkpoint(model, args.output)

    print(f"Saved {args.output}" + (" (DEMO ONLY: random data)" if args.demo else ""))


if __name__ == "__main__":
    main()
