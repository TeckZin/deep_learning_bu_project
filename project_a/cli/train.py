# import argparse
# from pathlib import Path
# import random
#
# import numpy as np
# import torch
# from torch import nn
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, TensorDataset
# from safetensors.torch import save_file
#
# from baseline import load_baseline
# from dataset import RoadDataset
# from submission.model import build_refiner
#
#
# def seg_loss(logits, target, valid):
#     # BCE + Dice over valid pixels only (LoveDA no-data pixels are excluded).
#     valid = valid.float()
#     bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
#     bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
#     p = logits.sigmoid() * valid
#     t = target * valid
#     dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
#     return bce + dice
#
#
# def unroll(model, features, initial_logits, steps):
#     # Same loop as predict.py: initialize once, then call step() K times.
#     state = model.initialize(features, initial_logits)
#     current = initial_logits
#     outputs = []
#     for _ in range(steps):
#         state, current = model.step(features, state, current)
#         outputs.append(current)
#     return outputs
#
#
# def random_flip(image, target, valid):
#     if random.random() < 0.5:
#         image, target, valid = image.flip(-1), target.flip(-1), valid.flip(-1)
#     if random.random() < 0.5:
#         image, target, valid = image.flip(-2), target.flip(-2), valid.flip(-2)
#     return image, target, valid
#
#
# def demo_dataset():
#     images = torch.rand(8, 3, 64, 80)
#     targets = (torch.rand(8, 1, 64, 80) > 0.8).float()
#     valid = torch.ones_like(targets, dtype=torch.bool)
#     return TensorDataset(images, targets, valid)
#
#
# def save_checkpoint(model, path):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, str(path))
#
#
# def main(argv=None):
#     p = argparse.ArgumentParser(description="Train the refiner in submission/model.py")
#     p.add_argument("--images", type=Path, help="Folder of training images")
#     p.add_argument("--masks", type=Path, help="Folder of LoveDA masks (3 = road, 0 = no-data)")
#     p.add_argument("--baseline", type=Path, help="Frozen baseline checkpoint")
#     p.add_argument("--output", type=Path, default=Path("refiner.safetensors"))
#     p.add_argument("--steps", type=int, default=4, choices=range(1, 9), metavar="K")
#     p.add_argument("--epochs", type=int, default=10)
#     p.add_argument("--lr", type=float, default=1e-4)
#     p.add_argument("--batch-size", type=int, default=4)
#     p.add_argument("--workers", type=int, default=4)
#     p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     p.add_argument("--seed", type=int, default=0)
#     p.add_argument("--demo", action="store_true", help="Random data + demo baseline, checks the code only")
#     args = p.parse_args(argv)
#
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     device = torch.device(args.device)
#
#     base = load_baseline(args.baseline, device, demo=args.demo)
#     if isinstance(base, nn.Module):
#         base.eval()
#         base.requires_grad_(False)
#
#     if args.demo:
#         dataset = demo_dataset()
#         workers = 0
#     else:
#         if args.images is None or args.masks is None or args.baseline is None:
#             p.error("--images, --masks and --baseline are required unless --demo")
#         dataset = RoadDataset(args.images, args.masks)
#         workers = args.workers
#     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
#                         num_workers=workers, pin_memory=device.type == "cuda", drop_last=False)
#
#     model = build_refiner(base.feature_channels).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
#
#     for epoch in range(args.epochs):
#         model.train()
#         total = 0.0
#         for batch in loader:
#             image, target, valid = (t.to(device, non_blocking=True) for t in batch[:3])
#             image, target, valid = random_flip(image, target, valid)
#             with torch.no_grad():
#                 features, initial_logits = base(image)
#             # Loss on every step so the mask is good at any K the runner uses.
#             outputs = unroll(model, features, initial_logits, args.steps)
#             loss = sum(seg_loss(y, target, valid) for y in outputs) / len(outputs)
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             opt.step()
#             total += loss.item()
#         print(f"epoch {epoch + 1}/{args.epochs}: loss {total / max(len(loader), 1):.4f}")
#         save_checkpoint(model, args.output)
#
#     print(f"Saved {args.output}" + (" (DEMO ONLY: random data)" if args.demo else ""))
#
#
# if __name__ == "__main__":
#     main()


import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
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


def loveda_dataset(root):
    # LoveDA layout: <root>/<Domain>/images_png and <root>/<Domain>/masks_png
    # (Domain = Rural, Urban). Combine every domain found into one dataset.
    parts = []
    for domain in sorted(d for d in Path(root).iterdir() if d.is_dir()):
        images, masks = domain / "images_png", domain / "masks_png"
        if images.is_dir() and masks.is_dir():
            parts.append(RoadDataset(images, masks))
            print(f"{domain.name}: {len(parts[-1])} images")
    if not parts:
        raise FileNotFoundError(f"No */images_png + */masks_png folders under {root}")
    return ConcatDataset(parts)


def take_subset(dataset, fraction=1.0, max_images=None, seed=0):
    # Random subset, fixed by the seed so the same images are picked every run.
    if not 0 < fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")
    n = max(1, round(len(dataset) * fraction))
    if max_images is not None:
        n = min(n, max_images)
    if n >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
    return Subset(dataset, indices)


def save_checkpoint(model, path):
    # Writes the same weights twice:
    #   <name>.safetensors  what predict.py loads (safe, no pickle)
    #   <name>.ckpt         torch.save state dict, load with torch.load(path, weights_only=True)
    path = path.with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(path))
    torch.save(state, str(path.with_suffix(".ckpt")))
    return path, path.with_suffix(".ckpt")


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the refiner in submission/model.py")
    p.add_argument("--data-root", type=Path,
                   help="LoveDA split folder, e.g. .../dataset/Train (uses every */images_png + */masks_png)")
    p.add_argument("--baseline", type=Path, help="Frozen baseline checkpoint")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Train on this random fraction of the images, e.g. 0.1 = 10%%")
    p.add_argument("--max-images", type=int,
                   help="Train on at most this many random images (applied after --fraction)")
    p.add_argument("--demo-baseline", action="store_true",
                   help="Real data + UNTRAINED demo baseline, for pipeline tests until staff release the baseline")
    p.add_argument("--output", type=Path, default=Path("weights/refiner.safetensors"),
                   help="Checkpoint path; a .ckpt copy is written next to it")
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

    use_demo_baseline = args.demo or args.demo_baseline
    if not use_demo_baseline and args.baseline is None:
        p.error("--baseline is required unless --demo or --demo-baseline")
    base = load_baseline(args.baseline, device, demo=use_demo_baseline)
    if isinstance(base, nn.Module):
        base.eval()
        base.requires_grad_(False)

    if args.demo:
        dataset = demo_dataset()
        workers = 0
    else:
        if args.data_root is None:
            p.error("--data-root is required unless --demo")
        dataset = loveda_dataset(args.data_root)
        workers = args.workers
    dataset = take_subset(dataset, args.fraction, args.max_images, args.seed)
    print(f"{len(dataset)} training images")
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
        st_path, ckpt_path = save_checkpoint(model, args.output)

    print(f"Saved {st_path} and {ckpt_path}" + (" (DEMO ONLY: random data)" if args.demo else ""))


if __name__ == "__main__":
    main()
