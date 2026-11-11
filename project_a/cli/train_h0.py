#!/usr/bin/env python3
"""Train our own H0 head on a frozen DINOv2 encoder -> stand-in baseline for development.

Only the small H0 head is trained (DINOv2 stays frozen). Training split only; the
validation manifest is used to pick the best epoch. The result is then FROZEN and used by
    python train.py --standin-baseline weights/h0_dino_best.safetensors ...

Writes next to --output:
    <name>.safetensors        latest epoch
    <name>_best.safetensors   best validation quality (needs --val-manifest)

Not the staff baseline: for refiner development only. Check with course staff before
reporting results that depend on it.
"""
import argparse
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from common import load_manifest
from metrics import Counts, image_counts, summarize
from standin_baseline import DinoEncoder, H0Head, StandinBaseline
from train import augment_tiles, load_rgb, loveda_dataset, seg_loss, take_subset


def save_head(head, path, dino, hidden):
    path = Path(path).with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in head.state_dict().items()}
    save_file(state, str(path), metadata={"kind": "standin-h0", "dino": dino, "hidden": str(hidden)})
    return path


@torch.no_grad()
def validate(encoder, head, rows, device):
    """Staff metrics for the stand-in baseline alone (this is the refiner's K=0)."""
    head.eval()
    base = StandinBaseline(encoder, head)
    total = Counts()
    for row in rows:
        image = load_rgb(row["image"], device)
        with Image.open(row["label"]) as im:
            labels = np.asarray(im)
        _, y = base(image)
        total = total + image_counts(y.float().sigmoid()[0, 0].cpu().numpy(), labels)
    head.train()
    return summarize(total)


def main(argv=None):
    p = argparse.ArgumentParser(description="Train H0 on frozen DINOv2 (stand-in baseline)")
    p.add_argument("--data-root", type=Path, required=True, help="Training split, e.g. .../dataset/Train")
    p.add_argument("--val-manifest", type=Path, help="Validation manifest; picks the best epoch")
    p.add_argument("--dino", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--max-images", type=int)
    p.add_argument("--part", choices=["all", "a", "b"], default="all",
                   help="Fixed 50/50 split of Train: use 'a' here and 'b' for train.py")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--output", type=Path, default=Path("weights/h0_dino.safetensors"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    if "val" in {part.lower() for part in args.data_root.resolve().parts}:
        p.error(f"Refusing to train on a validation folder: {args.data_root}")

    encoder = DinoEncoder(args.dino).to(device)
    head = H0Head(encoder.channels, args.hidden).to(device)
    print(f"DINO {args.dino} (frozen, {encoder.channels} channels); "
          f"H0 params {sum(q.numel() for q in head.parameters()):,}")

    dataset = take_subset(loveda_dataset(args.data_root), args.fraction, args.max_images, args.seed,
                          args.part)
    print(f"{len(dataset)} training images")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", drop_last=len(dataset) > args.batch_size,
                        persistent_workers=args.workers > 0)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_iters = max(1, args.epochs * len(loader))
    warmup = min(args.warmup, max(1, total_iters // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (it + 1) / warmup if it < warmup else
                                              0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (it - warmup) / max(1, total_iters - warmup)))))

    val_rows = load_manifest(args.val_manifest) if args.val_manifest else None
    stem = args.output.with_suffix("")
    best_path = stem.with_name(stem.name + "_best.safetensors")
    best = -1.0

    for epoch in range(args.epochs):
        head.train()
        total = 0.0
        for batch in loader:
            image, target, valid = (t.to(device, non_blocking=True) for t in batch[:3])
            image, target, valid = augment_tiles(image, target, valid)
            H, W = image.shape[-2:]
            f = encoder(image, (H // 16, W // 16))                     # frozen, no gradients
            logits = head(f, (H, W))
            loss = seg_loss(logits, target, valid, args.pos_weight, args.dice_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / max(len(loader), 1):.4f}, "
              f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        latest = save_head(head, args.output, args.dino, args.hidden)
        if val_rows:
            s = validate(encoder, head, val_rows, device)
            print(f"  val H0 alone: quality {100 * s['quality']:.2f}  IoU {100 * s['road_iou']:.2f}  "
                  f"clDice {100 * s['cldice']:.2f}  BIoU {100 * s['boundary_iou']:.2f}", flush=True)
            if s["quality"] > best:
                best = s["quality"]
                save_head(head, best_path, args.dino, args.hidden)
                print(f"  new best -> {best_path}", flush=True)

    print(f"Saved {latest}" + (f"; best quality {100 * best:.2f} -> {best_path}" if val_rows else ""))


if __name__ == "__main__":
    main()
