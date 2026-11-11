#!/usr/bin/env python3
"""Train the refiner in submission/model.py for the leaderboard (K = 1, 2, 4 + stress K = 4).

What one training step does, and why:
  1. Load full tiles; random flip / 90-degree rotation per tile; optional stress
     augmentation (public make_stress.py recipe) in the data loader.
  2. Run the frozen baseline on the FULL tile, exactly like predict.py.
  3. Take a random crop of features, initial logits and labels, aligned to the
     patch grid, and tell the model where the crop sits in the tile
     (model.pos_offset). The refiner trains on crops but sees the same inputs
     and position codes it gets at full-tile inference.
  4. Same loop as predict.py: initialize() once, step() K times. The loss is a
     weighted sum over every step (default favours step 4, like the leaderboard).
  5. AdamW + linear warmup + cosine decay, bf16 autocast, gradient clipping.

Each epoch: validates with the staff metrics at K = 0, 1, 2, 4 (and stress K = 4 if
--val-stress-manifest is given), keeps the checkpoint with the best
leaderboard-formula score, and writes <name>.resume.pt. Re-run with
--resume <name>.resume.pt to continue a job that was killed.

Checkpoints written next to --output:
    <name>.safetensors        latest epoch (what predict.py loads)
    <name>_best.safetensors   best validation score (needs --val-manifest)
    <name>.resume.pt          model + optimizer + scheduler + epoch, for --resume
"""
import argparse
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from safetensors.torch import save_file

from baseline import load_baseline
from common import load_manifest
from dataset import RoadDataset, StressAugment
from metrics import Counts, image_counts, leaderboard_score, summarize
from submission.model import build_refiner

VAL_KS = (0, 1, 2, 4)
CORE_LIMIT, TOTAL_LIMIT = 5_000_000, 10_000_000      # proposed caps from the handout


# ---------------------------------------------------------------- loss / loop

def seg_loss(logits, target, valid, pos_weight=1.0, dice_weight=1.0):
    """BCE + soft Dice over valid pixels only (no-data pixels excluded). Computed in fp32."""
    logits = logits.float()
    valid = valid.float()
    pw = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    p = logits.sigmoid() * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice_weight * dice


def unroll(model, features, initial_logits, steps):
    """Same loop as predict.py: initialize once, then call step() K times."""
    state = model.initialize(features, initial_logits)
    current = initial_logits
    outputs = []
    for _ in range(steps):
        state, current = model.step(features, state, current)
        outputs.append(current)
    return outputs


def parse_step_weights(text, steps):
    """Per-step loss weights, normalized to sum to 1. Default: 1 per step, 2 on step 4."""
    if text is None:
        weights = [1.0] * steps
        if steps >= 4:
            weights[3] = 2.0
    else:
        weights = [float(x) for x in text.split(",")]
    if len(weights) != steps or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError(f"--step-weights needs {steps} non-negative numbers, got {text!r}")
    total = sum(weights)
    return [w / total for w in weights]


# ---------------------------------------------------------------- augmentation / crops

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


# ---------------------------------------------------------------- data

def demo_dataset():
    images = torch.rand(8, 3, 64, 80)
    targets = (torch.rand(8, 1, 64, 80) > 0.8).float()
    valid = torch.ones_like(targets, dtype=torch.bool)
    return TensorDataset(images, targets, valid)


def loveda_dataset(root, transform=None):
    """LoveDA layout: <root>/<Domain>/images_png and <root>/<Domain>/masks_png."""
    parts = []
    for domain in sorted(d for d in Path(root).iterdir() if d.is_dir()):
        images, masks = domain / "images_png", domain / "masks_png"
        if images.is_dir() and masks.is_dir():
            parts.append(RoadDataset(images, masks, transform=transform))
            print(f"{domain.name}: {len(parts[-1])} images")
    if not parts:
        raise FileNotFoundError(f"No */images_png + */masks_png folders under {root}")
    return ConcatDataset(parts)


def take_subset(dataset, fraction=1.0, max_images=None, seed=0):
    """Random subset, fixed by the seed so the same images are picked every run."""
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


def load_rgb(path, device):
    """Same preprocessing as predict.py: RGB / 255, float32 [1, 3, H, W]."""
    with Image.open(path) as im:
        array = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)[None].to(device)


# ---------------------------------------------------------------- checkpoints

def save_weights(model, path):
    """Safetensors weights, the format predict.py loads (no pickle)."""
    path = Path(path).with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(path))
    return path


def save_resume(path, model, opt, sched, epoch, best_score, best_epoch):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch}, str(path))


# ---------------------------------------------------------------- validation

@torch.no_grad()
def validate(model, base, rows, device, ks=VAL_KS):
    """Staff metrics on full tiles at each K, fp32, no crop offset: the same as predict.py."""
    model.eval()
    model.pos_offset = None
    totals = {k: Counts() for k in ks}
    kmax = max(ks)
    for row in rows:
        image = load_rgb(row["image"], device)
        with Image.open(row["label"]) as im:
            labels = np.asarray(im)
        features, y = base(image)
        outputs = {0: y}
        if kmax > 0:
            state = model.initialize(features, y)
            for t in range(1, kmax + 1):
                state, y = model.step(features, state, y)
                if t in ks:
                    outputs[t] = y
        for k in ks:
            prob = outputs[k].float().sigmoid()[0, 0].cpu().numpy()
            totals[k] = totals[k] + image_counts(prob, labels)
    model.train()
    return {k: summarize(c) for k, c in totals.items()}


def selection_score(ordinary, stress=None):
    """Leaderboard formula in [0, 1]; without a stress set, the ordinary part renormalized."""
    a1, a2, a4 = (ordinary[k]["quality"] for k in (1, 2, 4))
    if stress is not None:
        return leaderboard_score(a1, a2, a4, stress[4]["quality"]) / 100
    return (0.2 * a1 + 0.2 * a2 + 0.4 * a4) / 0.8


def print_val(results, label, stress=None):
    line = "  ".join(f"K={k}: {100 * results[k]['quality']:.2f}" for k in sorted(results))
    extra = f"  stress K=4: {100 * stress[4]['quality']:.2f}" if stress is not None else ""
    print(f"  val {label} quality  {line}{extra}  | K=4 IoU {100 * results[4]['road_iou']:.2f}"
          f"  clDice {100 * results[4]['cldice']:.2f}  BIoU {100 * results[4]['boundary_iou']:.2f}",
          flush=True)


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description="Train the refiner in submission/model.py")
    p.add_argument("--data-root", type=Path, help="LoveDA training split, e.g. .../dataset/Train")
    p.add_argument("--baseline", type=Path, help="Frozen staff baseline checkpoint")
    p.add_argument("--demo-baseline", action="store_true", help="Real data + staff DemoBaseline")
    p.add_argument("--demo", action="store_true", help="Random data + demo baseline; checks the code only")
    p.add_argument("--fraction", type=float, default=1.0, help="Random fraction of training images")
    p.add_argument("--max-images", type=int, help="At most this many training images")
    p.add_argument("--output", type=Path, default=Path("weights/refiner.safetensors"))
    p.add_argument("--resume", type=Path, help="<name>.resume.pt written by a previous run")
    p.add_argument("--steps", type=int, default=4, choices=range(1, 9), metavar="K",
                   help="Unrolled steps during training (loss on every step)")
    p.add_argument("--step-weights", help="Comma list, one per step, e.g. 1,1,1,2 (default)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=512, help="Training crop size in pixels")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=500, help="Linear warmup iterations")
    p.add_argument("--pos-weight", type=float, default=1.0,
                   help="BCE weight on road pixels. Keep 1 with Dice; >1 over-predicts road at threshold 0.5")
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--stress-aug", type=float, default=0.3,
                   help="Probability of a shadow/occlusion/blur corruption per training tile (0 = off)")
    p.add_argument("--val-manifest", type=Path, help="Ordinary validation manifest (make_manifest.py)")
    p.add_argument("--val-stress-manifest", type=Path,
                   help="Stress version of it (make_stress.py); enables the full leaderboard score")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    weights = parse_step_weights(args.step_weights, args.steps)

    # Frozen baseline.
    use_demo_baseline = args.demo or args.demo_baseline
    if not use_demo_baseline and args.baseline is None:
        p.error("--baseline is required unless --demo or --demo-baseline")
    base = load_baseline(None if use_demo_baseline else args.baseline, device, demo=use_demo_baseline)
    base.eval()
    base.requires_grad_(False)

    # Data.
    if args.demo:
        dataset, workers = demo_dataset(), 0
    else:
        if args.data_root is None:
            p.error("--data-root is required unless --demo")
        if "val" in {part.lower() for part in args.data_root.resolve().parts}:
            p.error(f"Refusing to train on a validation folder: {args.data_root}")
        transform = StressAugment(args.stress_aug) if args.stress_aug > 0 else None
        dataset, workers = loveda_dataset(args.data_root, transform), args.workers
    dataset = take_subset(dataset, args.fraction, args.max_images, args.seed)
    print(f"{len(dataset)} training images")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers,
                        pin_memory=device.type == "cuda", drop_last=len(dataset) > args.batch_size,
                        persistent_workers=workers > 0)

    # Model.
    model = build_refiner(base.feature_channels).to(device)
    n_total = sum(q.numel() for q in model.parameters())
    n_core = sum(q.numel() for q in model.core.parameters())
    print(f"refiner params {n_total:,} (core {n_core:,}); proposed caps: core 5M, total 10M")
    if n_core > CORE_LIMIT or n_total > TOTAL_LIMIT:
        print("WARNING: over the proposed parameter caps")

    decay = [q for n, q in model.named_parameters() if q.ndim >= 2 and "queries" not in n]
    no_decay = [q for n, q in model.named_parameters() if q.ndim < 2 or "queries" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    total_iters = max(1, args.epochs * len(loader))
    warmup = min(args.warmup, max(1, total_iters // 10))

    def lr_factor(it):
        if it < warmup:
            return (it + 1) / warmup
        progress = min(1.0, (it - warmup) / max(1, total_iters - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    # Paths and resume.
    stem = args.output.with_suffix("")
    best_path = stem.with_name(stem.name + "_best.safetensors")
    resume_path = stem.with_name(stem.name + ".resume.pt")
    start_epoch, best_score, best_epoch = 0, -1.0, None
    if args.resume:
        ckpt = torch.load(str(args.resume), map_location=device, weights_only=False)  # our own file
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch, best_score, best_epoch = ckpt["epoch"], ckpt["best_score"], ckpt["best_epoch"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    val_rows = load_manifest(args.val_manifest) if args.val_manifest else None
    stress_rows = load_manifest(args.val_stress_manifest) if args.val_stress_manifest else None
    if stress_rows and not val_rows:
        p.error("--val-stress-manifest needs --val-manifest")
    if val_rows and start_epoch == 0:
        print(f"Validating on {len(val_rows)} images"
              + (f" (+ {len(stress_rows)} stress)" if stress_rows else ""))
        start = validate(model, base, val_rows, device)
        start_stress = validate(model, base, stress_rows, device, ks=(0, 4)) if stress_rows else None
        print_val(start, "before training", start_stress)
        print(f"  baseline alone (K=0): quality {100 * start[0]['quality']:.2f}  <- the number to beat")

    # Train.
    latest = args.output.with_suffix(".safetensors")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total = 0.0
        for batch in loader:
            image, target, valid = (t.to(device, non_blocking=True) for t in batch[:3])
            image, target, valid = augment_tiles(image, target, valid)
            with torch.no_grad():
                features, initial_logits = base(image)                       # full tile, fp32
            features, initial_logits, target, valid, offsets = random_crops(
                features, initial_logits, target, valid, args.crop, model.patch)
            model.pos_offset = offsets                                        # true crop position
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = unroll(model, features, initial_logits, args.steps)
            loss = sum(w * seg_loss(y, target, valid, args.pos_weight, args.dice_weight)
                       for w, y in zip(weights, outputs))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model.pos_offset = None
            total += loss.item()
        peak = (f", peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
                if device.type == "cuda" else "")
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / max(len(loader), 1):.4f}, "
              f"lr {sched.get_last_lr()[0]:.2e}{peak}", flush=True)
        latest = save_weights(model, args.output)

        if val_rows:
            results = validate(model, base, val_rows, device)
            stress = validate(model, base, stress_rows, device, ks=(4,)) if stress_rows else None
            print_val(results, f"epoch {epoch + 1}", stress)
            score = selection_score(results, stress)
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                save_weights(model, best_path)
                print(f"  new best (leaderboard-style score {100 * score:.2f}) -> {best_path}", flush=True)
        save_resume(resume_path, model, opt, sched, epoch + 1, best_score, best_epoch)

    print(f"Saved {latest}" + (" (DEMO ONLY: random data)" if args.demo else ""))
    if val_rows:
        print(f"Best epoch {best_epoch}: score {100 * best_score:.2f} -> {best_path}")


if __name__ == "__main__":
    main()
