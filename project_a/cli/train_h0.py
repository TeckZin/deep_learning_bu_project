#!/usr/bin/env python3
"""Train the H0 head(s) on the frozen encoder (doc stage 4) for one validation fold.

The encoder stays frozen; only small H0 heads are trained. Which heads, and on which
images, comes from splits.fold_plan (--h0-scheme):
    oof     h0_oof<j> for each training fold j (trained without fold j) + h0_all
    halves  h0_a (half a of the training folds)
All heads train in ONE pass over the data: the encoder runs once per batch and each
head takes a loss only on the images in its own training set. So OOF costs about
the same as training a single head.

Writes to --out-dir (default weights/h0/<encoder>_<scheme>/fold<val>):
    h0_<name>.safetensors   best validation epoch (or the last epoch without validation)
    plan.json               scheme, val fold, encoder, head names; train.py checks it

    python train_h0.py --val-fold 0          # $ROADSEG_DATA, splits/folds.csv
    python train.py --val-fold 0 ...         # finds weights/h0/dinov2_vits14_oof/fold0
"""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from frozen_encoder import FrozenEncoder, H0Head, build_encoder, save_head
from data import TileDataset, augment_tiles, subset_rows
from evaluation import format_summary, validate_heads
from losses import seg_loss
from splits import DEFAULT_FOLDS, default_data_root, fold_plan, read_folds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=default_data_root(), help="Default: $ROADSEG_DATA")
    p.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    p.add_argument("--val-fold", type=int, default=0)
    p.add_argument("--h0-scheme", choices=["oof", "halves"], default="oof")
    p.add_argument("--only", action="append", help="Train only these head names (repeatable)")
    p.add_argument("--encoder", default="dinov2_vits14", help="dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | pool")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--fraction", type=float, default=1.0, help="Random fraction of the training images")
    p.add_argument("--max-images", type=int, help="At most this many training images")
    p.add_argument("--val-max-images", type=int, help="Validate on at most this many fold images")
    p.add_argument("--val-every", type=int, default=1, help="0 = no validation (keep the last epoch)")
    p.add_argument("--out-dir", type=Path, help="Default: weights/h0/<encoder>_<scheme>/fold<val-fold>")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.data_root is None:
        p.error("--data-root or $ROADSEG_DATA is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    out_dir = args.out_dir or Path("weights/h0") / f"{args.encoder}_{args.h0_scheme}" / f"fold{args.val_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = fold_plan(read_folds(args.folds, args.data_root), args.val_fold, args.h0_scheme)
    names = [n for n in plan["h0"] if not args.only or n in args.only]
    if not names:
        p.error(f"--only matches none of {list(plan['h0'])}")
    members = {}                                          # image id -> indices of heads that train on it
    for i, name in enumerate(names):
        for r in plan["h0"][name]:
            members.setdefault(r["id"], []).append(i)
    rows = [r for r in plan["refiner"] + sum(plan["h0"].values(), []) if r["id"] in members]
    rows = subset_rows(list({r["id"]: r for r in rows}.values()), args.fraction, args.max_images, args.seed)
    val_rows = subset_rows(plan["val"], 1.0, args.val_max_images, args.seed)

    encoder = build_encoder(args.encoder).to(device)
    heads = [H0Head(encoder.channels, args.hidden) for _ in names]
    base = FrozenEncoder(encoder, heads).to(device)
    print(f"encoder {args.encoder} (frozen, {encoder.channels} ch); scheme {args.h0_scheme}, val fold "
          f"{args.val_fold}; heads " + ", ".join(f"{n} ({len(plan['h0'][n])} imgs)" for n in names)
          + f"; {sum(q.numel() for q in heads[0].parameters()):,} params each")
    print(f"{len(rows)} training images, {len(val_rows)} validation images")

    loader = DataLoader(TileDataset(rows), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        drop_last=len(rows) > args.batch_size, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW([q for h in heads for q in h.parameters()], lr=args.lr,
                            weight_decay=args.weight_decay)
    total_iters = max(1, args.epochs * len(loader))
    warmup = min(args.warmup, max(1, total_iters // 10))

    def lr_factor(it):
        if it < warmup:
            return (it + 1) / warmup
        progress = min(1.0, (it - warmup) / max(1, total_iters - warmup))
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    best = [-1.0] * len(names)
    meta = {"scheme": args.h0_scheme, "val_fold": str(args.val_fold)}

    def save(i):
        return save_head(heads[i], out_dir / f"h0_{names[i]}.safetensors", args.encoder,
                         {**meta, "name": names[i]})

    for epoch in range(args.epochs):
        for h in heads:
            h.train()
        totals = [0.0] * len(names)
        for image, target, valid, ids in loader:
            image, target, valid = (t.to(device, non_blocking=True) for t in (image, target, valid))
            image, target, valid = augment_tiles(image, target, valid)
            f = base.features(image)                                    # frozen, no gradients
            loss = 0.0
            for i, h in enumerate(heads):
                idx = [b for b, image_id in enumerate(ids) if i in members[image_id]]
                if idx:
                    l_i = seg_loss(h(f[idx], image.shape[-2:]), target[idx], valid[idx],
                                   args.pos_weight, args.dice_weight)
                    loss = loss + l_i                                   # heads are independent
                    totals[i] += l_i.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for h in heads:
                torch.nn.utils.clip_grad_norm_(h.parameters(), 1.0)
            opt.step()
            sched.step()
        n = max(len(loader), 1)
        print(f"epoch {epoch + 1}/{args.epochs}: lr {sched.get_last_lr()[0]:.2e}  loss "
              + "  ".join(f"{name} {t / n:.4f}" for name, t in zip(names, totals)), flush=True)

        last = epoch + 1 == args.epochs
        if val_rows and args.val_every > 0 and ((epoch + 1) % args.val_every == 0 or last):
            for i, s in enumerate(validate_heads(base, val_rows, device)):
                tag = ""
                if s["quality"] > best[i]:
                    best[i] = s["quality"]
                    save(i)
                    tag = "  <- best"
                print(f"  val h0_{names[i]}: {format_summary(s)}{tag}", flush=True)
        elif last or not val_rows or args.val_every == 0:
            for i in range(len(names)):
                save(i)

    plan_file = out_dir / "plan.json"
    old = json.loads(plan_file.read_text()) if plan_file.exists() else {}
    if old.get("scheme") not in (None, args.h0_scheme) or old.get("val_fold") not in (None, args.val_fold):
        old = {}
    heads_done = sorted(set(old.get("heads", [])) | set(names))
    plan_file.write_text(json.dumps({"scheme": args.h0_scheme, "val_fold": args.val_fold,
                                     "encoder": args.encoder, "folds": str(args.folds),
                                     "heads": heads_done, "eval_h0": plan["eval_h0"]}, indent=1))
    print(f"Saved {', '.join(f'h0_{n}' for n in names)} to {out_dir}"
          + (f"; best val quality " + ", ".join(f"{n} {100 * b:.2f}" for n, b in zip(names, best))
             if best[0] >= 0 else ""))


if __name__ == "__main__":
    main()
