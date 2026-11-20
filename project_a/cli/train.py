#!/usr/bin/env python3
"""Train the refiner (submission/) on one cross-validation fold.

    python train_h0.py --val-fold 0                                   # H0 heads first (once per fold)
    python train.py --config experiments/attempt7.json --val-fold 0   # then the refiner

What one training step does, and why:
  1. Load full tiles of the training folds; random flip / 90-degree rotation per tile;
     optional stress augmentation (public make_stress.py recipe).
  2. Run the frozen encoder + H0 on the FULL tile, exactly like inference.
     With --h0-scheme oof, each tile's y0 comes from the H0 that never saw its fold.
  3. Take a random crop of features, y0 and labels, aligned to the patch grid, and tell
     the model where the crop sits in the tile (model.pos_offset).
  4. initialize() once, step() K times; loss = weighted sum over every step.
  5. AdamW + linear warmup + cosine decay, bf16 autocast, gradient clipping.

Choosing what to train:
  --config FILE      JSON of defaults for any flag below (dest names, e.g. "cldice_weight"),
                     plus optional "fold_overrides": {"3": {"lr": 1e-4}} applied when
                     --val-fold is 3. Flags on the command line override the file.
  --arch NAME        architecture preset from submission/config.py (default: ACTIVE)
  --set KEY=VALUE    architecture override, repeatable, e.g. --set core.layers=6

Validation on the held-out fold at K = 0, 1, 2, 4 (+ stress K = 4 with
--val-stress-manifest) keeps the best leaderboard-style score.

Writes to weights/<run>/fold<val>/ (or next to --output):
    refiner.safetensors        latest epoch; metadata holds the architecture config
    refiner_best.safetensors   best validation score
    refiner.resume.pt          for --resume
    run.json                   all settings, git commit, H0 used for validation/inference
"""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from frozen_encoder import load_frozen_encoder
from data import DemoDataset, StressAugment, TileDataset, augment_tiles, degrade_logits, random_crops, subset_rows
from evaluation import print_val, selection_score, validate
from losses import seg_loss
from splits import DEFAULT_FOLDS, default_data_root, fold_plan, read_folds
from submission.config import ACTIVE, apply_overrides, config_to_json, get_config
from submission.image import ImagePackedBase
from submission.model import build_refiner

CORE_LIMIT, TOTAL_LIMIT = 5_000_000, 10_000_000      # proposed caps from the handout


def unroll(model, features, initial_logits, steps, detach_y=False):
    """Same loop as inference: initialize once, then call step() K times.

    detach_y=True (RAFT-style): the fed-back answer y_t is treated as a constant input,
    so each step is trained to improve whatever answer it is handed. Gradients still flow
    between steps through the memory z.
    """
    state = model.initialize(features, initial_logits)
    current = initial_logits
    outputs = []
    for _ in range(steps):
        state, current = model.step(features, state, current.detach() if detach_y else current)
        outputs.append(current)
    return outputs


def parse_step_weights(text, steps):
    """Per-step loss weights, normalized to sum to 1. Default: 1 per step, 2 on step 4."""
    if text is None:
        weights = [1.0] * steps
        if steps >= 4:
            weights[3] = 2.0
    else:
        weights = [float(x) for x in str(text).split(",")]
    if len(weights) != steps or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError(f"--step-weights needs {steps} non-negative numbers, got {text!r}")
    total = sum(weights)
    return [w / total for w in weights]


def save_weights(model, path, metadata):
    """Safetensors weights (no pickle) + string metadata, e.g. the architecture config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata=metadata)
    return path


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except OSError:
        return None


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="JSON file of defaults (see above)")
    p.add_argument("--run", default="dev", help="Run name: outputs go to weights/<run>/fold<val>/")
    p.add_argument("--output", type=Path, help="Override the refiner checkpoint path")
    # architecture
    p.add_argument("--arch", default=ACTIVE, help="Preset name in submission/config.py")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="Architecture override, e.g. core.layers=6 (repeatable)")
    # data / folds / frozen encoder + H0
    p.add_argument("--data-root", type=Path, default=default_data_root(), help="Default: $ROADSEG_DATA")
    p.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    p.add_argument("--val-fold", type=int, default=0)
    p.add_argument("--h0-scheme", choices=["oof", "halves"], default="oof")
    p.add_argument("--encoder", default="dinov2_vits14",
                   help="Encoder of the H0 heads: picks the default --h0-dir, checked against its plan.json")
    p.add_argument("--h0-dir", type=Path,
                   help="train_h0.py output for this fold (default weights/h0/<encoder>_<scheme>/fold<val>)")
    p.add_argument("--demo", action="store_true", help="Random data + staff demo baseline; checks the code only")
    p.add_argument("--fraction", type=float, default=1.0, help="Random fraction of training images")
    p.add_argument("--max-images", type=int, help="At most this many training images")
    p.add_argument("--val-max-images", type=int, help="Validate on at most this many fold images")
    p.add_argument("--val-every", type=int, default=1,
                   help="Validate every N epochs (always after the last). 0 = only before and after training")
    p.add_argument("--val-stress-manifest", type=Path,
                   help="Stress copy of the val fold (splits.py manifest --stress); enables the full score")
    # training recipe
    p.add_argument("--y0-aug", type=float, default=0.0,
                   help="Probability per tile of degrading y0 (refiner input only), e.g. 0.5")
    p.add_argument("--detach-y", action="store_true",
                   help="Do not backpropagate through the fed-back answer y_t (RAFT-style)")
    p.add_argument("--resume", type=Path, help="refiner.resume.pt written by a previous run")
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
    p.add_argument("--cldice-weight", type=float, default=0.0, help="Soft-clDice loss weight (0 = off)")
    p.add_argument("--edge-weight", type=float, default=1.0,
                   help="BCE multiplier on pixels within 3 px of a road edge (1 = off), e.g. 4")
    p.add_argument("--stress-aug", type=float, default=0.3,
                   help="Probability of a shadow/occlusion/blur corruption per training tile (0 = off)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def parse_args(argv=None):
    """Command line > fold_overrides[val fold] > config file > built-in defaults."""
    p = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    pre.add_argument("--val-fold", type=int)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        file_cfg = json.loads(known.config.read_text())
        fold_overrides = file_cfg.pop("fold_overrides", {})
        val_fold = known.val_fold if known.val_fold is not None else file_cfg.get("val_fold", 0)
        file_cfg.update(fold_overrides.get(str(val_fold), {}))
        dests = {a.dest for a in p._actions}
        unknown = sorted(set(file_cfg) - dests - {"comment"})
        if unknown:
            p.error(f"{known.config}: unknown keys {unknown}")
        file_cfg.pop("comment", None)
        for key in ("data_root", "folds", "h0_dir", "output", "resume", "val_stress_manifest"):
            if file_cfg.get(key) is not None:
                file_cfg[key] = Path(file_cfg[key])
        p.set_defaults(**file_cfg)
    return p, p.parse_args(argv)


def check_h0_dir(args, p):
    """The train_h0.py output folder for this fold, after checking it matches scheme and fold."""
    if args.data_root is None:
        p.error("--data-root or $ROADSEG_DATA is required unless --demo")
    h0_dir = args.h0_dir or Path("weights/h0") / f"{args.encoder}_{args.h0_scheme}" / f"fold{args.val_fold}"
    plan_file = h0_dir / "plan.json"
    if not plan_file.exists():
        p.error(f"{plan_file} not found: run train_h0.py --val-fold {args.val_fold} "
                f"--h0-scheme {args.h0_scheme} --encoder {args.encoder} first")
    saved = json.loads(plan_file.read_text())
    if saved["scheme"] != args.h0_scheme or saved["val_fold"] != args.val_fold:
        p.error(f"{h0_dir} was trained for scheme {saved['scheme']} / fold {saved['val_fold']}, "
                f"not {args.h0_scheme} / fold {args.val_fold}")
    if args.h0_dir is None and saved.get("encoder") != args.encoder:
        p.error(f"{h0_dir} holds {saved.get('encoder')} heads, not {args.encoder}")
    return h0_dir


# Settings that may differ when the same run is resumed (machine, job, code version).
VOLATILE_KEYS = {"resume", "workers", "device", "git_commit", "config"}


def check_run_dir(run_file, record, args, p, resume_path):
    """Refuse to write into a run folder that belongs to a different run, or to restart one silently.

    Run folders are shared (runs/ on the SCC), and jobs/train.sh resumes automatically
    when a resume file exists, so two people using the same --run would otherwise
    continue each other's training.
    """
    if not run_file.exists():
        return
    old = json.loads(run_file.read_text())
    new = json.loads(json.dumps(record))                   # same types as the saved file
    diff = sorted(k for k in set(old) | set(new) if k not in VOLATILE_KEYS and old.get(k) != new.get(k))
    if diff:
        p.error(f"{run_file.parent} already holds a run with different settings ({', '.join(diff)}); "
                f"choose a new --run")
    if not args.resume:
        p.error(f"{run_file.parent} already holds this run: pass --resume {resume_path} to continue it, "
                f"or choose a new --run")


def main(argv=None):
    p, args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    weights = parse_step_weights(args.step_weights, args.steps)
    cfg = apply_overrides(get_config(args.arch), args.set)
    out = args.output or Path("weights") / args.run / f"fold{args.val_fold}" / "refiner.safetensors"
    out = out.with_suffix(".safetensors")

    # Data and the frozen encoder + H0 head(s).
    eval_h0_path, route = None, None
    if args.demo:
        from baseline import DemoBaseline
        train_base = eval_base = DemoBaseline().to(device).eval()
        dataset, val_rows, workers = DemoDataset(), [], 0
    else:
        h0_dir = check_h0_dir(args, p)
        plan = fold_plan(read_folds(args.folds, args.data_root), args.val_fold, args.h0_scheme)
        train_names = sorted(set(plan["route"].values()))
        train_base = load_frozen_encoder([h0_dir / f"h0_{n}.safetensors" for n in train_names], device)
        eval_h0_path = h0_dir / f"h0_{plan['eval_h0']}.safetensors"
        eval_base = load_frozen_encoder(eval_h0_path, device, encoder=train_base.encoder)
        route = {image_id: train_names.index(name) for image_id, name in plan["route"].items()}
        rows = subset_rows(plan["refiner"], args.fraction, args.max_images, args.seed)
        transform = StressAugment(args.stress_aug) if args.stress_aug > 0 else None
        dataset, workers = TileDataset(rows, transform), args.workers
        val_rows = subset_rows(plan["val"], 1.0, args.val_max_images, args.seed)
        print(f"fold {args.val_fold} ({args.h0_scheme}): y0 for training from "
              f"{', '.join('h0_' + n for n in train_names)}; validation/inference H0 {eval_h0_path.name}")
    if cfg.image.use:       # the refiner sees the image inside the loop: pack it into f
        train_base = ImagePackedBase(train_base, cfg.image.stride)
        eval_base = ImagePackedBase(eval_base, cfg.image.stride)
    for b in (train_base, eval_base):
        b.requires_grad_(False)
    print(f"{len(dataset)} training images, {len(val_rows)} validation images")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=workers,
                        pin_memory=device.type == "cuda", drop_last=len(dataset) > args.batch_size,
                        persistent_workers=workers > 0)

    # Model.
    model = build_refiner(train_base.feature_channels, cfg).to(device)
    n_total = sum(q.numel() for q in model.parameters())
    n_core = sum(q.numel() for q in model.core.parameters())
    print(f"arch {args.arch}" + (f" + {args.set}" if args.set else "")
          + f": refiner params {n_total:,} (core {n_core:,}); proposed caps: core 5M, total 10M")
    if n_core > CORE_LIMIT or n_total > TOTAL_LIMIT:
        print("WARNING: over the proposed parameter caps")
    meta = {"refiner_cfg": config_to_json(cfg), "arch": args.arch, "overrides": json.dumps(args.set),
            "h0_scheme": args.h0_scheme, "val_fold": str(args.val_fold),
            "eval_h0": str(eval_h0_path) if eval_h0_path else "demo"}

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

    # Paths, run record, resume.
    best_path = out.with_name(out.stem + "_best.safetensors")
    resume_path = out.with_name(out.stem + ".resume.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    record.update({"refiner_cfg": asdict(cfg), "git_commit": git_commit(), "eval_h0": meta["eval_h0"],
                   "params": n_total, "core_params": n_core})
    check_run_dir(out.with_name("run.json"), record, args, p, resume_path)
    out.with_name("run.json").write_text(json.dumps(record, indent=1))
    start_epoch, best_score, best_epoch = 0, -1.0, None
    if args.resume:
        ckpt = torch.load(str(args.resume), map_location=device, weights_only=False)  # our own file
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch, best_score, best_epoch = ckpt["epoch"], ckpt["best_score"], ckpt["best_epoch"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    stress_rows = None
    if args.val_stress_manifest:
        from common import load_manifest
        stress_rows = load_manifest(args.val_stress_manifest)
    if val_rows and start_epoch == 0:
        start = validate(model, eval_base, val_rows, device)
        start_stress = validate(model, eval_base, stress_rows, device, ks=(0, 4)) if stress_rows else None
        print_val(start, "before training", start_stress)
        print(f"  frozen encoder + H0 alone (K=0): quality {100 * start[0]['quality']:.2f}  <- the number to beat")

    # Train.
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total = base_total = in_total = 0.0
        step_totals = [0.0] * args.steps                   # loss of y1 .. yK
        for image, target, valid, ids in loader:
            image, target, valid = (t.to(device, non_blocking=True) for t in (image, target, valid))
            image, target, valid = augment_tiles(image, target, valid)
            with torch.no_grad():                                             # full tile, fp32
                if route is None:
                    features, initial_logits = train_base(image)
                else:
                    features, initial_logits = train_base(image, [route[i] for i in ids])
            features, initial_logits, target, valid, offsets = random_crops(
                features, initial_logits, target, valid, args.crop, model.patch)
            model.pos_offset = offsets                                        # true crop position
            refiner_in = degrade_logits(initial_logits, args.y0_aug, model.patch)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                outputs = unroll(model, features, refiner_in, args.steps, args.detach_y)
            loss_args = (args.pos_weight, args.dice_weight, args.edge_weight, args.cldice_weight)
            step_losses = [seg_loss(y, target, valid, *loss_args) for y in outputs]
            loss = sum(w * l for w, l in zip(weights, step_losses))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            model.pos_offset = None
            total += loss.item()
            with torch.no_grad():
                base_total += seg_loss(initial_logits, target, valid, *loss_args).item()
                in_total += seg_loss(refiner_in, target, valid, *loss_args).item()
            for i, l in enumerate(step_losses):
                step_totals[i] += l.item()
        n = max(len(loader), 1)
        peak = (f", peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
                if device.type == "cuda" else "")
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / n:.4f}, "
              f"lr {sched.get_last_lr()[0]:.2e}{peak}", flush=True)
        print("  train loss by step:  y0 " + f"{base_total / n:.4f} (input {in_total / n:.4f})"
              + "".join(f" | y{i + 1} {s / n:.4f}" for i, s in enumerate(step_totals)), flush=True)
        save_weights(model, out, meta)

        last = epoch + 1 == args.epochs
        if val_rows and (last or (args.val_every > 0 and (epoch + 1) % args.val_every == 0)):
            results = validate(model, eval_base, val_rows, device)
            stress = validate(model, eval_base, stress_rows, device, ks=(4,)) if stress_rows else None
            print_val(results, f"epoch {epoch + 1}", stress)
            score = selection_score(results, stress)
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                save_weights(model, best_path, {**meta, "epoch": str(best_epoch)})
                print(f"  new best (leaderboard-style score {100 * score:.2f}) -> {best_path}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch + 1, "best_score": best_score, "best_epoch": best_epoch},
                   str(resume_path))

    print(f"Saved {out}" + (" (DEMO ONLY: random data)" if args.demo else ""))
    if val_rows:
        print(f"Best epoch {best_epoch}: score {100 * best_score:.2f} -> {best_path}")


if __name__ == "__main__":
    main()
