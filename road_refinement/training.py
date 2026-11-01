"""Short, resumable two-stage training; no labels enter inference methods."""
from contextlib import nullcontext
import importlib.metadata
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Subset

from project_a.cli.submission.model import build_refiner
from .baseline import TemporaryBaseline, initialize_encoder, load_weights, save_weights
from .common import PROTOCOL, git_revision, read_json, seed_all, sha256, source_hash, write_json
from .data import RoadCrops, discover, masked_bce
from .inference import evaluate_tiles


def append_log(path, value):
    import json
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
    print(value, flush=True)


def train(args, device):
    seed_all(args.seed)
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=True)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    tiles = discover(args.data_root, "Train", args.train_per_domain)
    validation = discover(args.data_root, "Val", args.val_per_domain)
    overlap = {t.image.resolve() for t in tiles} & {t.image.resolve() for t in validation}
    if overlap:
        raise ValueError("Training and validation resolve to overlapping image files")
    selection = {"train": [t.record() for t in tiles], "validation": [t.record() for t in validation]}
    config = {
        "protocol": PROTOCOL, "baseline_kind": "temporary_resnet18_unet", "official_baseline": False,
        "synthetic": args.synthetic, "seed": args.seed, "crop_size": args.crop_size,
        "batch_size": args.batch_size, "train_per_domain": args.train_per_domain,
        "val_per_domain": args.val_per_domain, "data_root": str(Path(args.data_root).resolve()),
        "selection": selection, "git_revision": git_revision(), "source_sha256": source_hash(),
        "architecture": {"encoder": "resnet18", "decoder": "unet", "feature_channels": 256,
                         "feature_stride": 16, "token_width": 256, "core_layers": 1, "heads": 8},
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "labels": {"road": 3, "ignore": 0},
        "training": {"baseline_updates": args.baseline_updates, "refiner_updates": args.refiner_updates,
                     "save_every": args.save_every, "workers": args.workers,
                     "mixed_precision": device.type == "cuda" and not args.no_amp,
                     "baseline_lr": 1e-3, "refiner_lr": 1e-4, "weight_decay": 1e-4,
                     "gradient_clip": 1.0, "training_steps": [1, 2, 4]},
        "environment": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "segmentation-models-pytorch", "numpy", "scikit-image", "safetensors")},
    }
    config_path = run / "config.json"
    if config_path.exists():
        previous = read_json(config_path)
        for key in ("synthetic", "seed", "crop_size", "batch_size", "data_root", "selection", "source_sha256", "environment"):
            if config[key] != previous[key]:
                raise ValueError(f"Run configuration changed ({key}); use a new --run-dir")
    else:
        write_json(config_path, config)
    append_log(run / "invocations.jsonl", {"stage": args.stage, "resume": args.resume,
               "baseline_updates": args.baseline_updates, "refiner_updates": args.refiner_updates,
               "workers": args.workers, "save_every": args.save_every, "device": str(device),
               "mixed_precision": device.type == "cuda" and not args.no_amp})
    stages = ("baseline", "refiner") if args.stage == "both" else (args.stage,)
    for stage in stages:
        stage_train(args, device, run, stage, tiles, validation)
    return run


def stage_train(args, device, run, stage, tiles, validation):
    # Stage initialization is independent of whether baseline training happened
    # in this process or in an earlier invocation.
    seed_all(args.seed + (1 if stage == "refiner" else 0))
    checkpoints = run / "checkpoints"
    resume_path = checkpoints / f"{stage}_trainer.pt"
    if resume_path.exists() and not args.resume:
        raise ValueError(f"{stage} already has a checkpoint; pass --resume or use a new --run-dir")
    baseline = TemporaryBaseline()
    refiner = build_refiner(baseline.feature_channels)
    total = args.baseline_updates if stage == "baseline" else args.refiner_updates
    if stage == "baseline":
        if not resume_path.exists():
            if args.synthetic:
                print("SYNTHETIC TEST: random encoder initialization, not a course-data result", flush=True)
            else:
                if not args.encoder_weights or not Path(args.encoder_weights).is_file():
                    raise ValueError("Missing ImageNet encoder weights. Run prepare, then pass --encoder-weights PATH")
                initialize_encoder(baseline, args.encoder_weights)
                write_json(run / "encoder.json", {"source": "torchvision ResNet18 IMAGENET1K_V1",
                                                  "sha256": sha256(args.encoder_weights)})
        baseline.freeze_encoder()
        model, learning_rate = baseline, 1e-3
    else:
        baseline_path = Path(args.baseline) if args.baseline else checkpoints / "baseline_best.safetensors"
        if not baseline_path.is_file():
            raise ValueError("Train baseline first or pass --baseline PATH")
        load_weights(baseline, baseline_path)
        baseline.eval().requires_grad_(False)
        baseline_hash = sha256(baseline_path)
        provenance = run / "refiner_baseline.json"
        if provenance.exists() and read_json(provenance)["sha256"] != baseline_hash:
            raise ValueError("Baseline changed since refiner training; use a new run directory")
        write_json(provenance, {"path": str(baseline_path.resolve()), "sha256": baseline_hash})
        # Store the actual frozen baseline used by this refiner for export.
        save_weights(baseline, checkpoints / "refiner_baseline.safetensors")
        model, learning_rate = refiner, 1e-4
    baseline.to(device)
    refiner.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=learning_rate, weight_decay=1e-4)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, previous_seconds = 0, -1.0, 0.0
    peak_training = 0 if device.type == "cuda" else None
    if args.resume and resume_path.exists():
        # Private optimizer/RNG state produced by this trainer, never exported.
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["source_sha256"] != source_hash():
            raise ValueError("Source changed since checkpoint; use a new run directory")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start, best = state["step"], state["best"]
        previous_seconds = state["seconds"]
        peak_training = state.get("peak_training_cuda_bytes", peak_training)
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    if start >= total:
        print(f"{stage}: already completed {start}/{total} updates", flush=True)
        return
    crops = RoadCrops(tiles, args.crop_size, total * args.batch_size, args.seed)
    # Index-based crops avoid replaying skipped batches when resuming.
    samples = Subset(crops, range(start * args.batch_size, total * args.batch_size))
    loader = DataLoader(samples, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda", shuffle=False,
                        generator=torch.Generator().manual_seed(args.seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    for step, (images, labels) in enumerate(loader, start=start + 1):
        if device.type == "cuda":
            # Exclude peaks from the previous full-tile validation pass.
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        if stage == "refiner":
            baseline.eval()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            if stage == "baseline":
                _, logits = baseline(images)
                loss = masked_bce(logits, labels)
                k = 0
            else:
                with torch.no_grad():
                    features, logits = baseline(images)
                memory = refiner.initialize(features, logits)
                k = random.Random(args.seed + step).choice((1, 2, 4))
                losses = []
                for _ in range(k):
                    memory, logits = refiner.step(features, memory, logits)
                    losses.append(masked_bce(logits, labels))
                loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss at {stage} update {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            peak_training = max(peak_training or 0, torch.cuda.max_memory_allocated(device))
        seconds = previous_seconds + time.perf_counter() - began
        if step == start + 1 or step % 10 == 0 or step == total:
            append_log(run / "metrics.jsonl", {
                "stage": stage, "step": step, "loss": loss.item(), "steps": k,
                "gradient_norm": float(norm), "seconds": seconds,
            })
        if step % args.save_every == 0 or step == total:
            report = evaluate_tiles(baseline, refiner, validation, device, 0 if stage == "baseline" else 4)
            quality = report["overall"]["road_iou"]
            save_weights(model, checkpoints / f"{stage}_latest.safetensors")
            if quality > best:
                best = quality
                save_weights(model, checkpoints / f"{stage}_best.safetensors")
            seconds = previous_seconds + time.perf_counter() - began
            snapshot = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "best": best, "seconds": seconds, "source_sha256": source_hash(),
                "peak_training_cuda_bytes": peak_training,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }
            temporary = resume_path.with_suffix(".tmp")
            torch.save(snapshot, temporary)
            temporary.replace(resume_path)
            write_json(run / f"{stage}_validation.json", report)
            append_log(run / "metrics.jsonl", {
                "stage": stage, "step": step, "validation": report["overall"], "validation_bce": report["bce"],
                "peak_training_cuda_bytes": peak_training, "seconds": seconds,
                "gpu_hours": seconds / 3600 if device.type == "cuda" else 0.0,
            })
