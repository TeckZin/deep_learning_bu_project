"""Command-line entry points used on laptops, SCC, and exported submissions."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys

import torch

from .common import DEFAULT_DATA, PROTOCOL, device_for, read_json, write_json


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_data(parser):
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", DEFAULT_DATA), type=Path)
    parser.add_argument("--train-per-domain", default=10, type=int, help="0 uses all training tiles")
    parser.add_argument("--val-per-domain", default=2, type=int, help="0 uses all validation tiles")


def add_device(parser):
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--cpu-threads", type=positive, default=int(os.environ.get("NSLOTS", "4")))
    commands = root.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Check environment and selected dataset pairs without training")
    add_data(check)
    check.add_argument("--require-cuda", action="store_true", help="Use only inside an allocated GPU job")
    check.add_argument("--output-root", type=Path)
    check.add_argument("--encoder-weights", type=Path)
    prepare = commands.add_parser("prepare", help="Download ImageNet ResNet18 weights once")
    prepare.add_argument("--encoder-weights", required=True, type=Path)
    train = commands.add_parser("train", help="Train a short baseline/refiner run, optionally resume")
    add_data(train)
    add_device(train)
    train.add_argument("--run-dir", required=True, type=Path)
    train.add_argument("--stage", choices=("baseline", "refiner", "both"), default="both")
    train.add_argument("--encoder-weights", type=Path, default=os.environ.get("ENCODER_WEIGHTS"))
    train.add_argument("--baseline", type=Path, help="Complete baseline weights for a refiner-only run")
    train.add_argument("--baseline-updates", type=positive, default=200)
    train.add_argument("--refiner-updates", type=positive, default=200)
    train.add_argument("--crop-size", type=positive, default=256)
    train.add_argument("--batch-size", type=positive, default=2)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--save-every", type=positive, default=50)
    train.add_argument("--seed", type=int, default=523)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--synthetic", action="store_true", help="TEST ONLY: random encoder, clearly marked artifacts")
    predict = commands.add_parser("predict", help="Offline inference on one image or an images-only folder")
    add_device(predict)
    predict.add_argument("--bundle", required=True, type=Path)
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--steps", type=int, choices=range(9), default=4)
    predict.add_argument("--save-trace", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Local ordinary/stress score using the temporary baseline")
    add_data(evaluate)
    add_device(evaluate)
    evaluate.add_argument("--bundle", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--stress-seed", type=int, default=523)
    export = commands.add_parser("export", help="Export complete local weights, source, and instructions")
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    return root


def check_environment(args):
    from .data import discover, read_tile
    versions = {}
    for package in ("torch", "torchvision", "segmentation-models-pytorch", "safetensors", "numpy", "Pillow", "scipy", "scikit-image"):
        versions[package] = importlib.metadata.version(package)
    # Imports exercise binary compatibility (not just installed package names).
    import torchvision
    import segmentation_models_pytorch
    train = discover(args.data_root, "Train", args.train_per_domain)
    validation = discover(args.data_root, "Val", args.val_per_domain)
    valid = {"Train": 0, "Val": 0}
    for split, tiles in (("Train", train), ("Val", validation)):
        for tile in tiles:
            _, mask = read_tile(tile)
            valid[split] += int((mask != 0).sum())
        if not valid[split]:
            raise ValueError(f"Selected {split} masks contain no valid pixels")
    if args.output_root:
        path = args.output_root.resolve()
        while not path.exists():
            path = path.parent
        if not os.access(path, os.W_OK):
            raise ValueError(f"Output root is not writable: {args.output_root}")
    if args.encoder_weights and not args.encoder_weights.is_file():
        raise ValueError(f"Missing encoder weights: {args.encoder_weights}")
    if args.require_cuda:
        device_for("cuda")
        # Test a real GPU operation, not just CUDA discovery.
        (torch.ones(2, device="cuda") + 1).cpu()
    return {"python": sys.version, "versions": versions, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "selected_tiles": {"Train": len(train), "Val": len(validation)}, "valid_pixels": valid}


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    torch.set_num_threads(args.cpu_threads)
    try:
        if args.command == "check":
            import json
            print(json.dumps(check_environment(args), indent=2))
        elif args.command == "prepare":
            from .baseline import prepare_encoder
            prepare_encoder(args.encoder_weights)
            print(f"Saved ImageNet encoder: {args.encoder_weights}")
        elif args.command == "train":
            if args.crop_size < 32 or args.crop_size % 32 or args.workers < 0:
                raise ValueError("crop-size must be a multiple of 32, at least 32; workers must be nonnegative")
            from .training import train
            print(f"Run directory: {train(args, device_for(args.device))}")
        elif args.command == "predict":
            from .inference import predict_folder
            print(predict_folder(args.bundle, args.input, args.output, args.steps,
                                 device_for(args.device), args.save_trace))
        elif args.command == "evaluate":
            from .common import course_module
            from .data import discover
            from .inference import evaluate_tiles, load_bundle
            device = device_for(args.device)
            baseline, refiner, manifest = load_bundle(args.bundle, device)
            tiles = discover(args.data_root, "Val", args.val_per_domain)
            reports = [evaluate_tiles(baseline, refiner, tiles, device, k) for k in (0, 1, 2, 4)]
            stress = evaluate_tiles(baseline, refiner, tiles, device, 4, args.stress_seed)
            quality = [r["overall"]["quality"] for r in reports[1:]] + [stress["overall"]["quality"]]
            score = course_module("metrics").leaderboard_score(*quality)
            write_json(args.output, {"protocol": PROTOCOL, "official_baseline": False,
                                    "baseline_kind": "temporary_resnet18_unet", "synthetic": manifest["synthetic"],
                                    "weights": manifest["weights"], "ordinary": reports, "stress": stress,
                                    "stress_seed": args.stress_seed, "local_score": score})
            print(f"Local temporary-baseline score: {score:.6f}; report: {args.output}")
        elif args.command == "export":
            from .artifacts import export_bundle
            print(export_bundle(args.run_dir, args.output))
    except (ValueError, FileNotFoundError, ImportError) as error:
        command_parser.exit(2, f"Error: {error}\n")
