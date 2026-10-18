#!/usr/bin/env python3
"""Staff-owned exact-K runner. Run `python predict.py --help`.

Hooks and metadata are debugging/audit aids, NOT a security boundary. Execute
untrusted student code in a separate offline container with resource limits.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

from baseline import load_baseline
from common import VERSION, check_probability, fresh_directory, sha256, write_json
from submission.model import build_refiner


def parameter_signature(model):
    return tuple((name, id(p), p.data_ptr(), tuple(p.shape))
                 for name, p in model.named_parameters())


def source_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted((root / "submission").rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def predict_tensor(base, model, image, steps, ablation="none"):
    """Returns probabilities for K=0..steps; single image/batch.

    This pure helper also underpins tests. The runner never passes total K to
    the model. One step must call the same registered core exactly once.
    """
    calls = [0]
    def count(_module, _args, _output):
        calls[0] += 1
    handle = model.core.register_forward_hook(count)
    signature = parameter_signature(model)
    try:
        with torch.inference_mode():
            features, initial_logits = base(image)
            expected = (image.shape[0], 1, *image.shape[-2:])
            if tuple(initial_logits.shape) != expected:
                raise ValueError("Baseline logits must match original resolution")
            if not torch.isfinite(initial_logits).all():
                raise ValueError("Nonfinite initial logits")
            current = initial_logits.clone()
            trace = [current.sigmoid().detach().cpu().numpy()]
            state = model.initialize(features.clone(), initial_logits.clone()) if steps else None
            if calls[0] != 0:
                raise ValueError("initialize() cannot call the recurrent core")
            for t in range(steps):
                if ablation == "reset-state":
                    state = model.initialize(features.clone(), initial_logits.clone())
                feedback = initial_logits.clone() if ablation == "no-mask-feedback" else current.clone()
                state, current = model.step(features.clone(), state, feedback)
                if calls[0] != t + 1:
                    raise ValueError("Each step must call model.core exactly once")
                if parameter_signature(model) != signature:
                    raise ValueError("Parameters were replaced or added during recurrence")
                if tuple(current.shape) != expected or not torch.isfinite(current).all():
                    raise ValueError("Step logits have wrong shape or nonfinite values")
                trace.append(current.sigmoid().detach().cpu().numpy())
        return trace, calls[0]
    finally:
        handle.remove()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--input", type=Path, required=True, help="RGB image directory (recursive)")
    p.add_argument("--output", type=Path, required=True, help="New or empty directory")
    p.add_argument("--steps", type=int, required=True, choices=range(9), metavar="K")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--save-trace", action="store_true")
    p.add_argument("--demo", action="store_true", help="UNTRAINED smoke test only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ablation", choices=("none", "reset-state", "no-mask-feedback"), default="none")
    args = p.parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    files = sorted(f for f in args.input.rglob("*")
                   if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        raise ValueError("No PNG/JPEG images in --input")
    ids = [f.relative_to(args.input).with_suffix("").as_posix() for f in files]
    if len(set(ids)) != len(ids):
        raise ValueError("Two input files map to the same output id")
    device = torch.device(args.device)
    base = load_baseline(args.baseline, device, demo=args.demo)
    model = build_refiner(base.feature_channels).to(device)
    model.load_state_dict(load_file(str(args.checkpoint), device=args.device), strict=True)
    model.eval()
    if not isinstance(model.core, torch.nn.Module):
        raise TypeError("Expose the shared recurrent core as model.core")
    output = fresh_directory(args.output)
    rows = []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for path, image_id in zip(files, ids):
        with Image.open(path) as im:
            array = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)[None].to(device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        trace, calls = predict_tensor(base, model, tensor, args.steps, args.ablation)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        prob = check_probability(trace[-1][0, 0].astype(np.float32), array.shape[:2])
        dest = output / "probabilities" / (image_id + ".npy")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, prob, allow_pickle=False)
        mask_path = output / "masks" / (image_id + ".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((prob >= 0.5).astype(np.uint8) * 255).save(mask_path)
        if args.save_trace:
            for k, prediction in enumerate(trace):
                dest = output / "trace" / f"k{k}" / (image_id + ".npy")
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(dest, prediction[0, 0].astype(np.float32), allow_pickle=False)
        rows.append({"id": image_id, "shape": list(array.shape[:2]),
                     "input_sha256": sha256(path), "core_calls": calls,
                     "seconds_with_trace_transfer": elapsed})
    write_json(output / "run.json", {
        "protocol": VERSION, "steps": args.steps, "demo": args.demo,
        "ablation": args.ablation, "seed": args.seed, "device": args.device,
        "checkpoint_sha256": sha256(args.checkpoint),
        "baseline_sha256": "demo-brightness-v1" if args.demo else sha256(args.baseline),
        "submission_source_sha256": source_hash(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "core_parameters": sum(p.numel() for p in model.core.parameters()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "timing_note": "Includes CPU trace transfer; no warmup. Not official latency benchmark.",
        "images": rows,
    })
    print(f"Saved {len(rows)} images after exactly {args.steps} steps to {output}")


if __name__ == "__main__":
    main()
