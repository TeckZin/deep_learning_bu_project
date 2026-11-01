"""Offline exact-K inference and course-metric evaluation for the midway model."""
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from project_a.cli.submission.model import build_refiner
from .baseline import TemporaryBaseline, load_weights, pad_image
from .common import PROTOCOL, course_module, fresh_directory, read_json, sha256, source_hash, write_json
from .data import image_tensor, read_tile


def load_bundle(folder, device):
    folder = Path(folder).resolve()
    manifest = read_json(folder / "bundle.json")
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Unsupported bundle protocol")
    if manifest["source_sha256"] != source_hash():
        raise ValueError("Bundle source differs from the installed implementation; install this bundle's code")
    for name in ("baseline.safetensors", "model.safetensors"):
        if sha256(folder / name) != manifest["weights"][name]:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    baseline = load_weights(TemporaryBaseline(), folder / "baseline.safetensors").to(device).eval()
    baseline.requires_grad_(False)
    refiner = load_weights(build_refiner(baseline.feature_channels), folder / "model.safetensors").to(device).eval()
    return baseline, refiner, manifest


def parameter_signature(model):
    return tuple((name, id(p), p.data_ptr(), tuple(p.shape)) for name, p in model.named_parameters())


@torch.no_grad()
def predict_image(baseline, refiner, image, steps, save_trace=False):
    """Pad once, perform exactly K updates, and remove padding from all outputs.

    No autocast is used here: inference has the same precision for all K, so
    separate K=2 and K=4 runs have identical shared prefixes.
    """
    if not 0 <= steps <= 8:
        raise ValueError("steps must be between 0 and 8")
    h, w = image.shape[-2:]
    signature = parameter_signature(refiner)
    calls = [0]
    def count(*_args):
        calls[0] += 1
    hook = refiner.core.register_forward_hook(count)
    trace = []
    try:
        padded = pad_image(image)
        features, logits = baseline(padded)
        expected = logits.shape
        if logits.shape != (image.shape[0], 1, *padded.shape[-2:]):
            raise ValueError("Baseline logits have wrong shape")
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite baseline logits")
        if save_trace:
            trace.append(logits[..., :h, :w].sigmoid().float().cpu().numpy())
        state = refiner.initialize(features, logits) if steps else None
        if calls[0]:
            raise ValueError("initialize must not call the recurrent core")
        for k in range(steps):
            state, logits = refiner.step(features, state, logits)
            if calls[0] != k + 1 or parameter_signature(refiner) != signature:
                raise ValueError("Core count or parameter sharing contract violated")
            if logits.shape != expected or not torch.isfinite(logits).all():
                raise ValueError("Invalid refinement logits")
            if save_trace:
                trace.append(logits[..., :h, :w].sigmoid().float().cpu().numpy())
        return logits[..., :h, :w].float(), trace, calls[0]
    finally:
        hook.remove()


def predict_folder(bundle, source, output, steps, device, save_trace=False):
    baseline, refiner, manifest = load_bundle(bundle, device)
    source = Path(source).resolve()
    extensions = {".png", ".jpg", ".jpeg"}
    if source.is_file():
        files = [source] if source.suffix.lower() in extensions else []
        ids = [source.stem] if files else []
    elif source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
        ids = [p.relative_to(source).with_suffix("").as_posix() for p in files]
    else:
        files, ids = [], []
    if not files:
        raise ValueError("Input must be an RGB PNG/JPEG image or a folder containing images")
    if len(ids) != len(set(ids)):
        raise ValueError("Two images map to the same output ID")
    root = fresh_directory(output)
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for path, image_id in zip(files, ids):
        with Image.open(path) as im:
            image = image_tensor(np.asarray(im.convert("RGB"), dtype=np.uint8))[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits, trace, calls = predict_image(baseline, refiner, image, steps, save_trace)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        probability = logits.sigmoid()[0, 0].cpu().numpy().astype(np.float32)
        save_prediction(root, image_id, probability)
        for k, values in enumerate(trace):
            dest = root / "trace" / f"k{k}" / (image_id + ".npy")
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, values[0, 0], allow_pickle=False)
        rows.append({"id": image_id, "shape": list(probability.shape), "core_calls": calls,
                     "input_sha256": sha256(path), "seconds": seconds})
    write_json(root / "run.json", {
        "protocol": PROTOCOL, "baseline_kind": "temporary_resnet18_unet", "official_baseline": False,
        "synthetic": manifest["synthetic"], "steps": steps, "weights": manifest["weights"],
        "source_sha256": source_hash(), "device": str(device), "images": rows,
        "parameters": sum(p.numel() for p in refiner.parameters()),
        "core_parameters": sum(p.numel() for p in refiner.core.parameters()),
        "baseline_parameters": sum(p.numel() for p in baseline.parameters()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "timing_note": "Diagnostic timing, without warmup; includes baseline and optional trace transfer.",
    })
    return root


def save_prediction(root, image_id, probability):
    dest = Path(root) / "probabilities" / (image_id + ".npy")
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, probability, allow_pickle=False)
    dest = Path(root) / "masks" / (image_id + ".png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((probability >= 0.5).astype(np.uint8) * 255).save(dest)


@torch.no_grad()
def evaluate_tiles(baseline, refiner, tiles, device, steps, stress_seed=None):
    """Pool original course metric counts; never resize validation labels."""
    metrics = course_module("metrics")
    baseline.eval()
    refiner.eval()
    total = metrics.Counts()
    domains = {"urban": metrics.Counts(), "rural": metrics.Counts()}
    corruptions = {}
    loss_sum = 0.0
    kinds = {}
    if stress_seed is not None:
        stress = course_module("make_stress")
        # Match the disclosed staff assignment, balanced within each domain.
        import hashlib
        for domain in domains:
            group = sorted((t for t in tiles if t.domain == domain), key=lambda t: hashlib.sha256(
                f"{stress_seed}:assignment:{t.id}".encode()).hexdigest())
            kinds.update({t.id: ("shadow", "occlusion", "blur")[i % 3] for i, t in enumerate(group)})
    for tile in tiles:
        rgb, labels = read_tile(tile)
        if stress_seed is not None:
            corrupted, _ = stress.corrupt(Image.fromarray(rgb), kinds[tile.id], stress.rng_for(stress_seed, tile.id))
            rgb = np.asarray(corrupted, dtype=np.uint8)
        logits, _, _ = predict_image(baseline, refiner, image_tensor(rgb)[None].to(device), steps)
        probability = logits.sigmoid()[0, 0].cpu().numpy().astype(np.float32)
        counts = metrics.image_counts(probability, labels)
        total += counts
        domains[tile.domain] += counts
        if stress_seed is not None:
            kind = kinds[tile.id]
            corruptions[kind] = corruptions.get(kind, metrics.Counts()) + counts
        target = torch.from_numpy((labels == 3).astype(np.float32)).to(device)[None, None]
        valid = torch.from_numpy(labels != 0).to(device)[None, None]
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        loss_sum += loss[valid].sum().item()
    return {"steps": steps, "condition": "stress" if stress_seed is not None else "ordinary",
            "overall": metrics.summarize(total), "bce": loss_sum / total.valid,
            "by_domain": {d: metrics.summarize(c) for d, c in domains.items() if c.valid},
            "by_corruption": {d: metrics.summarize(c) for d, c in corruptions.items() if c.valid},
            "tiles": [t.record() for t in tiles]}
