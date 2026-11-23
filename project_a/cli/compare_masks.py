#!/usr/bin/env python3
"""Run a DINOv2/H0 + refiner checkpoint and save RGB/prediction/truth panels.

Selects the first N filenames per domain, matching fetch_data_slice.sh under
LC_ALL=C. Use the local downloaded split as --data-root for exact local inputs.
Requires the model source and H0 baseline that belong to the refiner checkpoint.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from common import fresh_directory, sha256, write_json


def select_pairs(root, limit):
    pairs = []
    for domain in ("Urban", "Rural"):
        images = root / domain / "images_png"
        if not images.is_dir():
            raise ValueError(f"Missing image directory: {images}")
        files = sorted(images.glob("*.png"), key=lambda p: p.name)[:limit]
        if not files:
            raise ValueError(f"No PNG images in {images}")
        for image in files:
            mask = root / domain / "masks_png" / image.name
            if not mask.is_file():
                raise FileNotFoundError(mask)
            pairs.append((domain.lower(), image, mask))
    return pairs


def save_comparison(image, probability, labels, output, title):
    if probability.shape != labels.shape or labels.shape != (image.height, image.width):
        raise ValueError("Image, prediction and label dimensions must match")
    if labels.ndim != 2 or not np.isin(labels, np.arange(8)).all():
        raise ValueError("Expected raw LoveDA class IDs 0–7")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("Invalid road probabilities")
    # Binary road segmentation: yellow = road, black = other, gray = ignored.
    panels = [image.convert("RGB")]
    for roads in (probability >= 0.5, labels == 3):
        colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
        colored[roads] = (255, 204, 0)
        colored[labels == 0] = (96, 96, 96)
        panels.append(Image.fromarray(colored))
    w, h = image.size
    canvas = Image.new("RGB", (3 * w, h + 64), "white")
    draw = ImageDraw.Draw(canvas)
    headings = ("Input RGB", "Predicted roads (p >= 0.5)", "Ground truth roads (class 3)")
    for i, (panel, heading) in enumerate(zip(panels, headings)):
        canvas.paste(panel, (i * w, 40))
        draw.text((i * w + 8, 22), heading, fill="black")
    draw.text((8, 4), title, fill="black")
    draw.text((8, h + 44), "Yellow: road | Black: other | Gray: ground-truth ignore (both masks)", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True, type=Path, help="LoveDA Train or Val folder")
    p.add_argument("--checkpoint", type=Path, help="Refiner .safetensors; required for steps > 0")
    p.add_argument("--standin-baseline", required=True, type=Path, help="Matching H0 .safetensors")
    p.add_argument("--output", required=True, type=Path, help="New or empty output directory")
    p.add_argument("--first-per-domain", type=int, default=50)
    p.add_argument("--steps", type=int, choices=range(9), default=4)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = p.parse_args()
    if args.first_per_domain < 1:
        p.error("--first-per-domain must be positive")
    if args.steps and args.checkpoint is None:
        p.error("--checkpoint is required for steps > 0")
    pairs = select_pairs(args.data_root, args.first_per_domain)

    import torch
    from safetensors.torch import load_file
    from standin_baseline import load_standin
    from submission.model import build_refiner

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable; use a GPU allocation or --device cpu")
    base = load_standin(args.standin_baseline, device)
    model = None
    if args.steps:
        model = build_refiner(base.feature_channels).to(device).eval()
        model.load_state_dict(load_file(str(args.checkpoint), device=args.device), strict=True)
    output = fresh_directory(args.output)
    records = []
    with torch.inference_mode():
        for index, (domain, image_path, label_path) in enumerate(pairs, 1):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            with Image.open(label_path) as source:
                labels = np.array(source)
            array = np.array(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(device)
            features, logits = base(tensor)
            if model is not None:
                state = model.initialize(features, logits)
                for _ in range(args.steps):
                    state, logits = model.step(features, state, logits)
            probability = logits.sigmoid()[0, 0].float().cpu().numpy()
            image_id = f"{domain}/{image_path.stem}"
            save_comparison(image, probability, labels,
                            output / "comparisons" / f"{image_id}.png",
                            f"{image_id} | K={args.steps}")
            mask_path = output / "masks" / f"{image_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((probability >= 0.5).astype(np.uint8) * 255).save(mask_path)
            records.append({"id": image_id, "image": str(image_path.resolve()),
                            "label": str(label_path.resolve()), "input_sha256": sha256(image_path)})
            print(f"[{index}/{len(pairs)}] {image_id}", flush=True)
    write_json(output / "comparison_run.json", {
        "steps": args.steps, "device": args.device,
        "checkpoint": str(args.checkpoint.resolve()) if model is not None else None,
        "checkpoint_sha256": sha256(args.checkpoint) if model is not None else None,
        "baseline": str(args.standin_baseline.resolve()),
        "baseline_sha256": sha256(args.standin_baseline),
        "selection": "first filenames per domain, lexicographic order",
        "first_per_domain": args.first_per_domain, "images": records,
    })
    print(f"Saved {len(records)} comparisons and predicted masks to {output}")


if __name__ == "__main__":
    main()
