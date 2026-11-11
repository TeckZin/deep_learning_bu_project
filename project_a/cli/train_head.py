import argparse
from pathlib import Path
import random

import torch
from safetensors.torch import save_file

from common import load_manifest
from dataset import load_image, load_mask
from encoder import build_encoder
from encoder_baseline import EncoderBaseline
from train import seg_loss, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        choices=("resnet34", "resnet50", "dino_vits16"),
        required=True,
    )
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_rows = load_manifest(args.pilot_dir / "train" / "manifest.json")
    val_rows = load_manifest(args.pilot_dir / "val" / "manifest.json")

    encoder = build_encoder(args.encoder)
    torch.manual_seed(0)
    baseline = EncoderBaseline(encoder).to(device)
    optimizer = torch.optim.AdamW(baseline.head.parameters(), lr=1e-3)
    rng = random.Random(0)

    for epoch in range(args.epochs):
        baseline.train()
        order = train_rows.copy()
        rng.shuffle(order)
        total_loss = 0.0
        updates = 0

        for row in order:
            target, valid = load_mask(row["label"])
            if not valid.any():
                continue

            images = load_image(row["image"]).unsqueeze(0).to(device)
            targets = target.unsqueeze(0).to(device)
            valid_pixels = valid.unsqueeze(0).to(device)

            optimizer.zero_grad(set_to_none=True)
            _, logits = baseline(images)
            loss = seg_loss(logits, targets, valid_pixels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            updates += 1

        if updates == 0:
            raise RuntimeError("No training tiles contained valid pixels")

        print(
            f"Epoch {epoch + 1}: {updates} updates, "
            f"mean loss {total_loss / updates:.4f}",
            flush=True,
        )

    output = args.pilot_dir / f"{args.encoder}_head.safetensors"
    head_weights = {
        name: value.detach().cpu().contiguous()
        for name, value in baseline.head.state_dict().items()
    }
    save_file(
        head_weights,
        str(output),
        metadata={"kind": "roadseg_encoder_head", "encoder": args.encoder},
    )
    print(f"Saved {output}", flush=True)

    # At K=0 the validator scores only the baseline's initial prediction.
    scores = validate(baseline, baseline, val_rows, device, ks=(0,))[0]
    for metric in ("road_iou", "cldice", "boundary_iou", "quality"):
        print(f"K=0 {metric}: {scores[metric]:.4f}")


if __name__ == "__main__":
    main()