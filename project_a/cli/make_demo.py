#!/usr/bin/env python3
"""Create random weights and SYNTHETIC images; not course data or a trained model."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file
from common import fresh_directory, write_json
from submission.model import build_refiner


def generate(output):
    root = fresh_directory(output)
    torch.manual_seed(523)
    model = build_refiner(5)
    save_file(model.state_dict(), str(root / "model.safetensors"), metadata={"demo": "true", "trained": "false"})
    (root / "images").mkdir()
    (root / "labels").mkdir()
    rows = []
    for i, domain in enumerate(["urban", "rural"]):
        image_id = f"synthetic_{i}"
        mask = np.zeros((96, 96), dtype=bool)
        mask[42:52, 8:88] = True
        mask[8:88, 42+i*8:52+i*8] = True
        rgb = np.full((96, 96, 3), 35, dtype=np.uint8)
        rgb[mask] = 210
        labels = np.where(mask, 3, 1).astype(np.uint8)
        Image.fromarray(rgb).save(root / "images" / f"{image_id}.png")
        Image.fromarray(labels).save(root / "labels" / f"{image_id}.png")
        rows.append({"id": image_id, "image": f"images/{image_id}.png",
                     "label": f"labels/{image_id}.png", "domain": domain, "condition": "ordinary"})
    write_json(root / "manifest.json", rows)
    write_json(root / "DEMO_ONLY.json", {"trained": False, "dataset": "synthetic", "eligible": False})
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(generate(p.parse_args().output))
