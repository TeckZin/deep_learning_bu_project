#!/usr/bin/env python3
"""Public RGB corruption recipe. Labels are never read to choose perturbations.

One corrupted image per source tile; unchanged raw labels. Categories are
balanced to within one image within each domain. Run outside the submission.
Staff keep only private seeds/instances private, not this algorithm or ranges.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from common import fresh_directory, load_manifest, write_json


def rng_for(seed, key):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def corrupt(image, kind, rng):
    array = np.asarray(image, dtype=np.float32) / 255.0
    h, w = array.shape[:2]
    params = {"kind": kind}
    if kind == "blur":
        sigma = float(rng.uniform(1.0, 2.0))
        array = gaussian_filter(array, sigma=(sigma, sigma, 0), mode="reflect", truncate=4.0)
        params["sigma_pixels"] = sigma
    else:
        fraction = float(rng.uniform(*( (0.04, 0.12) if kind == "shadow" else (0.005, 0.02))))
        aspect = float(rng.uniform(0.5, 2.0))
        rh = max(1, min(h, round(np.sqrt(fraction * h * w / aspect))))
        rw = max(1, min(w, round(np.sqrt(fraction * h * w * aspect))))
        y = int(rng.integers(0, h - rh + 1))
        x = int(rng.integers(0, w - rw + 1))
        params.update({"rectangle_xywh": [x, y, rw, rh], "requested_area_fraction": fraction})
        if kind == "shadow":
            alpha = float(rng.uniform(0.3, 0.6))
            array[y:y+rh, x:x+rw] *= alpha
            params["brightness_factor"] = alpha
        elif kind == "occlusion":
            color = np.median(array, axis=(0, 1))
            array[y:y+rh, x:x+rw] = color
            params["fill_rgb01"] = color.tolist()
        else:
            raise ValueError(kind)
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)), params


def generate(manifest, output, seed):
    rows = load_manifest(manifest)
    if any(r["condition"] != "ordinary" for r in rows):
        raise ValueError("Input must be an ordinary manifest")
    root = fresh_directory(output).resolve()
    kinds = ("shadow", "occlusion", "blur")
    assignment = {}
    for domain in ("urban", "rural"):
        group = sorted((r["id"] for r in rows if r["domain"] == domain),
                       key=lambda key: hashlib.sha256(f"{seed}:assignment:{key}".encode()).hexdigest())
        assignment.update({key: kinds[i % 3] for i, key in enumerate(group)})
    new_rows = []
    for row in rows:
        with Image.open(row["image"]) as im:
            result, params = corrupt(im.convert("RGB"), assignment[row["id"]], rng_for(seed, row["id"]))
        path = root / "images" / (row["id"] + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        result.save(path)
        new_rows.append({**row, "condition": "stress", "image": str(path.relative_to(root)),
                         "corruption": params})
    write_json(root / "manifest.json", new_rows)
    return root / "manifest.json"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", required=True, type=int)
    args = p.parse_args()
    print(generate(args.manifest, args.output, args.seed))
