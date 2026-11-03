#!/usr/bin/env python3
"""Build a local evaluation set + manifest from a LoveDA split (e.g. dataset/Val).

Creates:
    <output>/images/urban/<id>.png   symlinks to the real images (no copying)
    <output>/images/rural/<id>.png
    <output>/manifest.json           rows in the format common.load_manifest expects

Image ids are "<domain>/<stem>", which is exactly what predict.py produces when
run with --input <output>/images. For local scoring only; the official
manifests come from course staff.
"""
import argparse
import json
from pathlib import Path
import random

DOMAINS = {"Urban": "urban", "Rural": "rural"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True, help="LoveDA split folder, e.g. .../dataset/Val")
    p.add_argument("--output", type=Path, required=True, help="New folder for images/ and manifest.json")
    p.add_argument("--max-per-domain", type=int, help="Random subset size per domain (for quick runs)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing nonempty output folder: {args.output}")
    rng = random.Random(args.seed)
    rows = []
    for folder, domain in DOMAINS.items():
        images = args.data_root / folder / "images_png"
        masks = args.data_root / folder / "masks_png"
        if not images.is_dir():
            print(f"skip {domain}: {images} not found")
            continue
        stems = sorted(f.stem for f in images.glob("*.png") if (masks / f.name).is_file())
        if args.max_per_domain is not None and len(stems) > args.max_per_domain:
            stems = sorted(rng.sample(stems, args.max_per_domain))
        link_dir = args.output / "images" / domain
        link_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            link = link_dir / f"{stem}.png"
            link.symlink_to((images / f"{stem}.png").resolve())
            rows.append({
                "id": f"{domain}/{stem}",
                "domain": domain,
                "condition": "ordinary",
                "image": str(link.relative_to(args.output)),
                "label": str((masks / f"{stem}.png").resolve()),
            })
        print(f"{domain}: {len(stems)} images")

    if not rows:
        raise SystemExit("No images found")
    (args.output / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
