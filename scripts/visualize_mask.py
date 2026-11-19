#!/usr/bin/env python3
"""Colorize a raw LoveDA mask (IDs 0–7) and save a PNG with a class legend.

Dependencies: numpy, Pillow, matplotlib.
Example, from the repository root:
    python scripts/visualize_mask.py ../00.png --image ../0.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures without requiring a display.
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


# Raw IDs, before the training loader remaps them:
# https://github.com/Junjue-Wang/LoveDA#readme
CLASSES = (
    ("No data / ignore", "#202020"),
    ("Background", "#bdbdbd"),
    ("Building", "#e53935"),
    ("Road", "#ffcc00"),
    ("Water", "#1687d9"),
    ("Barren", "#b57a45"),
    ("Forest", "#176b3a"),
    ("Agriculture", "#a6d96a"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask", type=Path, help="Raw, single-channel LoveDA mask")
    parser.add_argument("--image", type=Path, help="Optional matching RGB image")
    parser.add_argument("-o", "--output", type=Path,
                        help="Output PNG (default: outputs/<mask stem>_semantic.png)")
    args = parser.parse_args()
    output = args.output or Path("outputs") / f"{args.mask.stem}_semantic.png"
    if output.resolve() in {p.resolve() for p in (args.mask, args.image) if p}:
        parser.error("Output must differ from the input files.")

    with Image.open(args.mask) as source:
        mask = np.array(source)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        parser.error("Expected a single-channel integer class-ID mask.")
    present = np.unique(mask)
    unknown = present[(present < 0) | (present >= len(CLASSES))]
    if unknown.size:
        parser.error(f"Expected raw LoveDA IDs 0–7; found unexpected IDs {unknown.tolist()}.")

    image = None
    if args.image:
        with Image.open(args.image) as source:
            image = np.array(source.convert("RGB"))
        if image.shape[:2] != mask.shape:
            parser.error(f"Image shape {image.shape[:2]} differs from mask shape {mask.shape}.")

    fig, axes = plt.subplots(1, 2 if image is not None else 1,
                             figsize=(14 if image is not None else 8, 6),
                             squeeze=False)
    if image is not None:
        axes[0, 0].imshow(image)
        axes[0, 0].set_title(f"RGB image — {args.image.name}")
    cmap = ListedColormap([color for _, color in CLASSES])
    norm = BoundaryNorm(np.arange(len(CLASSES) + 1) - 0.5, cmap.N)
    axes[0, -1].imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
    axes[0, -1].set_title(f"Semantic classes — {args.mask.name}")
    for ax in axes.flat:
        ax.axis("off")
    handles = [Patch(facecolor=color, edgecolor="#555555",
                     label=f"{idx}: {name}" + (" (absent)" if idx not in present else ""))
               for idx, (name, color) in enumerate(CLASSES)]
    axes[0, -1].legend(handles=handles, title="LoveDA class IDs",
                       loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {output.resolve()} (class IDs present: {present.tolist()})")


if __name__ == "__main__":
    main()
