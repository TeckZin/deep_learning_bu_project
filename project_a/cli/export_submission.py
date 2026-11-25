#!/usr/bin/env python3
"""Build the leaderboard code submission: a self-contained folder (+ zip) that infer.py runs from.

    python export_submission.py --refiner weights/attempt7/fold0/refiner_best.safetensors \
        --out dist/submission --zip --check path/to/a/few/images

Contents of the folder:
    infer.py, frozen_encoder.py, submission/*.py      inference code only (no training or metrics)
    weights/submission.safetensors                ONE file: H0 head + refiner + architecture metadata
    requirements.txt, README.md
--h0 defaults to the H0 the refiner was validated with (stored in its metadata by train.py).
--check copies the folder into an empty temporary directory and runs infer.py there on
the given images, like a teammate starting from the zip would.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from frozen_encoder import head_metadata
from submission.config import ACTIVE, config_to_json, get_config

HERE = Path(__file__).resolve().parent
CODE = ["infer.py", "frozen_encoder.py"]
# Pinned to the environment the bundle was trained and checked in (BU SCC, Python 3.12.4; torch is the
# CUDA 12.4 build there, torch==2.6.0+cu124 from the PyTorch index; any torch 2.6.0 build runs the code).
REQUIREMENTS = """# Python 3.12
numpy==2.3.5
Pillow==12.3.0
safetensors==0.5.3
torch==2.6.0
"""

README = """# Road segmentation with a recurrent refiner: inference

Python 3.12 with the packages in requirements.txt (`pip install -r requirements.txt`).

The frozen encoder is the public DINOv2 ({encoder}) from torch.hub. It is downloaded on first use;
on a machine without internet, download it once beforehand:

    python -c "import torch; torch.hub.load('facebookresearch/dinov2', '{encoder}', trust_repo=True)"

Run (GPU used automatically if available):

    python infer.py --input IMAGES --output OUT

--input may also be a single image file (its masks are named after the file).

or in two stages (stage 2 checks stage 1 finished):

    python infer.py --input IMAGES --output OUT --stage encode    # encoder + H0 -> OUT_intermediate/
    python infer.py --input IMAGES --output OUT --stage refine    # refiner -> OUT/k<K>/

Output: OUT/k<K>/<id>.png for K in {steps}: the road mask after K refinement rounds
(k0 = initial mask before refinement). uint8 PNG, same height and width as the input,
1 = road, 0 = non-road. Our leaderboard entry is the folder k{main_k}.

Files: infer.py (CLI, pre/post-processing), frozen_encoder.py (frozen encoder + initial head H0),
submission/ (the recurrent refiner), weights/submission.safetensors (H0 + refiner weights).
Refiner architecture: {arch}.
"""


def bundle(h0_path, refiner_path, dest):
    """One safetensors file: h0.* + refiner.* tensors, metadata to rebuild both."""
    h0_meta = head_metadata(h0_path)
    with safe_open(str(refiner_path), framework="pt") as fh:
        r_meta = fh.metadata() or {}
    tensors = {f"h0.{k}": v for k, v in load_file(str(h0_path)).items()}
    tensors.update({f"refiner.{k}": v for k, v in load_file(str(refiner_path)).items()})
    meta = {"kind": "submission", "encoder": h0_meta["encoder"], "h0_hidden": h0_meta.get("hidden", "128"),
            "refiner_cfg": r_meta.get("refiner_cfg") or config_to_json(get_config()),
            "arch": r_meta.get("arch", ACTIVE), "h0_source": Path(h0_path).name,
            "refiner_source": Path(refiner_path).name}
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dest), metadata=meta)
    return meta


def check(folder, images, steps):
    """Run infer.py from a copy of the folder in an empty temp dir; assert every mask folder is complete."""
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "unzipped"
        shutil.copytree(folder, copy)
        out = Path(tmp) / "out"
        subprocess.run([sys.executable, "infer.py", "--input", str(Path(images).resolve()),
                        "--output", str(out), "--steps", steps], cwd=copy, check=True)
        n_in = sum(1 for f in Path(images).rglob("*")
                   if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        for k in steps.split(","):
            n_out = len(list((out / f"k{k}").rglob("*.png")))
            assert n_out == n_in, f"k{k}: {n_out} masks for {n_in} images"
        # a single image file as --input must work too (TA: "an entire folder / a single object")
        one = next(f for f in sorted(Path(images).rglob("*"))
                   if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        subprocess.run([sys.executable, "infer.py", "--input", str(one.resolve()),
                        "--output", str(Path(tmp) / "one"), "--steps", steps], cwd=copy, check=True)
        for k in steps.split(","):
            assert (Path(tmp) / "one" / f"k{k}" / f"{one.stem}.png").is_file(), f"single image: no k{k} mask"
        print(f"check passed: {n_in} images -> masks for K = {steps}; single image {one.name} too")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refiner", type=Path, required=True, help="refiner checkpoint from train.py")
    p.add_argument("--h0", type=Path, help="H0 head (default: the one stored in the refiner's metadata)")
    p.add_argument("--out", type=Path, default=Path("dist/submission"))
    p.add_argument("--steps", default="0,1,2,4,6,8", help="K folders written by default (README)")
    p.add_argument("--main-k", type=int, default=4, help="K named as the leaderboard entry in the README")
    p.add_argument("--zip", action="store_true", help="Also write <out>.zip")
    p.add_argument("--check", type=Path, help="Folder of a few images to test the export on")
    args = p.parse_args(argv)

    h0 = args.h0
    if h0 is None:
        with safe_open(str(args.refiner), framework="pt") as fh:
            h0 = (fh.metadata() or {}).get("eval_h0")
        if not h0 or h0 == "demo":
            p.error("the refiner has no H0 in its metadata; pass --h0")
        h0 = Path(h0)
    if args.out.exists() and any(args.out.iterdir()):
        p.error(f"Refusing nonempty output folder: {args.out}")

    (args.out / "submission").mkdir(parents=True, exist_ok=True)
    for name in CODE:
        shutil.copy2(HERE / name, args.out / name)
    for f in sorted((HERE / "submission").glob("*.py")):
        shutil.copy2(f, args.out / "submission" / f.name)
    meta = bundle(h0, args.refiner, args.out / "weights" / "submission.safetensors")
    (args.out / "requirements.txt").write_text(REQUIREMENTS)
    (args.out / "README.md").write_text(README.format(
        encoder=meta["encoder"], steps=args.steps, main_k=args.main_k,
        arch=f"{meta['arch']} {json.loads(meta['refiner_cfg'])}"))
    print(f"Exported {args.out} (encoder {meta['encoder']}, arch {meta['arch']}, "
          f"H0 {meta['h0_source']}, refiner {meta['refiner_source']})")
    if args.zip:
        archive = shutil.make_archive(str(args.out), "zip", root_dir=args.out)
        print(f"Zipped -> {archive}")
    if args.check:
        check(args.out, args.check, args.steps)


if __name__ == "__main__":
    main()
