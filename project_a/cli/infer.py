#!/usr/bin/env python3
"""Leaderboard inference: a folder of RGB images (or one image) -> one folder of road masks per refinement round K.

    python infer.py --input IMAGES --output OUT                    # both stages
    python infer.py --input IMAGES/urban/001.png --output OUT      # a single image (id = file name, "001")
    python infer.py --input IMAGES --output OUT --stage encode     # stage 1 only
    python infer.py --input IMAGES --output OUT --stage refine     # stage 2 only (checks stage 1 is complete)

Stage 1 (encode): the frozen encoder + H0 run on every image and save
    OUT_intermediate/<id>.safetensors   features f [1, C, H/16, W/16], initial logits y0 [1, 1, H, W]
                                        and the raw image [1, 3, H, W] uint8 (TA: the image is a representation)
    OUT_intermediate/index.json         ids and image sizes; written last, so it marks a finished stage 1
Stage 2 (refine): the refiner reads each intermediate file, runs K = 1..max(--steps) rounds and writes
    OUT/k<K>/<id>.png                   uint8, same H x W as the input, 1 = road, 0 = non-road
                                        (k0 = H0 alone, before any refinement)
    OUT/k<K>_prob/<id>.npy              float32 road probability (only with --save-prob)
Image ids are the input path relative to --input, without the extension (sub-folders kept);
for a single image file, the id is its file name without the extension.

Weights: ONE file (default weights/submission.safetensors next to this script) holding the
H0 head ("h0.*") and the refiner ("refiner.*"), with the architecture in its metadata.
The encoder is the public frozen DINOv2 from torch.hub; download it once with internet:
    python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)"

Memory: one image at a time, so RAM use does not grow with the number of images.
Disk: stage 1 stores ~13 MB per 1024x1024 image (fp32 ViT-S/14 features + logits, uint8 image).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from frozen_encoder import FrozenEncoder, H0Head, build_encoder
from submission.config import config_from_json
from submission.image import pack_image, packed_channels
from submission.model import load_refiner

HERE = Path(__file__).resolve().parent
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def list_images(folder):
    """[(path, id)] for a folder of images (searched recursively) or for ONE image file (id = file stem)."""
    folder = Path(folder)
    if folder.is_file():
        if folder.suffix.lower() not in IMAGE_SUFFIXES:
            raise SystemExit(f"{folder} is not an image ({', '.join(IMAGE_SUFFIXES)})")
        return [(folder, folder.stem)]
    files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise SystemExit(f"No images ({', '.join(IMAGE_SUFFIXES)}) under {folder}")
    ids = [f.relative_to(folder).with_suffix("").as_posix() for f in files]
    if len(set(ids)) != len(ids):
        raise SystemExit("Two input files map to the same id (same name, different extension)")
    return list(zip(files, ids))


def load_rgb(path, device):
    """Pre-processing: RGB, float32 in [0, 1], [1, 3, H, W]. Same as training."""
    with Image.open(path) as im:
        array = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1)[None].to(device)


def bundle_metadata(weights):
    with safe_open(str(weights), framework="pt") as fh:
        meta = fh.metadata() or {}
    if meta.get("kind") != "submission":
        raise SystemExit(f"{weights} is not a submission bundle (make one with export_submission.py)")
    return meta


def load_frozen_encoder(weights, device):
    """Frozen encoder + H0 from the bundle -> base(image) = (features, y0)."""
    meta = bundle_metadata(weights)
    encoder = build_encoder(meta["encoder"])
    head = H0Head(encoder.channels, int(meta["h0_hidden"]))
    state = load_file(str(weights))
    head.load_state_dict({k[3:]: v for k, v in state.items() if k.startswith("h0.")}, strict=True)
    base = FrozenEncoder(encoder, [head]).to(device).eval()
    base.requires_grad_(False)
    return base


@torch.no_grad()
def encode(items, weights, work, device):
    """Stage 1: features + y0 for every image -> work/<id>.safetensors, then work/index.json."""
    base = load_frozen_encoder(weights, device)
    index = {"feature_channels": base.feature_channels, "images": {}}
    for i, (path, image_id) in enumerate(items):
        image = load_rgb(path, device)
        features, y0 = base(image)
        if tuple(y0.shape) != (1, 1, *image.shape[-2:]) or not torch.isfinite(y0).all():
            raise ValueError(f"{image_id}: bad initial logits {tuple(y0.shape)}")
        dest = work / f"{image_id}.safetensors"
        dest.parent.mkdir(parents=True, exist_ok=True)
        raw = (image * 255).round().to(torch.uint8)
        save_file({"features": features.float().contiguous().cpu(), "y0": y0.float().contiguous().cpu(),
                   "image": raw.contiguous().cpu()}, str(dest))
        index["images"][image_id] = list(image.shape[-2:])
        print(f"[encode {i + 1}/{len(items)}] {image_id}", flush=True)
    (work / "index.json").write_text(json.dumps(index, indent=1))       # last: marks stage 1 complete


def check_intermediate(work, items):
    """Stage 2 precondition: stage 1 finished and every needed file is present."""
    index_file = work / "index.json"
    assert index_file.is_file(), f"{index_file} missing: run --stage encode first (or it did not finish)"
    index = json.loads(index_file.read_text())
    missing = [i for i in (items or index["images"]) if i not in index["images"]
               or not (work / f"{i}.safetensors").is_file()]
    assert not missing, f"{len(missing)} intermediate files missing, e.g. {missing[:3]}; rerun --stage encode"
    return index


@torch.no_grad()
def refine(index, weights, work, output, steps, device, save_prob=False, ids=None):
    """Stage 2: K rounds of the refiner on each intermediate file -> output/k<K>/<id>.png."""
    cfg = config_from_json(bundle_metadata(weights)["refiner_cfg"])
    channels = index["feature_channels"] + packed_channels(cfg)         # + the image packed into f
    model = load_refiner(weights, channels, device, prefix="refiner.").eval()
    kmax = max(steps)
    ids = ids or sorted(index["images"])
    for n, image_id in enumerate(ids):
        data = load_file(str(work / f"{image_id}.safetensors"), device=str(device))
        features, y = data["features"], data["y0"]
        if cfg.image.use:                   # the refiner sees the image inside the loop (never after it)
            assert "image" in data, f"{image_id}: no raw image in the intermediate file; rerun --stage encode"
            features = pack_image(features, data["image"].float() / 255, cfg.image.stride)
        size = tuple(index["images"][image_id])
        outputs = {0: y}
        if kmax > 0:
            state = model.initialize(features, y)
            for t in range(1, kmax + 1):
                state, y = model.step(features, state, y)
                if t in steps:
                    outputs[t] = y
        for k in steps:
            prob = outputs[k].float().sigmoid()[0, 0].cpu().numpy()
            assert prob.shape == size and np.isfinite(prob).all(), f"{image_id}: bad output at K={k}"
            dest = output / f"k{k}" / f"{image_id}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((prob >= 0.5).astype(np.uint8)).save(dest)             # 1 = road, 0 = not
            if save_prob:
                dest = output / f"k{k}_prob" / f"{image_id}.npy"
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.save(dest, prob.astype(np.float32), allow_pickle=False)
        print(f"[refine {n + 1}/{len(ids)}] {image_id}", flush=True)
    (output / "README.txt").write_text(
        "k<K>/<id>.png: road mask after K refinement rounds (k0 = initial H0 mask, no refinement).\n"
        "uint8 PNG, same height and width as the input image; 1 = road, 0 = non-road.\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="Folder of RGB images (searched recursively), or one image file")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", type=Path, default=HERE / "weights" / "submission.safetensors")
    p.add_argument("--steps", default="0,1,2,4,6,8", help="Comma list of K to save (one run to the max K)")
    p.add_argument("--stage", choices=["all", "encode", "refine"], default="all")
    p.add_argument("--work", type=Path, help="Intermediate folder (default <OUTPUT>_intermediate, next to OUTPUT)")
    p.add_argument("--save-prob", action="store_true", help="Also save float32 probabilities (.npy)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    steps = sorted({int(k) for k in args.steps.split(",")})
    if not steps or min(steps) < 0:
        p.error("--steps needs non-negative integers")
    if args.stage in ("all", "encode") and args.input is None:
        p.error("--input is required for the encode stage")
    if not args.weights.is_file():
        p.error(f"weights not found: {args.weights}")
    device = torch.device(args.device)
    work = args.work or args.output.with_name(args.output.name + "_intermediate")
    work.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    items = list_images(args.input) if args.input else None
    if args.stage in ("all", "encode"):
        encode(items, args.weights, work, device)
    if args.stage in ("all", "refine"):
        ids = [i for _, i in items] if items else None
        index = check_intermediate(work, ids)
        refine(index, args.weights, work, args.output, steps, device, args.save_prob, ids)
        print(f"Saved masks for K = {steps} under {args.output}/k<K>/")


if __name__ == "__main__":
    main()
