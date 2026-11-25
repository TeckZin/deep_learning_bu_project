"""Example panels for a trained refiner: inputs, ground truth and predictions at every K.

Scores every validation image of a fold at K=0,1,2,4 (and K=4 with z reset before every step), then
draws panels for the images in each category:

  worst      lowest quality at K=4
  best       highest quality at K=4
  improves   largest gain K=0 -> K=4
  no_gain    smallest gain K=0 -> K=4 (gets worse over K)
  late       largest gain K=1 -> K=4 (steps 2-4 matter)
  uses_z     most K=4 pixels flip when z is reset before every step

Per-image quality is noisy when an image has almost no road (an empty truth and an empty prediction
score 1.0), so the rankings skip images with less than --min-road road pixels.

Panel (one PNG per image), three rows of four tiles:
  image | truth | K=0 | K=1
  K=2 | K=4 | K=4 z-reset | |p4 - p4 z-reset|
  road probability at K=0 | K=1 | K=2 | K=4      (viridis, 0 = dark purple, 1 = yellow; scale bar below)
Truth uses the LoveDA class colors of scripts/visualize_mask.py (road = yellow).
Prediction colors: white = road hit, red = false road, blue = missed road, black = background,
gray = no-data (ignored by the metrics).

    python examples.py --weights weights/fdzeolla-attempt10/fold0/refiner_best.safetensors \
        --h0 weights/h0/dinov2_vits14_oof/fold0/h0_all.safetensors --val-fold 0 --out examples/attempt10_f0
    python examples.py ... --max-images 20          # quick check
    python examples.py ... --from-csv examples/attempt10_f0/scores.csv   # redraw without rescoring

Outputs in --out: scores.csv (one row per image), <category>/<rank>_<id>.png, chosen.json.

Sharing (--share DIR): a small copy for the repo, which GitHub shows as a gallery. DIR gets README.md
(summary, one section per category with a score table and the panels inline) and
<category>/<rank>_<id>.jpg (JPEG, about 100-200 KB instead of 500 KB). Without --weights it only builds the gallery from the panels already in --out:

    python examples.py --out examples/attempt10_f0 --share ../../docs/experiments/attempt10_examples \
        --title "attempt10, fold 0"
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file
import torch

from data import load_image
from evaluation import _labels
from frozen_encoder import load_frozen_encoder
from metrics import image_counts, summarize
from splits import DEFAULT_FOLDS, default_data_root, read_folds
from submission.config import config_from_metadata, get_config
from submission.image import ImagePackedBase
from submission.model import build_refiner

KS = (0, 1, 2, 4)
CATEGORIES = ("worst", "best", "improves", "no_gain", "late", "uses_z")
DESCRIPTIONS = {
    "worst": "Lowest quality at K=4.",
    "best": "Highest quality at K=4.",
    "improves": "Largest gain from K=0 (H0) to K=4.",
    "no_gain": "Smallest gain from K=0 to K=4 (the refiner makes these worse).",
    "late": "Largest gain from K=1 to K=4, i.e. where steps 2-4 matter most.",
    "uses_z": "Most K=4 pixels flip when the memory z is reset before every step.",
}
TILE = 384          # panel tile size in pixels
SHARE_QUALITY = 88  # JPEG quality of the shared panels: about 100 KB each, captions stay sharp


@torch.no_grad()
def run_image(model, base, image, steps=4):
    """{name: logits [H, W]} for K=0..steps and K=steps with z reset before every step."""
    features, y0 = base(image)
    out = {"k0": y0}
    state, y = model.initialize(features, y0), y0
    for t in range(1, steps + 1):
        state, y = model.step(features, state, y)
        out[f"k{t}"] = y
    state, y = model.initialize(features, y0), y0
    for _ in range(steps):
        state, y = model.step(features, model.initialize(features, y0), y)
    out["reset"] = y
    return {k: v.float()[0, 0].cpu() for k, v in out.items()}


def score(outputs, labels):
    """Per-image metrics for the outputs we rank on."""
    row = {}
    for name in [f"k{k}" for k in KS] + ["reset"]:
        s = summarize(image_counts(outputs[name].sigmoid().numpy(), labels))
        row[f"q_{name}"] = 100 * s["quality"]
        if name in ("k0", "k4"):
            row[f"iou_{name}"] = 100 * s["road_iou"]
            row[f"cldice_{name}"] = 100 * s["cldice"]
            row[f"biou_{name}"] = 100 * s["boundary_iou"]
    valid = labels != 0
    p4, pr = outputs["k4"].numpy() > 0, outputs["reset"].numpy() > 0
    row["z_flip"] = 100 * float(((p4 != pr) & valid).sum() / max(valid.sum(), 1))
    row["road_frac"] = float(((labels == 3) & valid).sum() / max(valid.sum(), 1))
    return row


def choose(rows, n, min_road):
    """{category: rows} from the per-image scores."""
    ranked = [r for r in rows if r["road_frac"] >= min_road]
    def top(key, reverse):
        return sorted(ranked, key=key, reverse=reverse)[:n]
    return {
        "worst": top(lambda r: r["q_k4"], False),
        "best": top(lambda r: r["q_k4"], True),
        "improves": top(lambda r: r["q_k4"] - r["q_k0"], True),
        "no_gain": top(lambda r: r["q_k4"] - r["q_k0"], False),
        "late": top(lambda r: r["q_k4"] - r["q_k1"], True),
        "uses_z": top(lambda r: r["z_flip"], True),
    }


# ---- drawing ----

def _tile(array):
    return Image.fromarray(array).resize((TILE, TILE), Image.NEAREST)


# Raw LoveDA class colors, the same as scripts/visualize_mask.py (0 no-data, 1 background, 2 building,
# 3 road, 4 water, 5 barren, 6 forest, 7 agriculture).
LOVEDA = np.array([(0x20, 0x20, 0x20), (0xbd, 0xbd, 0xbd), (0xe5, 0x39, 0x35), (0xff, 0xcc, 0x00),
                   (0x16, 0x87, 0xd9), (0xb5, 0x7a, 0x45), (0x17, 0x6b, 0x3a), (0xa6, 0xd9, 0x6a)], np.uint8)


def truth_tile(labels):
    """All LoveDA classes (road is yellow), so errors can be read against what is really there."""
    return _tile(LOVEDA[np.clip(labels, 0, 7)])


def error_tile(logits, labels):
    pred, truth, valid = logits.numpy() > 0, labels == 3, labels != 0
    rgb = np.zeros(labels.shape + (3,), np.uint8)
    rgb[pred & truth] = (255, 255, 255)
    rgb[pred & ~truth] = (230, 50, 50)
    rgb[~pred & truth] = (40, 120, 255)
    rgb[~valid] = 90
    return _tile(rgb)


# Viridis anchor colors (dark purple = 0, yellow = 1): readable at a glance and in grayscale.
VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], np.float32)


def colormap(values):
    """values in [0, 1] -> RGB uint8 by linear interpolation between the VIRIDIS anchors."""
    x = np.clip(values, 0, 1) * (len(VIRIDIS) - 1)
    i = np.minimum(x.astype(int), len(VIRIDIS) - 2)
    t = (x - i)[..., None]
    return ((1 - t) * VIRIDIS[i] + t * VIRIDIS[i + 1]).astype(np.uint8)


def prob_tile(logits, labels):
    """Road probability sigmoid(y) on the same 0-1 scale for every K; no-data pixels gray."""
    rgb = colormap(logits.sigmoid().numpy())
    rgb[labels == 0] = 90
    return _tile(rgb)


def colorbar(width, height=12):
    bar = colormap(np.tile(np.linspace(0, 1, width), (height, 1)))
    return Image.fromarray(bar)


def diff_tile(a, b):
    d = (a.sigmoid() - b.sigmoid()).abs().numpy() ** 0.5          # sqrt: small differences stay visible
    heat = np.stack([d, d * 0.6, np.zeros_like(d)], -1)          # black -> orange
    return _tile((255 * heat.clip(0, 1)).astype(np.uint8))


def panel(row, image, labels, outputs):
    img = _tile((255 * image[0].permute(1, 2, 0).cpu().numpy()).clip(0, 255).astype(np.uint8))
    tiles = [(img, "image"), (truth_tile(labels), "truth"),
             *[(error_tile(outputs[f"k{k}"], labels), f"K={k}  q {row[f'q_k{k}']:.1f}") for k in KS],
             (error_tile(outputs["reset"], labels), f"K=4 z-reset  q {row['q_reset']:.1f}"),
             (diff_tile(outputs["k4"], outputs["reset"]), f"|p4 - p4 z-reset|  flip {row['z_flip']:.1f}%"),
             *[(prob_tile(outputs[f"k{k}"], labels), f"road probability, K={k}") for k in KS]]
    cols, head, cap, foot = 4, 34, 22, 40
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * TILE, head + rows * (TILE + cap) + foot), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), f"{row['id']}   K=4 IoU {row['iou_k4']:.1f}  clDice {row['cldice_k4']:.1f}  "
                      f"BIoU {row['biou_k4']:.1f}   road {100 * row['road_frac']:.1f}%   "
                      "white hit / red false road / blue missed / gray no-data", fill=(0, 0, 0))
    for i, (tile, caption) in enumerate(tiles):
        x, y = (i % cols) * TILE, head + (i // cols) * (TILE + cap)
        draw.text((x + 6, y + 4), caption, fill=(0, 0, 0))
        out.paste(tile, (x, y + cap))
    # probability scale for the bottom row; 0.5 is the road / not-road threshold
    y0, bar_w = head + rows * (TILE + cap) + 8, 2 * TILE
    out.paste(colorbar(bar_w), (TILE, y0))
    draw.text((TILE - 64, y0), "probability 0", fill=(0, 0, 0))
    draw.text((TILE + bar_w + 6, y0), "1", fill=(0, 0, 0))
    draw.line([(TILE + bar_w // 2, y0 - 3), (TILE + bar_w // 2, y0 + 15)], fill=(0, 0, 0))
    draw.text((TILE + bar_w // 2 - 40, y0 + 17), "0.5 = threshold", fill=(0, 0, 0))
    return out


# ---- sharing ----

def _panel_name(rank, image_id):
    return f"{rank:02d}_{image_id.replace('/', '_')}"


def breakdown(scored, min_road=0.01):
    """Markdown lines: where the refiner helps and hurts, H0 (K=0) -> K=4, by domain, road density and H0 quality.

    Per-image means. The domain comes from the id prefix (urban/..., rural/...).
    """
    def mean(rs, key):
        return sum(r[key] for r in rs) / len(rs)

    def line(name, rs):
        if not rs:
            return None
        q0, q4 = mean(rs, "q_k0"), mean(rs, "q_k4")
        worse = 100 * sum(r["q_k4"] < r["q_k0"] - 1 for r in rs) / len(rs)
        deltas = [mean(rs, f"{m}_k4") - mean(rs, f"{m}_k0") for m in ("iou", "cldice", "biou")]
        return (f"| {name} | {len(rs)} | {q0:.1f} -> {q4:.1f} (**{q4 - q0:+.1f}**) | "
                + " | ".join(f"{d:+.1f}" for d in deltas) + f" | {worse:.0f}% |")

    road = [r for r in scored if r["road_frac"] >= min_road]
    domain = lambda r: r["id"].split("/", 1)[0]
    groups = []
    for d in sorted({domain(r) for r in road}):
        groups.append((d.capitalize(), [r for r in road if domain(r) == d]))
    for d in sorted({domain(r) for r in road}):
        for lo, hi, name in ((min_road, 0.03, "thin roads"), (0.03, 0.08, "medium roads"),
                             (0.08, 1.01, "dense roads")):
            groups.append((f"{d.capitalize()}, {name} ({100 * lo:.0f}-{min(100 * hi, 100):.0f}% road)",
                           [r for r in road if domain(r) == d and lo <= r["road_frac"] < hi]))
    for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 101)):
        groups.append((f"H0 quality {lo}-{min(hi, 100)}", [r for r in road if lo <= r["q_k0"] < hi]))
    groups.append((f"Almost no road (< {100 * min_road:.0f}%)", [r for r in scored if r["road_frac"] < min_road]))

    lines = ["## Where it helps and where it hurts", "",
             f"Change from H0 (K=0) to K=4, as means of per-image scores. Images with under {100 * min_road:.0f}% road "
             "are left out of every row but the last; there an empty truth and an empty prediction score 100, so "
             "that row mostly measures removed false road. The last column is the share of images that get worse "
             "by more than 1 point.", "",
             "| Group | Images | Quality H0 -> K=4 | IoU | clDice | BIoU | Worse by >1 |",
             "|---|---|---|---|---|---|---|"]
    lines += [l for l in (line(name, rs) for name, rs in groups) if l]
    return lines + [""]


def share(out, dest, title=None, min_road=0.01):
    """Gallery for teammates from the panels, scores.csv and chosen.json in `out` (see module doc)."""
    with (out / "scores.csv").open() as fh:
        rows = {r["id"]: {k: (v if k == "id" else float(v)) for k, v in r.items()}
                for r in csv.DictReader(fh)}
    chosen = json.loads((out / "chosen.json").read_text())
    dest.mkdir(parents=True, exist_ok=True)

    scored = list(rows.values())
    mean = {k: sum(r[k] for r in scored) / len(scored) for k in ("q_k0", "q_k1", "q_k2", "q_k4", "q_reset")}
    lines = [f"# Example panels: {title or out.name}", "",
             f"Made by `project_a/cli/examples.py --share` ({len(scored)} validation images scored; "
             "the full-size PNGs and scores.csv stay with the run on SCC). Quality = 0.5 IoU + 0.3 clDice + 0.2 Boundary IoU, per image, in percent.", "",
             "| Mean of per-image quality | K=0 (H0) | K=1 | K=2 | K=4 | K=4, z reset |", "|---|---|---|---|---|---|",
             "| All scored images | " + " | ".join(f"{mean[k]:.1f}" for k in mean) + " |", "",
             "This averages each image's own score, so near-empty images count as much as busy ones. The "
             "fold metric in the results docs counts pixels over all images together and is different.", "",
             "How to read a panel (three rows of four tiles):", "",
             "- Row 1: image, truth (LoveDA classes, road = yellow), prediction at K=0 and K=1.",
             "- Row 2: prediction at K=2, K=4, K=4 with z reset before every step, and how much the road "
             "probability changes when z is reset (black = no change, orange = large).",
             "- Row 3: road probability at K=0, 1, 2, 4 (dark purple = 0, yellow = 1).",
             "- Prediction colors: white = road hit, red = false road, blue = missed road, black = background, "
             "gray = no-data.", "",
             "Categories: " + ", ".join(f"[{c}](#{c})" for c in chosen), ""]
    lines += breakdown(scored, min_road)

    for cat, ids in chosen.items():
        folder = dest / cat
        folder.mkdir(exist_ok=True)
        for rank, image_id in enumerate(ids, 1):
            name = _panel_name(rank, image_id)
            im = Image.open(out / cat / f"{name}.png").convert("RGB")
            im.save(folder / f"{name}.jpg", quality=SHARE_QUALITY, optimize=True)

        lines += [f"## {cat}", "", DESCRIPTIONS.get(cat, ""), "",
                  "| # | Image | K=0 | K=1 | K=2 | K=4 | K=4 z reset | K=4 IoU | K=4 clDice | K=4 BIoU "
                  "| z flip % | Road % |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for rank, image_id in enumerate(ids, 1):
            r = rows[image_id]
            cells = [r["q_k0"], r["q_k1"], r["q_k2"], r["q_k4"], r["q_reset"],
                     r["iou_k4"], r["cldice_k4"], r["biou_k4"], r["z_flip"], 100 * r["road_frac"]]
            link = f"[{image_id}]({cat}/{_panel_name(rank, image_id)}.jpg)"
            lines.append(f"| {rank} | {link} | " + " | ".join(f"{c:.1f}" for c in cells) + " |")
        lines.append("")
        for rank, image_id in enumerate(ids, 1):
            name = _panel_name(rank, image_id)
            lines += [f"**{rank}. {image_id}**", "", f"![{image_id}]({cat}/{name}.jpg)", ""]

    (dest / "README.md").write_text("\n".join(lines))
    print(f"shared gallery: {dest}/README.md")


# ---- main ----

def load_model(weights, h0, device):
    with safe_open(str(weights), "pt") as fh:
        meta = fh.metadata() or {}
    cfg = config_from_metadata(meta) or get_config(meta.get("arch"))
    base = load_frozen_encoder(h0, device)
    if cfg.image.use:
        base = ImagePackedBase(base, cfg.image.stride)
    model = build_refiner(base.feature_channels, cfg).to(device).eval()
    model.load_state_dict(load_file(str(weights)))
    model.pos_offset = None
    return model, base.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, help="refiner .safetensors from train.py (omit to only --share)")
    p.add_argument("--h0", type=Path, help="H0 head used for validation (run.json eval_h0)")
    p.add_argument("--val-fold", type=int, default=0)
    p.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    p.add_argument("--data-root", type=Path, default=default_data_root())
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=6, help="images per category")
    p.add_argument("--min-road", type=float, default=0.01, help="skip images with less road (fraction)")
    p.add_argument("--max-images", type=int, help="score only the first N validation images")
    p.add_argument("--from-csv", type=Path, help="reuse scores.csv instead of scoring again")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--share", type=Path, help="also write a small gallery for the repo here (README.md + JPEGs)")
    p.add_argument("--title", help="gallery title, e.g. 'attempt10, fold 0' (default: the --out folder name)")
    args = p.parse_args()
    if args.weights is None:
        if args.share is None:
            p.error("--weights and --h0 are required unless you only build a gallery with --share")
        share(args.out, args.share, args.title, args.min_road)
        return
    if args.h0 is None:
        p.error("--h0 is required with --weights")
    if args.data_root is None:
        p.error("--data-root or $ROADSEG_DATA is required")

    device = torch.device(args.device)
    model, base = load_model(args.weights, args.h0, device)
    val = [r for r in read_folds(args.folds, args.data_root) if r["fold"] == args.val_fold]
    val = val[:args.max_images] if args.max_images else val
    by_id = {r["id"]: r for r in val}
    args.out.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        with args.from_csv.open() as fh:
            rows = [{k: (v if k == "id" else float(v)) for k, v in r.items()} for r in csv.DictReader(fh)]
    else:
        rows = []
        for i, r in enumerate(val):
            outputs = run_image(model, base, load_image(r["image"])[None].to(device))
            rows.append({"id": r["id"], **score(outputs, _labels(r))})
            if (i + 1) % 50 == 0:
                print(f"scored {i + 1}/{len(val)}", flush=True)
        with (args.out / "scores.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    chosen = choose(rows, args.n, args.min_road)
    for cat, picks in chosen.items():
        folder = args.out / cat
        folder.mkdir(exist_ok=True)
        for rank, row in enumerate(picks, 1):
            r = by_id[row["id"]]
            image = load_image(r["image"])[None]
            outputs = run_image(model, base, image.to(device))
            panel(row, image, _labels(r), outputs).save(folder / f"{_panel_name(rank, row['id'])}.png")
        print(f"{cat}: " + ", ".join(f"{r['id']} (q4 {r['q_k4']:.1f}, q0 {r['q_k0']:.1f}, "
                                     f"flip {r['z_flip']:.1f}%)" for r in picks), flush=True)
    (args.out / "chosen.json").write_text(json.dumps(
        {cat: [r["id"] for r in picks] for cat, picks in chosen.items()}, indent=1))
    print(f"wrote {args.out}")
    if args.share:
        share(args.out, args.share, args.title, args.min_road)


if __name__ == "__main__":
    main()
