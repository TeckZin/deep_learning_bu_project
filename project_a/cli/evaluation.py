"""Evaluation (doc stages 4b peek-in and 14): staff metrics on full tiles, at any K.

K=0 is y0 from the frozen encoder + H0 alone (the "peek-in" number the refiner must beat); the same
function scores the refiner at K=1..8. Rows need "image" and "label" paths (split rows
from splits.py or a manifest from common.load_manifest).
"""
import numpy as np
from PIL import Image
import torch

from data import load_image
from metrics import Counts, image_counts, leaderboard_score, summarize

VAL_KS = (0, 1, 2, 4)


def _labels(row):
    with Image.open(row["label"]) as im:
        return np.asarray(im)


def _counts(logits, labels):
    return image_counts(logits.float().sigmoid()[0, 0].cpu().numpy(), labels)


@torch.no_grad()
def validate(model, base, rows, device, ks=VAL_KS):
    """{K: summary} for the refiner on top of base, fp32, full tiles, no crop offset (like inference).

    model=None scores only K=0 (the frozen encoder + H0 alone).
    """
    ks = (0,) if model is None else tuple(ks)
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
        model.pos_offset = None
    totals = {k: Counts() for k in ks}
    kmax = max(ks)
    for row in rows:
        image = load_image(row["image"])[None].to(device)
        labels = _labels(row)
        features, y = base(image)
        outputs = {0: y}
        if kmax > 0:
            state = model.initialize(features, y)
            for t in range(1, kmax + 1):
                state, y = model.step(features, state, y)
                if t in ks:
                    outputs[t] = y
        for k in ks:
            totals[k] = totals[k] + _counts(outputs[k], labels)
    if model is not None:
        model.train(was_training)
    return {k: summarize(c) for k, c in totals.items()}


@torch.no_grad()
def validate_heads(base, rows, device):
    """K=0 summary for EVERY H0 head in base, running the encoder once per image."""
    heads = list(base.heads)
    modes = [h.training for h in heads]
    for h in heads:
        h.eval()
    totals = [Counts() for _ in heads]
    for row in rows:
        image = load_image(row["image"])[None].to(device)
        labels = _labels(row)
        f = base.features(image)
        for i, h in enumerate(heads):
            totals[i] = totals[i] + _counts(h(f, image.shape[-2:]), labels)
    for h, m in zip(heads, modes):
        h.train(m)
    return [summarize(c) for c in totals]


def selection_score(ordinary, stress=None):
    """Leaderboard formula in [0, 1]; without a stress set, the ordinary part renormalized."""
    a1, a2, a4 = (ordinary[k]["quality"] for k in (1, 2, 4))
    if stress is not None:
        return leaderboard_score(a1, a2, a4, stress[4]["quality"]) / 100
    return (0.2 * a1 + 0.2 * a2 + 0.4 * a4) / 0.8


def format_summary(s):
    return (f"quality {100 * s['quality']:.2f}  IoU {100 * s['road_iou']:.2f}  "
            f"clDice {100 * s['cldice']:.2f}  BIoU {100 * s['boundary_iou']:.2f}")


def print_val(results, label, stress=None):
    line = "  ".join(f"K={k}: {100 * results[k]['quality']:.2f}" for k in sorted(results))
    extra = f"  stress K=4: {100 * stress[4]['quality']:.2f}" if stress is not None else ""
    kmax = max(results)
    print(f"  val {label} quality  {line}{extra}  | K={kmax} IoU {100 * results[kmax]['road_iou']:.2f}"
          f"  clDice {100 * results[kmax]['cldice']:.2f}  BIoU {100 * results[kmax]['boundary_iou']:.2f}",
          flush=True)
