"""Evaluation (doc stages 4b peek-in and 14): staff metrics on full tiles, at any K.

K=0 is y0 from the frozen encoder + H0 alone (the "peek-in" number the refiner must beat); the same
function scores the refiner at K=1..8. Rows need "image" and "label" paths (split rows
from splits.py or a manifest from common.load_manifest).
"""
from collections import defaultdict

import numpy as np
from PIL import Image
import torch

from data import load_image, load_mask
from losses import seg_loss
from metrics import Counts, image_counts, leaderboard_score, summarize

VAL_KS = (0, 1, 2, 4)
DIAG_STEPS = 8          # loop_diagnostics runs past the trained K=4 to see whether the loop settles


def _labels(row):
    with Image.open(row["label"]) as im:
        return np.asarray(im)


def _counts(logits, labels):
    return image_counts(logits.float().sigmoid()[0, 0].cpu().numpy(), labels)


@torch.no_grad()
def validate(model, base, rows, device, ks=VAL_KS, loss_args=None):
    """{K: summary} for the refiner on top of base, fp32, full tiles, no crop offset (like inference).

    model=None scores only K=0 (the frozen encoder + H0 alone).
    loss_args: (pos_weight, dice_weight, edge_weight, cldice_weight) as in training. When given, each
    summary also has "loss": the training loss of y_K averaged over tiles (reuses the outputs, so cheap).
    """
    ks = (0,) if model is None else tuple(ks)
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
        model.pos_offset = None
    totals = {k: Counts() for k in ks}
    losses = {k: 0.0 for k in ks}
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
        if loss_args is not None:
            target, valid = (t[None].to(device) for t in load_mask(row["label"]))   # [1, 1, H, W]
            for k in ks:
                losses[k] += seg_loss(outputs[k], target, valid, *loss_args).item()
    if model is not None:
        model.train(was_training)
    results = {k: summarize(c) for k, c in totals.items()}
    if loss_args is not None:
        for k in ks:
            results[k]["loss"] = losses[k] / max(len(rows), 1)
    return results


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


@torch.no_grad()
def loop_diagnostics(model, base, rows, device, loss_args, steps=DIAG_STEPS):
    """How the loop behaves step by step, on a few full validation tiles (fp32, like inference).

    Per step t (mean over tiles):
      dy_t     mean |y_t - y_t-1|: size of each correction (healthy: shrinks, but is not ~0 from t=2)
      dz_t     mean |z_t - z_t-1|: how much the memory changes
      gate_t   mean GRU gate u (spatial memory only): ~0 = z frozen, ~1 = z overwritten every step
      loss_t   training loss of y_t, t = 0..steps: past the trained K it should level off, not rise
    Plus quality at K=4 and K=steps, and at K=4 under the staff runner's ablations (reset-state,
    no-mask-feedback): both should be clearly below plain K=4 if the recurrent state is in charge.
    """
    was_training = model.training
    model.eval()
    model.pos_offset = None
    sums = defaultdict(float)
    counts = {name: Counts() for name in ("k4", f"k{steps}", "k4_reset", "k4_nofeedback")}
    for row in rows:
        image = load_image(row["image"])[None].to(device)
        labels = _labels(row)
        target, valid = (t[None].to(device) for t in load_mask(row["label"]))
        features, y0 = base(image)
        sums["loss_t0"] += seg_loss(y0, target, valid, *loss_args).item()
        state, y = model.initialize(features, y0), y0
        for t in range(1, steps + 1):
            next_state, next_y = model.step(features, state, y)
            sums[f"dy_t{t}"] += (next_y - y).abs().mean().item()
            sums[f"dz_t{t}"] += (next_state - state).abs().mean().item()
            gate = getattr(model.head, "gate_mean", None)
            if gate is not None:
                sums[f"gate_t{t}"] += gate.item()
            sums[f"loss_t{t}"] += seg_loss(next_y, target, valid, *loss_args).item()
            state, y = next_state, next_y
            if t == 4:
                counts["k4"] += _counts(y, labels)
        counts[f"k{steps}"] += _counts(y, labels)
        # The runner's ablations: reset z before every step / always feed back y0 instead of y_t.
        for name, reset, feedback in (("k4_reset", True, True), ("k4_nofeedback", False, False)):
            state, y = model.initialize(features, y0), y0
            for _ in range(4):
                if reset:
                    state = model.initialize(features, y0)
                state, y = model.step(features, state, y if feedback else y0)
            counts[name] += _counts(y, labels)
    model.train(was_training)
    out = {key: value / max(len(rows), 1) for key, value in sums.items()}
    out.update({f"quality_{name}": 100 * summarize(c)["quality"] for name, c in counts.items()})
    return out


def print_diagnostics(diag, n_tiles, steps=DIAG_STEPS):
    """Log lines for loop_diagnostics."""
    def row(prefix, first=1):
        return " ".join(f"{diag[f'{prefix}_t{t}']:.4f}" for t in range(first, steps + 1) if f"{prefix}_t{t}" in diag)
    print(f"  loop ({n_tiles} tiles)  |dy| t1..{steps}: {row('dy')}", flush=True)
    print(f"                    |dz| t1..{steps}: {row('dz')}", flush=True)
    if "gate_t1" in diag:
        print(f"                    gate t1..{steps}: {row('gate')}", flush=True)
    print(f"                    loss t0..{steps}: {row('loss', 0)}", flush=True)
    print(f"                    quality K=4 {diag['quality_k4']:.2f}  K={steps} {diag[f'quality_k{steps}']:.2f}"
          f"  | K=4 reset-state {diag['quality_k4_reset']:.2f}"
          f"  no-mask-feedback {diag['quality_k4_nofeedback']:.2f}", flush=True)


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
    if "loss" in results[kmax]:
        print("  val loss by step:    " + " | ".join(f"y{k} {results[k]['loss']:.4f}" for k in sorted(results)),
              flush=True)
