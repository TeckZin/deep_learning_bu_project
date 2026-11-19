"""Segmentation losses over valid pixels (no-data pixels excluded). Computed in fp32.

    seg_loss = BCE (optionally edge-weighted) + dice_weight * soft Dice (+ cldice_weight * soft clDice)
"""
import torch
from torch.nn import functional as F


def edge_band(target, radius=3):
    """1 on pixels within `radius` px of a road edge (both sides), else 0.

    Matches the Boundary IoU band: the metric erodes with a 3x3 square 3 times,
    i.e. a 7x7 square, so radius=3 covers the same pixels.
    """
    k = 2 * radius + 1
    dilated = F.max_pool2d(target, k, stride=1, padding=radius)
    eroded = -F.max_pool2d(-target, k, stride=1, padding=radius)
    return (dilated - eroded).clamp(0, 1)


def _soft_erode(x):
    return -F.max_pool2d(-x, 3, stride=1, padding=1)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_skeleton(x, iters=10):
    """Differentiable skeleton (Shit et al., clDice, CVPR 2021)."""
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(logits, target, valid, iters=10):
    """1 - soft clDice over valid pixels; rewards connected, centred road skeletons."""
    p = logits.float().sigmoid() * valid
    t = target * valid
    sp, st = soft_skeleton(p, iters), soft_skeleton(t, iters)
    tprec = ((sp * t).sum() + 1) / (sp.sum() + 1)
    tsens = ((st * p).sum() + 1) / (st.sum() + 1)
    return 1 - 2 * tprec * tsens / (tprec + tsens)


def seg_loss(logits, target, valid, pos_weight=1.0, dice_weight=1.0, edge_weight=1.0,
             cldice_weight=0.0):
    """BCE + soft Dice over valid pixels only (no-data pixels excluded). Computed in fp32.

    edge_weight > 1 multiplies the BCE of pixels near road edges (the Boundary IoU band),
    so the model is paid to put edges in the right place. 1.0 = plain BCE.
    """
    logits = logits.float()
    valid = valid.float()
    pw = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    weight = valid
    if edge_weight != 1.0:
        weight = valid * (1 + (edge_weight - 1) * edge_band(target))
    bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
    p = logits.sigmoid() * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    loss = bce + dice_weight * dice
    if cldice_weight > 0:
        loss = loss + cldice_weight * soft_cldice_loss(logits, target, valid)
    return loss
