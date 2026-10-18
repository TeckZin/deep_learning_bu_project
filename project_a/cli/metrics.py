"""Exact hard-mask metrics from the handout. Counts are pooled across tiles.

Do not average per-image scores. No epsilon is used in evaluation; all empty
cases are defined. Zhang skeletonization is pinned by requirements.txt.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.morphology import skeletonize

from common import check_probability

SQUARE = np.ones((3, 3), dtype=bool)


@dataclass
class Counts:
    valid: int = 0
    intersection: int = 0
    union: int = 0
    predicted_skeleton: int = 0
    true_skeleton: int = 0
    skeleton_precision_hits: int = 0
    skeleton_recall_hits: int = 0
    boundary_intersection: int = 0
    boundary_union: int = 0

    def __add__(self, other):
        return Counts(**{key: value + getattr(other, key) for key, value in asdict(self).items()})


def image_counts(probability, labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Course label must be a 2D integer array, not RGB")
    if (labels < 0).any() or (labels > 7).any():
        raise ValueError("Expected supplied label IDs 0..7 (road=3, ignore=0)")
    probability = check_probability(probability, labels.shape)
    valid = labels != 0
    predicted = (probability >= 0.5) & valid
    truth = labels == 3
    sp = skeletonize(predicted, method="zhang")
    sg = skeletonize(truth, method="zhang")
    # Ignore image frame and neighborhoods of no-data in boundary metric.
    safe = binary_erosion(valid, structure=SQUARE, iterations=3, border_value=0)
    def boundary(mask):
        eroded = binary_erosion(mask, structure=SQUARE, iterations=3, border_value=0)
        return mask & ~eroded & safe
    bp, bg = boundary(predicted), boundary(truth)
    n = lambda x: int(np.count_nonzero(x))
    return Counts(n(valid), n(predicted & truth), n(predicted | truth), n(sp), n(sg),
                  n(sp & truth), n(sg & predicted), n(bp & bg), n(bp | bg))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 1.0


def summarize(counts):
    if counts.valid == 0:
        raise ValueError("Evaluation subset contains no valid pixels")
    iou = ratio(counts.intersection, counts.union)
    precision = ratio(counts.skeleton_precision_hits, counts.predicted_skeleton)
    recall = ratio(counts.skeleton_recall_hits, counts.true_skeleton)
    cldice = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    boundary_iou = ratio(counts.boundary_intersection, counts.boundary_union)
    return {"road_iou": iou, "cldice": cldice, "boundary_iou": boundary_iou,
            "skeleton_precision": precision, "skeleton_recall": recall,
            "quality": 0.5 * iou + 0.3 * cldice + 0.2 * boundary_iou,
            "counts": asdict(counts)}


def leaderboard_score(a1, a2, a4, stress4):
    values = np.asarray([a1, a2, a4, stress4], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Quality values must be finite and in [0,1]")
    return float(100 * np.dot(values, [0.2, 0.2, 0.4, 0.2]))
