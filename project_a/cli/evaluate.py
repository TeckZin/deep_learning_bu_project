#!/usr/bin/env python3
"""Score ONE (condition,K) run using an explicit manifest and supplied course masks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import VERSION, load_manifest, read_json, sha256, write_json
from metrics import Counts, image_counts, summarize


def evaluate(manifest, predictions):
    rows = load_manifest(manifest)
    conditions = {r["condition"] for r in rows}
    if len(conditions) != 1:
        raise ValueError("Each manifest must contain exactly one condition")
    root = Path(predictions)
    run = read_json(root / "run.json")
    if run["protocol"] != VERSION:
        raise ValueError("Protocol version mismatch")
    expected = {r["id"] for r in rows}
    recorded = {r["id"] for r in run["images"]}
    actual = {f.relative_to(root / "probabilities").with_suffix("").as_posix()
              for f in (root / "probabilities").rglob("*.npy")}
    if expected != recorded or expected != actual or len(recorded) != len(run["images"]):
        raise ValueError("Manifest/run/prediction IDs must match exactly, without duplicates")
    info = {r["id"]: r for r in run["images"]}
    total, domains = Counts(), {"urban": Counts(), "rural": Counts()}
    corruptions = {}
    fingerprint = []
    for row in sorted(rows, key=lambda r: r["id"]):
        image_id = row["id"]
        if info[image_id]["core_calls"] != run["steps"]:
            raise ValueError("Incorrect recurrent call count")
        if info[image_id]["input_sha256"] != sha256(row["image"]):
            raise ValueError("Predictions were generated for a different input image")
        probability = np.load(root / "probabilities" / (image_id + ".npy"), allow_pickle=False)
        with Image.open(row["label"]) as im:
            labels = np.asarray(im)
        with Image.open(row["image"]) as im:
            if labels.shape != (im.height, im.width):
                raise ValueError("Image/label dimensions disagree")
        c = image_counts(probability, labels)
        total = total + c
        domains[row["domain"]] = domains[row["domain"]] + c
        kind = row.get("corruption", {}).get("kind")
        if kind:
            corruptions[kind] = corruptions.get(kind, Counts()) + c
        fingerprint.append([image_id, row["domain"], sha256(row["label"])])
    return {"protocol": VERSION, "condition": next(iter(conditions)),
            "steps": run["steps"], "demo": run["demo"], "ablation": run["ablation"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "baseline_sha256": run["baseline_sha256"],
            "submission_source_sha256": run["submission_source_sha256"],
            "source_set_sha256": hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest(),
            "overall": summarize(total),
            "by_domain": {d: summarize(c) for d, c in domains.items() if c.valid},
            "by_corruption": {k: summarize(c) for k, c in corruptions.items() if c.valid},
            "number_of_images": len(rows)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = evaluate(args.manifest, args.predictions)
    write_json(args.output, report)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
