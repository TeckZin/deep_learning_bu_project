#!/usr/bin/env python3
"""Cross-validation splits (doc stage 3) and which data trains H0 vs the refiner.

The leaderboard is our test set. Train + Val are pooled and split into K folds
(default 5), stratified by domain (urban / rural). A run validates on one fold and
trains on the rest. Fold assignments live in ONE committed file, splits/folds.csv,
so every teammate and every job uses identical folds.

    python splits.py make --data-root $ROADSEG_DATA            # once; then commit splits/folds.csv
    python splits.py manifest --fold 0 --out val_fold0 --stress  # manifest (+ stress copy) for predict.py/evaluate.py

H0 vs refiner data inside the training folds (fold_plan, --h0-scheme):
  oof (default)  One H0 per training fold j, trained on the other training folds. The refiner
                 trains on ALL training folds; each image's y0 comes from the H0 that left its
                 fold out, so y0 always has unseen-image errors. A final H0 ("all") trained on
                 every training fold is used for validation and inference.
  halves         Training folds split 50/50 per domain: H0 on half a, refiner on half b.
                 The same H0 ("a") is used for validation and inference.

Data layout: <data-root>/{Train,Val}/{Urban,Rural}/{images_png,masks_png}/<stem>.png
Image ids are "<domain>/<stem>" (LoveDA stems are unique across Train and Val).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
DEFAULT_FOLDS = HERE / "splits" / "folds.csv"
SOURCES = ("Train", "Val")
DOMAINS = {"Urban": "urban", "Rural": "rural"}
FIELDS = ("id", "source", "domain", "stem", "fold")


def default_data_root():
    root = os.environ.get("ROADSEG_DATA")
    return Path(root) if root else None


def scan(data_root):
    """All (source, domain, stem) with both an image and a mask under data_root."""
    entries, seen = [], {}
    for source in SOURCES:
        for folder, domain in DOMAINS.items():
            images = Path(data_root) / source / folder / "images_png"
            masks = Path(data_root) / source / folder / "masks_png"
            if not images.is_dir():
                continue
            for path in sorted(images.glob("*.png")):
                if not (masks / path.name).is_file():
                    continue
                image_id = f"{domain}/{path.stem}"
                if image_id in seen:
                    raise ValueError(f"Duplicate id {image_id} in {seen[image_id]} and {source}")
                seen[image_id] = source
                entries.append({"id": image_id, "source": source, "domain": domain, "stem": path.stem})
    if not entries:
        raise FileNotFoundError(f"No {{Train,Val}}/{{Urban,Rural}}/images_png + masks_png under {data_root}")
    return entries


def make_folds(entries, k=5, seed=0):
    """Stratified by domain: each domain is shuffled (fixed seed) and dealt round-robin into k folds."""
    rows = []
    for domain in sorted({e["domain"] for e in entries}):
        group = sorted((e for e in entries if e["domain"] == domain), key=lambda e: e["id"])
        random.Random(f"{seed}:{domain}").shuffle(group)
        rows += [{**e, "fold": i % k} for i, e in enumerate(group)]
    return sorted(rows, key=lambda r: r["id"])


def write_folds(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in FIELDS} for r in rows)


def read_folds(path=DEFAULT_FOLDS, data_root=None):
    """Rows of folds.csv; with data_root, each row also gets absolute "image" and "label" paths."""
    with Path(path).open(newline="") as fh:
        rows = [{**r, "fold": int(r["fold"])} for r in csv.DictReader(fh)]
    if data_root is not None:
        folder = {v: k for k, v in DOMAINS.items()}
        for r in rows:
            base = Path(data_root) / r["source"] / folder[r["domain"]]
            r["image"] = str(base / "images_png" / f"{r['stem']}.png")
            r["label"] = str(base / "masks_png" / f"{r['stem']}.png")
    return rows


def halves(rows, seed=0):
    """Split rows 50/50 within each domain: returns (a, b)."""
    a, b = [], []
    for domain in sorted({r["domain"] for r in rows}):
        group = sorted((r for r in rows if r["domain"] == domain), key=lambda r: r["id"])
        random.Random(f"{seed}:halves:{domain}").shuffle(group)
        a += group[: len(group) // 2]
        b += group[len(group) // 2:]
    return a, b


def fold_plan(rows, val_fold=0, scheme="oof", seed=0):
    """Which rows train each H0, which train the refiner, and which H0 gives each image its y0.

    Returns a dict:
      val       rows of the validation fold
      refiner   rows the refiner trains on
      h0        {name: rows that H0 head trains on}   (names: oof<j> + all, or a)
      route     {image id: H0 name}   for every refiner row (never an H0 that saw the image)
      eval_h0   name of the H0 used for validation / inference
    """
    folds = sorted({r["fold"] for r in rows})
    if val_fold not in folds:
        raise ValueError(f"val fold {val_fold} not in folds {folds}")
    val = [r for r in rows if r["fold"] == val_fold]
    train = [r for r in rows if r["fold"] != val_fold]
    if scheme == "oof":
        train_folds = [f for f in folds if f != val_fold]
        h0 = {f"oof{j}": [r for r in train if r["fold"] != j] for j in train_folds}
        h0["all"] = train
        route = {r["id"]: f"oof{r['fold']}" for r in train}
        return {"val": val, "refiner": train, "h0": h0, "route": route, "eval_h0": "all"}
    if scheme == "halves":
        a, b = halves(train, seed)
        return {"val": val, "refiner": b, "h0": {"a": a}, "route": {r["id"]: "a" for r in b},
                "eval_h0": "a"}
    raise ValueError(f"Unknown H0 scheme {scheme!r}; use 'oof' or 'halves'")


def write_manifest(rows, out):
    """Folder with images/<domain>/<stem>.png symlinks + manifest.json (the make_manifest.py format)."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing nonempty output folder: {out}")
    manifest = []
    for r in rows:
        link = out / "images" / f"{r['id']}.png"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(Path(r["image"]).resolve())
        manifest.append({"id": r["id"], "domain": r["domain"], "condition": "ordinary",
                         "image": str(link.relative_to(out)), "label": str(Path(r["label"]).resolve())})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out / "manifest.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="Assign every Train+Val image to a fold -> splits/folds.csv")
    m.add_argument("--data-root", type=Path, default=default_data_root(), help="Default: $ROADSEG_DATA")
    m.add_argument("--k", type=int, default=5)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", type=Path, default=DEFAULT_FOLDS)
    m.add_argument("--force", action="store_true", help="Overwrite an existing folds file")
    v = sub.add_parser("manifest", help="Manifest folder for one fold (for predict.py / evaluate.py)")
    v.add_argument("--data-root", type=Path, default=default_data_root(), help="Default: $ROADSEG_DATA")
    v.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    v.add_argument("--fold", type=int, default=0)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--stress", action="store_true", help="Also write <out>_stress with make_stress.py")
    v.add_argument("--seed", type=int, default=0, help="Stress seed")
    args = p.parse_args(argv)
    if args.data_root is None:
        p.error("--data-root or $ROADSEG_DATA is required")

    if args.cmd == "make":
        if args.out.exists() and not args.force:
            p.error(f"{args.out} exists; it is shared by the whole team. Use --force to overwrite.")
        rows = make_folds(scan(args.data_root), args.k, args.seed)
        write_folds(rows, args.out)
        for f in range(args.k):
            n = [sum(r["fold"] == f and r["domain"] == d for r in rows) for d in ("urban", "rural")]
            print(f"fold {f}: {n[0]} urban + {n[1]} rural")
        print(f"Wrote {len(rows)} rows to {args.out}")
    else:
        rows = [r for r in read_folds(args.folds, args.data_root) if r["fold"] == args.fold]
        manifest = write_manifest(rows, args.out)
        print(f"fold {args.fold}: {len(rows)} images -> {manifest}")
        if args.stress:
            from make_stress import generate
            generate(manifest, Path(f"{args.out}_stress"), args.seed)
            print(f"stress copy -> {args.out}_stress/manifest.json")


if __name__ == "__main__":
    main()
