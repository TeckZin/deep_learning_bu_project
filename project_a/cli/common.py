"""Shared I/O validation. Never load pickled checkpoint or array objects."""
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "ec523-project-a-v1-draft"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_id(value):
    p = Path(value)
    if not value or p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError(f"Invalid relative image id: {value!r}")
    if value != p.as_posix():
        raise ValueError(f"Noncanonical id: {value!r}")
    return value


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing nonempty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(path):
    path = Path(path)
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list")
    ids = set()
    for row in rows:
        image_id = safe_id(row["id"])
        if image_id in ids:
            raise ValueError(f"Duplicate id: {image_id}")
        ids.add(image_id)
        if row["domain"] not in ("urban", "rural"):
            raise ValueError("domain must be urban or rural")
        if row["condition"] not in ("ordinary", "stress"):
            raise ValueError("condition must be ordinary or stress")
        for key in ("image", "label"):
            row[key] = str((path.parent / row[key]).resolve())
    return rows


def check_probability(p, shape):
    if p.shape != tuple(shape) or p.dtype != np.float32:
        raise ValueError(f"Expected float32 probability map {shape}, got {p.shape}/{p.dtype}")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    return p
