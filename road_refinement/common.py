"""Small utilities shared by training and the offline inference package."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from functools import lru_cache

import numpy as np
import torch

PROTOCOL = "ec523-midway-temporary-baseline-v1"
DEFAULT_DATA = "/projectnb/ec523kb/projects/RoadSeg/dataset"
TEAM_ROOT = Path("/projectnb/ec523kb/projects/CowenZeollaZinTan")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "exported-source"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_for(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; run inside a GPU job or explicitly use --device cpu")
    return torch.device(name)


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing nonempty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def source_hash():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("road_refinement", "src", "project_a/cli/submission"):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=None)
def course_module(name):
    """Load original course modules despite their script-local imports."""
    from project_a.cli import common
    from project_a import cli
    path = Path(next(iter(cli.__path__))) / f"{name}.py"
    module_name = f"road_refinement._course_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("common")
    sys.modules["common"] = common
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    return module
