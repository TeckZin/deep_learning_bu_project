"""Export a complete, local-only inference directory, including required code."""
from pathlib import Path
import shutil

from .common import PROTOCOL, fresh_directory, read_json, sha256, source_hash, write_json


def export_bundle(run, output):
    run = Path(run).resolve()
    config = read_json(run / "config.json")
    checkpoints = run / "checkpoints"
    baseline = checkpoints / "refiner_baseline.safetensors"
    refiner = checkpoints / "refiner_best.safetensors"
    for path in (baseline, refiner):
        if not path.is_file():
            raise ValueError(f"Training artifact missing: {path}")
    if config["source_sha256"] != source_hash():
        raise ValueError("Run source differs from current source; export from the same implementation")
    root = fresh_directory(output)
    shutil.copyfile(baseline, root / "baseline.safetensors")
    shutil.copyfile(refiner, root / "model.safetensors")
    shutil.copyfile(run / "config.json", root / "config.json")
    for name in ("encoder.json", "refiner_baseline.json", "invocations.jsonl", "metrics.jsonl",
                 "baseline_validation.json", "refiner_validation.json"):
        if (run / name).is_file():
            shutil.copyfile(run / name, root / name)
    write_json(root / "bundle.json", {
        "protocol": PROTOCOL, "baseline_kind": "temporary_resnet18_unet", "official_baseline": False,
        "synthetic": config["synthetic"], "git_revision": config["git_revision"],
        "source_sha256": source_hash(), "architecture": config["architecture"],
        "weights": {name: sha256(root / name) for name in ("baseline.safetensors", "model.safetensors")},
    })
    source = Path(__file__).resolve().parents[1]
    code = root / "code"
    code.mkdir()
    for name in ("road_refinement", "src", "project_a/cli"):
        shutil.copytree(source / name, code / name, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.safetensors", "*.pt", "*.pth", "smoke_outputs", "predictions", ".venv"))
    shutil.copyfile(source / "pyproject.toml", code / "pyproject.toml")
    # Include exact inference dependencies; optimizer state and dataset files are excluded.
    shutil.copyfile(source / "requirements_midway.txt", root / "requirements.txt")
    (root / "README.md").write_text(
        "# EC523 midway inference bundle\n\n"
        "This uses our temporary ResNet18/U-Net baseline, not the staff baseline.\n"
        + ("**SYNTHETIC TEST ARTIFACT: not trained on course data.**\n\n" if config["synthetic"] else "\n")
        + "Create a Python 3.12 environment. Install matched torch 2.6.0 / torchvision 0.21.0 "
        "CPU or CUDA wheels, then run `python -m pip install -r requirements.txt` and "
        "`python -m pip install --no-deps ./code`. Dependencies must be installed before offline execution; "
        "the inference command never downloads model weights.\n\n"
        "From this bundle directory:\n\n"
        "```bash\npython -m road_refinement predict --bundle . --input /path/to/image_or_folder "
        "--output /path/to/new_predictions --steps 4 --device cpu\n```\n\n"
        "Use `--device cuda` inside an allocated GPU job. K may be 0 through 8. "
        "Input is RGB PNG/JPEG; a folder is searched recursively and should contain images only. "
        "Output contains probabilities/<relative_id>.npy (float32 HxW), masks/<relative_id>.png "
        "(0/255 at probability >= 0.5), and run.json. `--save-trace` saves steps 0..K. "
        "Labels are not needed or read during inference.\n"
    )
    return root
