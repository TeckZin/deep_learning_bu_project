#!/usr/bin/env python3
"""Exercise the REAL command-line workflow with synthetic data and random weights.

No training or GPU benchmark is performed. Outputs are explicitly marked demo.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from common import fresh_directory, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    root = fresh_directory(args.output).resolve()
    code = Path(__file__).resolve().parent
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    def run(script, *options):
        subprocess.run([sys.executable, str(code / script), *map(str, options)],
                       cwd=code, env=env, check=True)
    run("make_demo.py", "--output", root / "demo")
    run("make_stress.py", "--manifest", root / "demo/manifest.json",
        "--output", root / "stress", "--seed", 523)
    for name, k, folder in [("ordinary1", 1, "demo"), ("ordinary2", 2, "demo"),
                            ("ordinary4", 4, "demo"), ("stress4", 4, "stress")]:
        run("predict.py", "--demo", "--checkpoint", root / "demo/model.safetensors",
            "--input", root / folder / "images", "--output", root / name,
            "--steps", k, "--save-trace")
        run("evaluate.py", "--manifest", root / folder / "manifest.json",
            "--predictions", root / name, "--output", root / (name + ".json"))
    run("score.py", "--ordinary-1", root / "ordinary1.json", "--ordinary-2", root / "ordinary2.json",
        "--ordinary-4", root / "ordinary4.json", "--stress-4", root / "stress4.json",
        "--output", root / "score.json", "--allow-demo")
    result = read_json(root / "score.json")
    assert result["demo"] and 0 <= result["score"] <= 100
    print("PASS: four-run CLI workflow; this is NOT a course-data performance result.")


if __name__ == "__main__":
    main()
