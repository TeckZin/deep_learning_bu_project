#!/usr/bin/env python3
"""Combine validated evaluation reports using the public leaderboard formula."""
import argparse
from pathlib import Path
from common import VERSION, read_json, write_json
from metrics import leaderboard_score


def combine(reports, allow_demo=False):
    if len(reports) != 4:
        raise ValueError("Exactly four reports are required")
    for report, (condition, k) in zip(reports, [("ordinary", 1), ("ordinary", 2),
                                              ("ordinary", 4), ("stress", 4)]):
        if (report["condition"], report["steps"]) != (condition, k):
            raise ValueError("Wrong condition or K for score component")
        if report["protocol"] != VERSION or report["ablation"] != "none":
            raise ValueError("Protocol mismatch or diagnostic ablation")
        if report["demo"] and not allow_demo:
            raise ValueError("Demo runs cannot enter the official leaderboard")
    for field in ("checkpoint_sha256", "baseline_sha256", "submission_source_sha256",
                  "source_set_sha256", "demo"):
        if len({r[field] for r in reports}) != 1:
            raise ValueError(f"Runs differ in {field}")
    quality = [r["overall"]["quality"] for r in reports]
    return {"protocol": VERSION, "demo": reports[0]["demo"], "quality_components": quality,
            "weights": [0.2, 0.2, 0.4, 0.2], "score": leaderboard_score(*quality)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("ordinary-1", "ordinary-2", "ordinary-4", "stress-4"):
        p.add_argument("--" + arg, required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--allow-demo", action="store_true")
    args = p.parse_args()
    result = combine([read_json(getattr(args, name)) for name in
                      ("ordinary_1", "ordinary_2", "ordinary_4", "stress_4")], args.allow_demo)
    write_json(args.output, result)
    print(f"Score: {result['score']:.6f}" + (" (DEMO ONLY)" if result["demo"] else ""))


if __name__ == "__main__":
    main()
