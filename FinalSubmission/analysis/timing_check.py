#!/usr/bin/env python3
"""
Quick check against the 15 s limit: time featurize on only the circuits most likely to be slowest.

Usage:
  python analysis/timing_check.py                                  # slowest training circuits
  python analysis/timing_check.py --circuits ../holdout-circuits   # largest files in another folder

Circuits run one at a time. Close other heavy programs first, or the times will be too high.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "submission-harness"))

from model import circuit_features  # noqa: E402
from run import read_qasm  # noqa: E402

LIMIT_S = 15.0


def slowest_training_circuits(count):
    with open(ROOT / "profile_features.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["feature_seconds"]), reverse=True)
    return [ROOT.parent / "train-circuits" / f"{r['filename']}.zst" for r in rows[:count]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuits", help="Folder to check. Without this, uses the slowest training circuits.")
    ap.add_argument("--top", type=int, default=25, help="How many circuits to time.")
    args = ap.parse_args()

    if args.circuits:
        files = [p for p in Path(args.circuits).rglob("*") if p.suffix in (".qasm", ".zst") and p.is_file()]
        files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:args.top]
    else:
        files = slowest_training_circuits(args.top)

    results = []
    for i, path in enumerate(files, 1):
        text = read_qasm(path)
        started = time.perf_counter()
        features = circuit_features(text)
        seconds = time.perf_counter() - started
        limits = features["hit_time_limit"] + features["mps8_hit_time_limit"] + features["mps16_hit_time_limit"]
        results.append((seconds, path.name, limits))
        print(f"  {i:3d}/{len(files)}  {seconds:6.2f} s  {path.name}" + ("  (safety limit hit)" if limits else ""), flush=True)

    results.sort(reverse=True)
    over = [r for r in results if r[0] > LIMIT_S]
    print(f"\nSlowest: {results[0][0]:.2f} s ({results[0][1]}). Over {LIMIT_S:.0f} s: {len(over)}. "
          f"Safety limit hit: {sum(r[2] > 0 for r in results)}.")
    if over:
        sys.exit(1)


if __name__ == "__main__":
    main()
