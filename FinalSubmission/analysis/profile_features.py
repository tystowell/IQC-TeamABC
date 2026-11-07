#!/usr/bin/env python3
"""
Compute all model features for every training circuit and write analysis/profile_features.csv.

Usage: python analysis/profile_features.py [--workers 2]   (or: python pipeline.py features --all)

Timing is measured the way the harness measures featurize: decompression isn't included.
"""

import argparse
import csv
import multiprocessing
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "submission-harness"))

from model import circuit_features  # noqa: E402
from run import read_qasm  # noqa: E402


def featurize(path):
    row = circuit_features(read_qasm(path))
    row["filename"] = path.name[:-4]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    files = sorted((REPO / "train-circuits").glob("*.qasm.zst"), key=lambda p: p.stat().st_size)
    rows = []
    with multiprocessing.Pool(args.workers, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.imap_unordered(featurize, files, chunksize=1), 1):
            rows.append(row)
            if i % 100 == 0 or i == len(files):
                print(f"{i}/{len(files)} circuits", flush=True)
    rows.sort(key=lambda r: r["filename"])
    out = Path(__file__).resolve().parent / "profile_features.csv"
    fields = ["filename"] + [k for k in rows[0] if k != "filename"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    seconds = sorted(r["feature_seconds"] for r in rows)
    print(f"Wrote {out}")
    print(f"Seconds per circuit: median {seconds[len(seconds) // 2]:.3f}, max {seconds[-1]:.2f}; "
          f"over 15 s: {sum(s > 15 for s in seconds)}")
    print(f"Fully tracked: {sum(r['coverage'] >= 1 for r in rows)} of {len(rows)}")


if __name__ == "__main__":
    main()
