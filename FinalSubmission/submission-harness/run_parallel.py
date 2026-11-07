#!/usr/bin/env python3
"""
run_parallel.py  --  make submission.csv faster by running the unmodified run.py on batches at once.

Usage:
  python run_parallel.py --team "Your Team Name" --circuits ../holdout-circuits --workers 4
  (or, from the repo root: python pipeline.py predict --folder ../holdout-circuits --team "Your Team Name")

Parallel runs compete for CPU, so the recorded per-circuit times can be higher than in a normal
run. The script reports any over the 15 s limit. Use a lower --workers value if it warns.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP_S = 15.0
FIELDS = ["team", "filename", "threshold", "pred_duration_s", "parse_s", "predict_s"]


def find_circuits(folder):
    """The same files run.py would find."""
    return sorted(p.resolve() for p in Path(folder).rglob("*") if p.suffix in (".qasm", ".zst") and p.is_file())


def split_into_batches(files, workers):
    """Give each batch a similar total file size, placing the largest files first."""
    batches = [[] for _ in range(workers)]
    totals = [0] * workers
    for path in sorted(files, key=lambda p: p.stat().st_size, reverse=True):
        smallest = totals.index(min(totals))
        batches[smallest].append(path)
        totals[smallest] += path.stat().st_size
    return [batch for batch in batches if batch]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True, help="Your team name (appears in the output).")
    ap.add_argument("--circuits", default="../train-circuits", help="Directory of circuit files.")
    ap.add_argument("--thresholds", default="16,64,256,512,1024", help="Comma-separated thresholds to predict for.")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    args = ap.parse_args()

    files = find_circuits(args.circuits)
    if not files:
        sys.exit(f"No .qasm / .qasm.zst files found under {args.circuits}/")
    names = [p.name for p in files]
    if len(set(names)) != len(names):
        sys.exit("Two circuits share a filename, so their results couldn't be told apart.")

    started = time.perf_counter()
    batches = split_into_batches(files, args.workers)
    print(f"Running {len(files)} circuits in {len(batches)} batches ...", flush=True)

    # One numeric thread per process, so batches don't compete with their own math libraries.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1")
    rows = []
    failures = []
    with tempfile.TemporaryDirectory() as temp:
        jobs = []
        for i, batch in enumerate(batches):
            folder = Path(temp) / f"batch_{i}"
            folder.mkdir()
            for path in batch:
                (folder / path.name).symlink_to(path)
            out = Path(temp) / f"batch_{i}.csv"
            log = open(Path(temp) / f"batch_{i}.log", "w")
            process = subprocess.Popen([sys.executable, "run.py", "--team", args.team, "--circuits", str(folder),
                                        "--thresholds", args.thresholds, "--out", str(out)],
                                       cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((i, process, out, log))
        for i, process, out, log in jobs:
            code = process.wait()
            log.close()
            log_text = (Path(temp) / f"batch_{i}.log").read_text()
            problems = [line for line in log_text.splitlines() if line.strip().startswith("!")]
            if code != 0 or not out.exists():
                failures.append(f"batch {i} exited with code {code}:\n{log_text[-1500:]}")
                continue
            failures.extend(problems)
            with open(out, newline="") as f:
                rows.extend(csv.DictReader(f))
            print(f"  batch {i} finished at {time.perf_counter() - started:.0f} s", flush=True)

    rows.sort(key=lambda r: (r["filename"], int(r["threshold"])))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    expected = len(files) * len([t for t in args.thresholds.split(",") if t.strip()])
    per_circuit = {r["filename"]: float(r["parse_s"]) for r in rows}
    slowest = max(per_circuit.values(), default=0.0)
    over = [n for n, s in per_circuit.items() if s > CAP_S] + [r["filename"] for r in rows if float(r["predict_s"]) > CAP_S]
    print(f"\nWrote {len(rows)} of {expected} expected rows to {args.out} in {time.perf_counter() - started:.0f} s")
    print(f"Slowest featurize: {slowest:.2f} s")
    if failures:
        print(f"PROBLEMS, check before submitting ({len(failures)}):")
        for line in failures[:10]:
            print(" ", line)
    if over:
        print(f"WARNING: {len(set(over))} circuits were over the {CAP_S:.0f} s limit: {sorted(set(over))[:5]}. Try a lower --workers.")
    if len(rows) != expected:
        print("WARNING: rows are missing, and a missing run scores 0.")
    if failures or over or len(rows) != expected:
        sys.exit(1)


if __name__ == "__main__":
    main()
