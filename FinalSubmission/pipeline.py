#!/usr/bin/env python3
"""
pipeline.py  --  the whole runtime predictor, one step at a time.

    Step 1  parse     read one circuit into a standard dictionary of operations
    Step 2  features  turn circuits into feature vectors (one circuit, or every training circuit)
    Step 3  train     fit the model on the training features and labels
    Step 4  predict   predict runtimes: one circuit, or a whole folder (the hold-out set) -> submission.csv
            score     score a submission against labels with the organizers' scorer
            check     known-answer tests and the 15 s timing check

Typical use:

    python pipeline.py parse    train-circuits/0071a13b.qasm.zst
    python pipeline.py features train-circuits/0071a13b.qasm.zst     # one circuit, printed
    python pipeline.py features --all --workers 4                     # all training circuits -> analysis/profile_features.csv (~5 min)
    python pipeline.py train                                          # -> submission-harness/artifacts/runtime_model.joblib (~2 min)
    python pipeline.py predict  train-circuits/0071a13b.qasm.zst      # one circuit, all five thresholds
    python pipeline.py check                                          # tests + timing (~3 min)

On hold-out day (circuits arrive in the challenge repo's holdout-circuits/ folder):

    python pipeline.py predict --folder ../challenge/holdout-circuits --team "Our Team" --workers 4
    # -> submission-harness/submission.csv, made by the organizers' unmodified run.py; send it as a DM

Steps 2 and 3 only need re-running if the feature code or the model settings change; the repo ships
analysis/profile_features.csv for the training circuits. The batch steps call the same scripts the
model was validated with (analysis/*.py and submission-harness/run.py), so results are identical.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HARNESS = ROOT / "submission-harness"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HARNESS))
THRESHOLDS = (16, 64, 256, 512, 1024)


def run_script(*args, cwd=ROOT):
    """Run one of the repo's scripts with this Python, stopping if it fails."""
    command = [sys.executable, *map(str, args)]
    print("$", " ".join(command[1:]), flush=True)
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


# ------------------------------------------------------------------ step 1: parse
def parse_circuit(path):
    """Read a .qasm or .qasm.zst file (OpenQASM 2 or 3) into the standard dictionary from visualize_qasm.parse_qasm."""
    from visualize_qasm import parse_qasm
    return parse_qasm(path, max_ops=200_000)


def cmd_parse(args):
    circuit = parse_circuit(args.circuit)
    names = {}
    for op in circuit["operations"]:
        names[op["name"]] = names.get(op["name"], 0) + 1
    print(f"{circuit['source']}: OpenQASM {circuit['qasm_version']}, {circuit['num_qubits']} qubits, "
          f"{circuit['num_operations']:,} operations{' (stopped early)' if circuit['truncated'] else ''}")
    print("operations by name:", dict(sorted(names.items(), key=lambda kv: -kv[1])))
    for op in circuit["operations"][:args.show]:
        print("  ", op["name"], op["qubits"], op["params"] or "")


# ------------------------------------------------------------------ step 2: features
def extract_features(qasm_text):
    """All model inputs for one circuit: structural entanglement profile, two small MPS probes, and a text scan.
    This is exactly what the harness calls as featurize(), and must finish within 15 s."""
    from model import circuit_features
    return circuit_features(qasm_text)


def read_circuit_text(path):
    from run import read_qasm  # the organizers' reader, handles .zst
    return read_qasm(Path(path))


def cmd_features(args):
    if args.all:
        run_script("analysis/profile_features.py", "--workers", args.workers)
        return
    if not args.circuit:
        sys.exit("give a circuit file, or --all for every training circuit")
    started = time.perf_counter()
    features = extract_features(read_circuit_text(args.circuit))
    seconds = time.perf_counter() - started
    keys = ["n_qubits", "n_ops", "n_multi", "coverage", "peak_est", "mps8_peak_max_entropy", "mps16_peak_max_entropy",
            "mps16_discarded_total", "mps16_tracked_ops", "n_measure", "n_reset", "text_qubit_top10_share", "text_angle_median"]
    print(f"{len(features)} features in {seconds:.2f} s. A few of them:")
    for k in keys:
        print(f"  {k:26s} {features[k]:.4g}" if isinstance(features[k], float) else f"  {k:26s} {features[k]}")


# ------------------------------------------------------------------ step 3: train
def cmd_train(args):
    """Fits physics formula + trees (+ gated neural network) on analysis/profile_features.csv and train-runtime-data.csv."""
    extra = ["--no-network"] if args.no_network else []
    if args.exclude_list:
        extra += ["--exclude-list", Path(args.exclude_list).resolve()]
    run_script("analysis/train_model.py", *extra)


# ------------------------------------------------------------------ step 4: predict
def load_model():
    from model import RuntimeModel
    return RuntimeModel(HARNESS / "artifacts")


def cmd_predict(args):
    if args.folder:
        # The real submission: the organizers' run.py, unmodified, on batches in parallel, merged into one CSV.
        out = Path(args.out).resolve()
        run_script("run_parallel.py", "--team", args.team, "--circuits", Path(args.folder).resolve(), "--out", out,
                   "--workers", args.workers, cwd=HARNESS)
        print(f"\nWrote {out}. Check above for any WARNING about the 15 s limit before sending it.")
        return
    if not args.circuit:
        sys.exit("give a circuit file, or --folder DIR --team NAME for a whole folder")
    model = load_model()
    started = time.perf_counter()
    features = model.featurize(read_circuit_text(args.circuit))
    featurize_s = time.perf_counter() - started
    print(f"featurize: {featurize_s:.2f} s; familiar to the training data: {model.is_familiar(features)}")
    for threshold in ([args.threshold] if args.threshold else THRESHOLDS):
        print(f"  threshold {threshold:>5}: {model.predict(features, threshold):10.3f} s")


def cmd_score(args):
    run_script("score.py", "--pred", Path(args.pred).resolve(), "--labels", Path(args.labels).resolve(), cwd=HARNESS)


def cmd_check(args):
    tests = ROOT / "tests" if (ROOT / "tests").exists() else ROOT / "release-v2" / "tests"
    for test in ("test_profile.py", "test_probe.py"):
        run_script(tests / test)
    run_script("analysis/timing_check.py", "--top", args.top)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = ap.add_subparsers(dest="step", required=True)

    p = steps.add_parser("parse", help="step 1: read one circuit and summarize it")
    p.add_argument("circuit")
    p.add_argument("--show", type=int, default=10, help="how many operations to print")
    p.set_defaults(run=cmd_parse)

    p = steps.add_parser("features", help="step 2: features for one circuit, or --all training circuits")
    p.add_argument("circuit", nargs="?")
    p.add_argument("--all", action="store_true", help="recompute analysis/profile_features.csv for every training circuit")
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(run=cmd_features)

    p = steps.add_parser("train", help="step 3: fit the model")
    p.add_argument("--no-network", action="store_true", help="trees and physics only")
    p.add_argument("--exclude-list", help="file of circuit names to leave out (for honest held-out tests)")
    p.set_defaults(run=cmd_train)

    p = steps.add_parser("predict", help="step 4: predict one circuit, or a folder -> submission.csv")
    p.add_argument("circuit", nargs="?")
    p.add_argument("--threshold", type=int)
    p.add_argument("--folder", help="folder of circuits, e.g. the hold-out set")
    p.add_argument("--team", default="Our Team")
    p.add_argument("--out", default=str(HARNESS / "submission.csv"))
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(run=cmd_predict)

    p = steps.add_parser("score", help="score a submission with the organizers' score.py")
    p.add_argument("--pred", default=str(HARNESS / "submission.csv"))
    p.add_argument("--labels", default=str(ROOT / "train-runtime-data.csv"))
    p.set_defaults(run=cmd_score)

    p = steps.add_parser("check", help="known-answer tests and 15 s timing check")
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(run=cmd_check)

    args = ap.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
