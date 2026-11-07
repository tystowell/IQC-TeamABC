#!/usr/bin/env python3
"""
Train the submission model and write submission-harness/artifacts/runtime_model.joblib.

Usage: python analysis/train_model.py [--exclude-list circuits.txt] [--artifacts DIR]

--exclude-list leaves listed circuits out, so the real harness can be scored on circuits the
model never saw. Needs analysis/profile_features.csv.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "submission-harness"))

from model import CAP_SECONDS, TIMEOUT_TARGET, feature_vector, physics_arrays, physics_log_seconds  # noqa: E402

N_TREES = 300
LEAF = 2


def load_runs():
    """One row per (circuit, threshold), with each circuit's profile attached as a dictionary."""
    profile = pd.read_csv(ROOT / "profile_features.csv")
    labels = pd.read_csv(ROOT.parent / "train-runtime-data.csv")
    labels["runtime"] = labels.duration_s.where(labels.status == "success", float(CAP_SECONDS))
    labels["timed_out"] = labels.status == "timeout"
    profiles = {row["filename"]: row for row in profile.to_dict("records")}
    labels["profile"] = labels.filename.map(profiles)
    return labels.reset_index(drop=True)


def fit_physics(arrays, log_runtime, timed_out):
    def loss(params):
        error = physics_log_seconds(params, arrays) - log_runtime
        # Over-predicting a timeout costs nothing under the metric.
        error = np.where(timed_out, np.minimum(error, 0), error)
        return np.mean(np.sqrt(error ** 2 + 1e-4))
    start = np.array([np.log(0.15), np.log(1e-5), np.log(1e-8), np.log(1e-2), 2.0])
    return minimize(loss, start, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-3, "fatol": 1e-5}).x


def trees(seed, n_jobs=4):
    return ExtraTreesRegressor(N_TREES, min_samples_leaf=LEAF, n_jobs=n_jobs, random_state=seed)


def fit_bundle(runs, feature_set="probe", seed=0, timeout_target=TIMEOUT_TARGET, n_jobs=4):
    profiles = list(runs.profile)
    log_runtime = np.log10(runs.runtime.values)
    timed_out = runs.timed_out.values
    target = np.where(timed_out, timeout_target, log_runtime)
    X = np.array([feature_vector(p, t, feature_set) for p, t in zip(profiles, runs.threshold)])
    arrays = physics_arrays(profiles, runs.threshold.values)
    params = fit_physics(arrays, log_runtime, timed_out)
    direct = trees(seed, n_jobs).fit(X, target)
    residual = trees(seed, n_jobs).fit(X, target - physics_log_seconds(params, arrays))
    return {"direct": direct, "residual": residual, "physics_params": params, "feature_set": feature_set}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude-list", help="File with one circuit filename per line to leave out of training.")
    ap.add_argument("--artifacts", default=str(ROOT.parent / "submission-harness" / "artifacts"))
    ap.add_argument("--no-probes", action="store_true", help="Train without the small simulations: structure only.")
    args = ap.parse_args()
    runs = load_runs()
    if args.exclude_list:
        excluded = {line.strip() for line in open(args.exclude_list) if line.strip()}
        runs = runs[~runs.filename.isin(excluded)].reset_index(drop=True)
    bundle = fit_bundle(runs, feature_set="all" if args.no_probes else "probe")
    out = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "runtime_model.joblib", compress=3)
    names = ["startup s", "s per op", "entanglement constant", "extra s per dynamic op"]
    print(f"Trained on {runs.filename.nunique()} circuits, {len(runs)} runs")
    print({name: float(f"{np.exp(value):.3g}") for name, value in zip(names, bundle["physics_params"][:4])},
          "chi exponent", round(float(bundle["physics_params"][4]), 2))
    print(f"Wrote {out / 'runtime_model.joblib'} ({(out / 'runtime_model.joblib').stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
