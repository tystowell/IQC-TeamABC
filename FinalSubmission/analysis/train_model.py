#!/usr/bin/env python3
"""
Train the submission model and write submission-harness/artifacts/runtime_model.joblib.

Usage: python analysis/train_model.py [--exclude-list circuits.txt] [--artifacts DIR] [--no-network] [--no-probes]
   (or: python pipeline.py train)

Fits, on every (circuit, threshold) run: the physics formula, trees for runtime and for the formula's
error, five small neural networks for each, and the "is this circuit familiar?" gate. Needs
analysis/profile_features.csv and train-runtime-data.csv.

--exclude-list leaves listed circuits out, so the real harness can be scored on circuits the
model never saw.
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

from model import (CAP_SECONDS, GATE_PERCENTILE, PHYSICS_NAMES, PHYSICS_START, TIMEOUT_TARGET, feature_vector, gate_family,  # noqa: E402
                   make_gate_space, make_network, physics_arrays, physics_log_seconds)

N_TREES = 300
LEAF = 2


def warn_if_features_are_stale():
    """Saved features must come from the current feature code."""
    saved = (ROOT / "profile_features.csv").stat().st_mtime
    sources = [ROOT.parent / name for name in ("entanglement_profile.py", "mps_probe.py", "visualize_qasm.py", "submission-harness/model.py")]
    newer = [p.name for p in sources if p.stat().st_mtime > saved + 60]
    if newer:
        print(f"WARNING: {', '.join(newer)} changed after profile_features.csv was made. If feature code changed, "
              "run python analysis/profile_features.py first.", flush=True)


def load_runs():
    """One row per (circuit, threshold), with each circuit's profile attached as a dictionary."""
    warn_if_features_are_stale()
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
    return minimize(loss, np.array(PHYSICS_START), method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-3, "fatol": 1e-5}).x


def trees(seed, n_jobs=4):
    return ExtraTreesRegressor(N_TREES, min_samples_leaf=LEAF, n_jobs=n_jobs, random_state=seed)


def fit_networks(X, target, base, seed=0, members=5):
    """Neural networks for the direct target and for the physics residual, five of each, averaged at prediction."""
    return {"direct": [make_network(1000 * seed + m).fit(X, target) for m in range(members)],
            "residual": [make_network(1000 * seed + 100 + m).fit(X, target - base) for m in range(members)]}


def fit_gate(profiles, feature_set):
    """Feature space and radius for deciding whether a new circuit resembles a training circuit."""
    from sklearn.neighbors import NearestNeighbors
    seen, rows = set(), []
    for p in profiles:
        if p["filename"] not in seen:
            seen.add(p["filename"])
            rows.append(feature_vector(p, 16, feature_set))
    space = make_gate_space().fit(np.array(rows))
    points = space.transform(np.array(rows))
    nearest = NearestNeighbors(n_neighbors=2).fit(points).kneighbors(points)[0][:, 1]
    return space, points, float(np.percentile(nearest, GATE_PERCENTILE))


def fit_bundle(runs, feature_set="probe", seed=0, timeout_target=TIMEOUT_TARGET, n_jobs=4, networks=True):
    profiles = list(runs.profile)
    log_runtime = np.log10(runs.runtime.values)
    timed_out = runs.timed_out.values
    target = np.where(timed_out, timeout_target, log_runtime)
    X = np.array([feature_vector(p, t, feature_set) for p, t in zip(profiles, runs.threshold)])
    arrays = physics_arrays(profiles, runs.threshold.values)
    params = fit_physics(arrays, log_runtime, timed_out)
    direct = trees(seed, n_jobs).fit(X, target)
    base = physics_log_seconds(params, arrays)
    residual = trees(seed, n_jobs).fit(X, target - base)
    bundle = {"direct": direct, "residual": residual, "physics_params": params, "feature_set": feature_set}
    if networks:
        space, points, radius = fit_gate(profiles, feature_set)
        bundle.update(networks=fit_networks(X, target, base, seed), known_families={gate_family(p) for p in profiles},
                      gate_space=space, gate_points=points, gate_radius=radius)
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude-list", help="File with one circuit filename per line to leave out of training.")
    ap.add_argument("--artifacts", default=str(ROOT.parent / "submission-harness" / "artifacts"))
    ap.add_argument("--no-probes", action="store_true", help="Train without the small simulations: structure only.")
    ap.add_argument("--no-network", action="store_true", help="Trees and physics only, without the gated neural network.")
    args = ap.parse_args()
    runs = load_runs()
    if args.exclude_list:
        excluded = {line.strip() for line in open(args.exclude_list) if line.strip()}
        runs = runs[~runs.filename.isin(excluded)].reset_index(drop=True)
    bundle = fit_bundle(runs, feature_set="all" if args.no_probes else "probe", networks=not args.no_network)
    out = Path(args.artifacts)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "runtime_model.joblib", compress=3)
    print(f"Trained on {runs.filename.nunique()} circuits, {len(runs)} runs")
    params = bundle["physics_params"]
    print({name: float(f"{(value if name == 'chi exponent' else np.exp(value)):.3g}") for name, value in zip(PHYSICS_NAMES, params)})
    print(f"Wrote {out / 'runtime_model.joblib'} ({(out / 'runtime_model.joblib').stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
