#!/usr/bin/env python3
"""
Add one change at a time and measure its effect on the official metric. Three tests:
  circuits   random circuits held out
  similar    circuits with similar size held out together
  families   circuits with the same gate set held out together, a rough stand-in for new algorithms

Usage: python analysis/evaluate_model.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "submission-harness"))

from entanglement_profile import FAMILY_NAMES  # noqa: E402
from model import CAP_SECONDS, MIN_SECONDS, TIMEOUT_TARGET, feature_vector, physics_arrays, physics_log_seconds  # noqa: E402
from train_model import fit_physics, load_runs, trees  # noqa: E402

runs = load_runs()
log_runtime = np.log10(runs.runtime.values)
timed_out = runs.timed_out.values
profiles = list(runs.profile)
thresholds = runs.threshold.values
arrays = physics_arrays(profiles, thresholds)

runs["similar"] = [f"{p['n_qubits']}_{round(np.log2(max(p['n_ops'], 1)))}_{round(np.log2(1 + p['n_multi']))}" for p in profiles]
runs["family"] = ["+".join(f for f in FAMILY_NAMES if f not in ("measure", "h") and p[f"family_{f}"] > 0) for p in profiles]
GROUPINGS = {"circuits": "filename", "similar": "similar", "families": "family"}
print(f"circuits {runs.filename.nunique()}, similar groups {runs.similar.nunique()}, gate-set families {runs.family.nunique()}")
sizes = runs.drop_duplicates("filename").family.value_counts()
print("largest gate-set families:", sizes.head(6).to_dict())

MATRICES = {name: np.array([feature_vector(p, t, name) for p, t in zip(profiles, thresholds)])
            for name in ("base", "structure", "signatures", "all", "probe")}


def score(pred_log):
    pred = 10.0 ** pred_log
    pred = np.where(timed_out, np.minimum(pred, CAP_SECONDS), pred)
    return np.maximum(0, 1 - np.abs(np.log10(pred / runs.runtime.values)))


def subset(arr, index):
    ops, dynamic, updates, thr = arr
    return ops[index], dynamic[index], updates[index], thr[index]


def run(feature_set="base", average=True, timeout_target=None, clip=False, seeds=range(2), groupings=GROUPINGS):
    X = MATRICES[feature_set]
    target = np.where(timed_out, TIMEOUT_TARGET if timeout_target is None else timeout_target, log_runtime)
    if timeout_target == 0:
        target = log_runtime
    results = {}
    for label, column in groupings.items():
        seed_scores = []
        for seed in seeds:
            predictions = np.zeros(len(runs))
            for train, test in GroupKFold(5, shuffle=True, random_state=seed).split(X, target, runs[column]):
                direct = trees(seed).fit(X[train], target[train]).predict(X[test])
                if average:
                    params = fit_physics(subset(arrays, train), log_runtime[train], timed_out[train])
                    base_train = physics_log_seconds(params, subset(arrays, train))
                    base_test = physics_log_seconds(params, subset(arrays, test))
                    residual = trees(seed).fit(X[train], target[train] - base_train).predict(X[test]) + base_test
                    predictions[test] = (direct + residual) / 2
                else:
                    predictions[test] = direct
            if clip:
                predictions = np.maximum(predictions, np.log10(MIN_SECONDS))
            seed_scores.append(score(predictions).mean())
        results[label] = (np.mean(seed_scores), np.std(seed_scores))
    return results


STEPS = [
    ("Direct trees, base features", dict(feature_set="base", average=False, timeout_target=0)),
    ("+ average with physics + correction", dict(feature_set="base", average=True, timeout_target=0)),
    ("+ timeouts trained above the cap", dict(feature_set="base", average=True)),
    ("+ minimum prediction of 0.15 s", dict(feature_set="base", average=True, clip=True)),
    ("+ gate mix and structure features", dict(feature_set="structure", average=True, clip=True)),
    ("+ block signatures instead of structure", dict(feature_set="signatures", average=True, clip=True)),
    ("+ structure and block signatures", dict(feature_set="all", average=True, clip=True)),
    ("+ MPS probes at chi=8 and 16", dict(feature_set="probe", average=True, clip=True)),
]
print(f"\n{'':44s}{'circuits':>16s}{'similar':>16s}{'families':>16s}")
for label, options in STEPS:
    results = run(**options)
    print(f"{label:44s}" + "".join(f"{results[k][0]:10.1%} ±{results[k][1]:.1%}" for k in GROUPINGS), flush=True)

print("\nTimeout training target above the cap, in log10 units (similar circuits):")
for over in (0.2, 0.6, 1.3):
    results = run(feature_set="probe", clip=True, timeout_target=np.log10(CAP_SECONDS) + over, groupings={"similar": "similar"})
    print(f"  +{over}: {results['similar'][0]:.1%}", flush=True)
