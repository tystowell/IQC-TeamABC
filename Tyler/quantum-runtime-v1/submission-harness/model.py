"""
model.py  --  predicts simulation runtime for the Quantum Rings challenge.

featurize  estimates how entanglement grows through the circuit from its structure (see
           entanglement_profile.py), then measures real singular values with a very small,
           budgeted tensor-network simulation (see mps_probe.py).
predict    averages two models in log space:
             1. Trees that predict runtime directly from the features.
             2. A fitted physics model plus trees that predict its error. The physics model
                adds up startup, per-gate, dynamic-circuit and entanglement costs.

Train with: python ../analysis/train_model.py  (writes artifacts/runtime_model.joblib)
"""

import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from entanglement_profile import FAMILY_NAMES, _index_gates, profile_text  # noqa: E402
from mps_probe import mps_features  # noqa: E402

CAP_SECONDS = 6 * 60 * 60  # 6-hour timeout cap
MIN_SECONDS = 0.15  # the fastest runs in the training data are about this long
BINS = 12
TIMEOUT_TARGET = math.log10(CAP_SECONDS) + 0.6  # training target for timeouts: the metric caps predictions at the limit

COUNT_COLUMNS = ["n_ops", "tracked_ops", "n_1q", "n_multi", "n_effective", "n_cancelled", "n_t", "n_measure", "n_reset",
                 "n_conditional", "n_gate_defs", "magic"]
PLAIN_COLUMNS = ["n_qubits", "coverage", "largest_group", "peak_est", "peak_power", "peak_stab", "peak_group", "peak_mean_est",
                 "peak_mean_power", "peak_mean_stab", "peak_mean_group", "final_max_est", "final_mean_est", "final_max_stab",
                 "final_mean_stab", "final_max_power", "final_mean_power"]
STRUCTURE_COUNTS = ["depth", "depth_2q", "n_pairs", "n_1q_clifford", "n_nearest_neighbor", "n_ops_on_measured"]
STRUCTURE_PLAIN = ["max_degree", "mean_span", "n_distinct_gates"]
SIGNATURE_COLUMNS = ["n_cx_rotation_cx", "n_phase_ladder", "n_cp_small", "n_toffoli_motif"]
PROBES = ((8, 50_000), (16, 40_000))  # (bond dimension, work budget), together about 4.5 s at most
PROBE_PLAIN = ["final_mean_entropy", "final_max_entropy", "peak_mean_entropy", "peak_max_entropy", "final_mean_rank", "final_max_rank",
               "frac_at_cap", "saturation", "truncated_share", "at_cap_share", "discarded_rate", "entropy_growth", "stopped_early"]
TOTAL_TIME_LIMIT = 13.5  # seconds for all of featurize; only reached on a machine much slower than expected


def circuit_features(qasm_text, use_probes=True):
    """Every feature for one circuit: the structural profile plus, optionally, one probe per bond dimension."""
    started = time.perf_counter()
    indexed = _index_gates(qasm_text)
    features = profile_text(qasm_text, indexed=indexed)
    features["n_skipped"] = sum(features.pop("skipped").values())
    for i, (chi, work) in enumerate(PROBES if use_probes else ()):
        remaining = TOTAL_TIME_LIMIT - (time.perf_counter() - started)
        share = remaining / (len(PROBES) - i)
        features.update(mps_features(qasm_text, chi=chi, max_work=work, time_limit=max(share, 0.05), prefix=f"mps{chi}", indexed=indexed))
    features["feature_seconds"] = time.perf_counter() - started
    return features


def is_dynamic(profile):
    return profile["n_reset"] + profile["n_conditional"] + profile["n_ops_on_measured"] > 0


def _costs(profile, threshold, prefix, powers):
    chi = np.minimum(2.0 ** np.arange(BINS), float(threshold))
    updates = np.array([profile[f"{prefix}_updates_{i}"] for i in range(BINS)], dtype=float)
    return [math.log10(1.0 + float((updates * chi ** p).sum())) for p in powers]


def feature_vector(profile, threshold, feature_set="probe"):
    """Turn one circuit's profile plus one threshold into model inputs."""
    row = [math.log2(threshold), float(is_dynamic(profile))]
    row += [math.log10(1.0 + profile[c]) for c in COUNT_COLUMNS]
    row += [float(profile[c]) for c in PLAIN_COLUMNS]
    for prefix in ("est", "stab", "power"):
        row += _costs(profile, threshold, prefix, (2, 3))
    row += [math.log10(1.0 + profile[f"est_updates_{i}"]) for i in range(BINS)]
    if feature_set in ("structure", "all", "probe"):
        n_ops = max(profile["n_ops"], 1.0)
        row += [math.log10(1.0 + profile[c]) for c in STRUCTURE_COUNTS]
        row += [float(profile[c]) for c in STRUCTURE_PLAIN]
        row += [profile[f"family_{family}"] / n_ops for family in FAMILY_NAMES]
    if feature_set in ("signatures", "all", "probe"):
        n_multi = max(profile["n_multi"], 1.0)
        row += [profile[c] / n_multi for c in SIGNATURE_COLUMNS]
        row += [math.log10(1.0 + profile[c]) for c in SIGNATURE_COLUMNS]
    if feature_set == "probe":
        n_ops = max(profile["n_ops"], 1.0)
        for chi, _ in PROBES:
            prefix = f"mps{chi}_"
            row += [float(profile[prefix + c]) for c in PROBE_PLAIN]
            row.append(math.log10(1.0 + profile[prefix + "discarded_total"]))
            row.append(math.log10(1.0 + profile[prefix + "updates"]))
            row.append(profile[prefix + "tracked_ops"] / n_ops)
            # How far this run's threshold sits above the entanglement the probe measured.
            row.append(math.log2(threshold) - profile[prefix + "peak_max_entropy"])
    return np.array(row, dtype=float)


def physics_arrays(profiles, thresholds):
    """Inputs for the physics model, as arrays with one entry per run."""
    ops = np.array([p["n_ops"] for p in profiles], dtype=float)
    dynamic = np.array([is_dynamic(p) for p in profiles], dtype=float)
    updates = np.array([[p[f"est_updates_{i}"] for i in range(BINS)] for p in profiles], dtype=float)
    return ops, dynamic, updates, np.asarray(thresholds, dtype=float)


def physics_log_seconds(params, arrays):
    """seconds = startup + per-op + extra per-op on dynamic circuits + entanglement cost at chi^p"""
    ops, dynamic, updates, thresholds = arrays
    startup, per_op, entanglement, per_dynamic_op = np.exp(params[:4])
    chi = np.minimum.outer(2.0 ** np.arange(BINS), thresholds).T
    seconds = startup + per_op * ops + per_dynamic_op * dynamic * ops + entanglement * (updates * chi ** params[4]).sum(axis=1)
    return np.log10(seconds)


class RuntimeModel:
    def __init__(self, artifacts_dir=HERE / "artifacts"):
        bundle = joblib.load(Path(artifacts_dir) / "runtime_model.joblib")
        self.direct = bundle["direct"]
        self.residual = bundle["residual"]
        self.params = bundle["physics_params"]
        self.feature_set = bundle["feature_set"]
        # Prediction runs one row at a time, so parallel workers would only add overhead.
        self.direct.n_jobs = self.residual.n_jobs = 1

    def featurize(self, qasm_text: str) -> dict:
        return circuit_features(qasm_text, use_probes=self.feature_set == "probe")

    def predict(self, features: dict, threshold: int) -> float:
        row = feature_vector(features, threshold, self.feature_set).reshape(1, -1)
        physics = physics_log_seconds(self.params, physics_arrays([features], [threshold]))[0]
        direct = self.direct.predict(row)[0]
        residual = self.residual.predict(row)[0] + physics
        return float(max(10.0 ** ((direct + residual) / 2), MIN_SECONDS))
