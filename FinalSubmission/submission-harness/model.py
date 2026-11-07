"""
model.py  --  predicts simulation runtime for the Quantum Rings challenge.

featurize  (circuit_features) estimates how entanglement grows through the circuit from its
           structure (entanglement_profile.py), measures real singular values with two very small,
           budgeted tensor-network simulations (mps_probe.py), and scans the text for how work is
           spread over qubits and how large rotation angles are.
predict    averages two models in log space:
             1. Trees that predict runtime directly from the features.
             2. A fitted physics formula plus trees that predict its error. The formula adds up
                startup, per-gate, dynamic-circuit, entanglement, shot-sampling and qubit-movement
                (swap) costs.
           If the circuit resembles the training data (same gate-set family, and close in feature
           space), one third of the prediction comes from a small neural network instead.
           Predictions are kept between 0.15 s and the 21,600 s timeout cap, and a rule-based check
           (likely_timeout) sends runs that will clearly time out straight to the cap.

Train with: python pipeline.py train  (writes artifacts/runtime_model.joblib). See pipeline.py for all steps.
"""

import collections
import math
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from entanglement_profile import FAMILY_NAMES, _index_gates, profile_text, signed_angle  # noqa: E402
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
PROBES = ((8, 25_000), (16, 20_000))  # (bond dimension, work budget); larger budgets did not improve the score
PROBE_PLAIN = ["final_mean_entropy", "final_max_entropy", "peak_mean_entropy", "peak_max_entropy", "final_mean_rank", "final_max_rank",
               "frac_at_cap", "saturation", "truncated_share", "at_cap_share", "discarded_rate", "entropy_growth", "stopped_early"]
TOTAL_TIME_LIMIT = 13.5  # seconds for all of featurize; only reached on a machine much slower than expected


def circuit_features(qasm_text, use_probes=True):
    """Every feature for one circuit: the structural profile plus, optionally, one probe per bond dimension."""
    started = time.perf_counter()
    indexed = _index_gates(qasm_text)
    features = profile_text(qasm_text, indexed=indexed)
    features["n_skipped"] = sum(features.pop("skipped").values())
    features.update(text_scan_features(qasm_text))
    for i, (chi, work) in enumerate(PROBES if use_probes else ()):
        remaining = TOTAL_TIME_LIMIT - (time.perf_counter() - started)
        share = remaining / (len(PROBES) - i)
        features.update(mps_features(qasm_text, chi=chi, max_work=work, time_limit=max(share, 0.05), prefix=f"mps{chi}", indexed=indexed))
    features["feature_seconds"] = time.perf_counter() - started
    return features


TEXT_SCAN_CHARS = 2_000_000  # long circuits are sampled from the start, which keeps this well under a second
_ROTATION = re.compile(r"^\s*(rx|ry|rz|p|u1|phase|cp|cu1|crz|crx|cry|rzz|rxx|ryy|u3|u|U|u2)\s*\(([^)]*)\)", re.MULTILINE)
_TWO_QUBIT_ROTATIONS = {"cp", "cu1", "crz", "crx", "cry", "rzz", "rxx", "ryy"}
_INDEX = re.compile(r"\w+\[(\d+)\]")


def text_scan_features(qasm_text):
    """
    Quick scan of the text. Found by studying long circuits the model misjudged:
    - when most operations land on a few qubits, the model tends to predict too slow;
    - larger rotation angles build entanglement faster, so runtime grows more with threshold.
    """
    text = qasm_text[:TEXT_SCAN_CHARS]
    load = collections.Counter(int(q) for q in _INDEX.findall(text))
    counts = np.array(sorted(load.values(), reverse=True), dtype=float) if load else np.ones(1)
    angles, two_qubit = [], []
    for name, params in _ROTATION.findall(text):
        for piece in params.split(","):
            value = signed_angle(piece.strip())
            if value is None:
                continue
            value = abs((value + math.pi) % (2 * math.pi) - math.pi)
            angles.append(value)
            if name in _TWO_QUBIT_ROTATIONS:
                two_qubit.append(value)
    return {"text_qubit_top10_share": float(counts[:max(1, len(counts) // 10)].sum() / counts.sum()),
            "text_qubit_load_cv": float(counts.std() / counts.mean()),
            "text_angle_median": float(np.median(angles)) if angles else 0.0,
            "text_angle2_mean": float(np.mean(two_qubit)) if two_qubit else -1.0}


TEXT_COLUMNS = ["text_qubit_top10_share", "text_qubit_load_cv", "text_angle_median", "text_angle2_mean"]


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
    if feature_set in ("all", "probe"):
        row += sampling_features(profile, threshold)
        row += [float(profile[c]) for c in TEXT_COLUMNS]
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


def final_state_entangled(profile):
    """0 to 1: is the final state entangled? From the chi=16 probe, or the structural estimate without probes."""
    value = profile.get("mps16_final_max_entropy")
    return min(max(float(profile["final_max_est"] if value is None else value), 0.0), 1.0)


def probe_entropy(profile):
    """Largest entanglement (bits) the chi=16 probe measured, or the structural estimate (capped at 4) without probes."""
    value = profile.get("mps16_peak_max_entropy")
    return max(float(min(profile["peak_est"], 4.0) if value is None else value), 0.0)


def total_swap_distance(profile):
    """How far long-range gates must move qubits, summed over the whole circuit."""
    return profile["n_multi"] * profile["mean_span"]


def sampling_features(profile, threshold):
    """Shot sampling, qubit movement and the largest bond the qubit count allows. Measured on the training data:
    drawing 1024 shots costs about 7 ms per measured qubit, or about 35 ms once the final state is entangled."""
    measured, span, n_qubits = profile["n_measure"], total_swap_distance(profile), profile["n_qubits"]
    sample_seconds = measured * 0.007 * (1 + 4 * final_state_entangled(profile))
    largest_bond = 2.0 ** math.floor(n_qubits / 2)
    return [math.log10(1.0 + measured), math.log10(0.17 + sample_seconds), math.log10(1.0 + span), math.log10(1.0 + span * n_qubits),
            span / max(profile["n_ops"], 1.0), math.log2(min(threshold, largest_bond))]


def physics_arrays(profiles, thresholds):
    """Inputs for the physics model, as arrays with one entry per run."""
    ops = np.array([p["n_ops"] for p in profiles], dtype=float)
    dynamic = np.array([is_dynamic(p) for p in profiles], dtype=float)
    updates = np.array([[p[f"est_updates_{i}"] for i in range(BINS)] for p in profiles], dtype=float)
    measured = np.array([p["n_measure"] for p in profiles], dtype=float)
    entangled = np.array([final_state_entangled(p) for p in profiles], dtype=float)
    span = np.array([total_swap_distance(p) for p in profiles], dtype=float)
    entropy = np.array([probe_entropy(p) for p in profiles], dtype=float)
    return ops, dynamic, updates, np.asarray(thresholds, dtype=float), measured, entangled, span, entropy


def physics_log_seconds(params, arrays):
    """
    seconds = startup + per-op + extra per-op on dynamic circuits (growing with measured entanglement)
              + entanglement cost at chi^p
              + shot sampling per measured qubit (more if the final state is entangled) + per swap for long-range gates
    Dynamic circuits are simulated once per shot, so their per-op cost grows with how entangled the state is.
    """
    ops, dynamic, updates, thresholds, measured, entangled, span, entropy = arrays
    startup, per_op, entanglement, per_dynamic_op = np.exp(params[:4])
    per_measured, entangled_factor, per_swap, dynamic_per_bit = np.exp(params[5:9])
    chi = np.minimum.outer(2.0 ** np.arange(BINS), thresholds).T
    seconds = (startup + per_op * ops + per_dynamic_op * dynamic * ops * (1 + dynamic_per_bit * entropy)
               + entanglement * (updates * chi ** params[4]).sum(axis=1)
               + per_measured * measured * (1 + entangled_factor * entangled) + per_swap * span)
    return np.log10(seconds)


PHYSICS_START = [math.log(0.15), math.log(1e-5), math.log(1e-8), math.log(1e-2), 2.0, math.log(7e-3), math.log(4.0), math.log(1e-6), 0.0]
PHYSICS_NAMES = ["startup s", "s per op", "entanglement constant", "extra s per dynamic op", "chi exponent",
                 "s per measured qubit", "entangled sampling factor", "s per swap", "dynamic extra per entanglement bit"]


NETWORK_WEIGHT = 1 / 3  # share of the neural network in the blend, used only for familiar circuits
GATE_PERCENTILE = 90  # "familiar" = closer to a training circuit than 90% of training circuits are to their nearest neighbour


def gate_family(profile):
    """The circuit's gate-set family, e.g. 'cx+rz+rx'. Used to decide whether a circuit resembles the training data."""
    return "+".join(f for f in FAMILY_NAMES if f not in ("measure", "h") and profile[f"family_{f}"] > 0)


def make_network(seed):
    """A small neural network on rank-transformed features. Five of these are averaged."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    net = MLPRegressor(hidden_layer_sizes=(256, 128, 64), alpha=0.1, learning_rate_init=1e-3, max_iter=800, early_stopping=True,
                       validation_fraction=0.12, n_iter_no_change=40, random_state=seed)
    return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0), StandardScaler(), net)


def make_gate_space():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0), StandardScaler())


def likely_timeout(profile, threshold):
    """
    A teammate's rule for runs that will hit the 6-hour cap. It is applied after the model, and sets those runs to the cap.
    On the training data it flags 51 runs: 48 real timeouts, plus 3 that finished between 4,400 and 17,600 s.
    Branch 1 matches our own finding that dynamic circuits rerun every shot; branch 3 catches long, dense, all-to-all
    rotation circuits. (Branch 2 matches no training circuit.)
    """
    qubits = max(profile["n_qubits"], 1.0)
    # Gate 0: circuits where under half the qubits are ever linked by two-qubit gates don't time out.
    if min(profile["largest_group"] / qubits, 1.0) < 0.5:
        return False
    density = profile["n_pairs"] / max(qubits * (qubits - 1) / 2, 1.0)
    gates_per_qubit = max(profile["n_ops"], 1.0) / qubits
    if profile["n_reset"] > 0 and qubits >= 20:
        start = 16
    elif (qubits >= 100 and profile["max_degree"] >= 60 and profile["n_nearest_neighbor"] / max(profile["n_multi"], 1.0) < 0.03
          and profile["depth"] >= 2000 and density >= 0.9):
        start = 512
    elif qubits >= 32 and density >= 0.95 and profile["n_multi"] / max(profile["n_pairs"], 1.0) >= 35 and gates_per_qubit >= 2300:
        start = 256 if gates_per_qubit >= 12000 else 512 if gates_per_qubit >= 4000 else 1024
    else:
        return False
    return threshold >= start


class RuntimeModel:
    def __init__(self, artifacts_dir=HERE / "artifacts"):
        bundle = joblib.load(Path(artifacts_dir) / "runtime_model.joblib")
        self.direct = bundle["direct"]
        self.residual = bundle["residual"]
        self.params = bundle["physics_params"]
        self.feature_set = bundle["feature_set"]
        # Optional neural network, blended in only for circuits that resemble the training data.
        self.networks = bundle.get("networks")
        self.known_families = bundle.get("known_families", set())
        self.gate_space, self.gate_points, self.gate_radius = bundle.get("gate_space"), bundle.get("gate_points"), bundle.get("gate_radius")
        # Prediction runs one row at a time, so parallel workers would only add overhead.
        self.direct.n_jobs = self.residual.n_jobs = 1

    def featurize(self, qasm_text: str) -> dict:
        return circuit_features(qasm_text, use_probes=self.feature_set == "probe")

    def is_familiar(self, features):
        """Same gate-set family as some training circuit, and close to one in feature space."""
        if self.networks is None or gate_family(features) not in self.known_families:
            return False
        point = self.gate_space.transform(feature_vector(features, 16, self.feature_set).reshape(1, -1))
        return float(np.sqrt(((self.gate_points - point) ** 2).sum(axis=1)).min()) <= self.gate_radius

    def predict(self, features: dict, threshold: int) -> float:
        if likely_timeout(features, threshold):
            return float(CAP_SECONDS)
        row = feature_vector(features, threshold, self.feature_set).reshape(1, -1)
        physics = physics_log_seconds(self.params, physics_arrays([features], [threshold]))[0]
        log_seconds = (self.direct.predict(row)[0] + self.residual.predict(row)[0] + physics) / 2
        if self.is_familiar(features):
            direct = np.mean([net.predict(row)[0] for net in self.networks["direct"]])
            residual = np.mean([net.predict(row)[0] for net in self.networks["residual"]]) + physics
            log_seconds = (1 - NETWORK_WEIGHT) * log_seconds + NETWORK_WEIGHT * (direct + residual) / 2
        # A run that times out is scored as exactly the cap, and a run that finishes is never slower than the cap,
        # so a prediction above the cap can only lose points.
        return float(min(max(10.0 ** log_seconds, MIN_SECONDS), CAP_SECONDS))
