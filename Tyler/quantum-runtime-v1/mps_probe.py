#!/usr/bin/env python3
"""
mps_probe.py  --  run a circuit at a very small bond dimension and measure its real entanglement.

This is a cheap simulation, not a prediction of the result: chi is tiny and fidelity is ignored.
What it provides that structural analysis can't is real singular values, which show whether a
circuit's entanglement actually grows, levels off, or shrinks.

Uses time-evolving block decimation on a qubit chain. A long-range gate moves one qubit next to the
other with swaps, applies the gate, and swaps back. Runs stop after a fixed amount of work, charged
by each update's cost, so results don't depend on machine speed. A time limit is only a safety net.
Measurement and reset are skipped, so a circuit with mid-circuit operations can be overestimated.
"""

import math
import time

import numpy as np

from entanglement_profile import _index_gates, _LazyGates, _lines_with_position, signed_angle
from visualize_qasm import _Parser

TOLERANCE = 1e-5  # singular values this far below the largest are dropped
FLOOR = 1e-7  # keeps the division by small singular values stable
TIME_LIMIT = 8.0
MAX_WORK = 75_000  # about 3 s on the machine this was calibrated on; a low-rank update costs 1 unit

_I = np.eye(2, dtype=np.complex64)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex64) / math.sqrt(2)
_FIXED_1Q = {
    "id": _I, "x": _X, "y": _Y, "z": _Z, "h": _H,
    "s": np.diag([1, 1j]).astype(np.complex64), "sdg": np.diag([1, -1j]).astype(np.complex64),
    "t": np.diag([1, np.exp(1j * math.pi / 4)]).astype(np.complex64),
    "tdg": np.diag([1, np.exp(-1j * math.pi / 4)]).astype(np.complex64),
    "sx": np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex64) / 2,
    "sxdg": np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=np.complex64) / 2,
}
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex64)


def _angles(params):
    values = []
    for text in params.split(","):
        value = signed_angle(text.strip())
        values.append(1.0 if value is None else value)
    return values


def _rx(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex64)


def _ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex64)


def _rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)]).astype(np.complex64)


def _phase(t):
    return np.diag([1, np.exp(1j * t)]).astype(np.complex64)


def _u3(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=np.complex64)


def _one_qubit_matrix(name, params):
    if name in _FIXED_1Q:
        return _FIXED_1Q[name]
    a = _angles(params) if params else [0.0]
    if name == "rx":
        return _rx(a[0])
    if name == "ry":
        return _ry(a[0])
    if name == "rz":
        return _rz(a[0])
    if name in ("p", "u1"):
        return _phase(a[0])
    if name == "u2" and len(a) >= 2:
        return _u3(math.pi / 2, a[0], a[1])
    if name in ("u3", "u", "U") and len(a) >= 3:
        return _u3(a[0], a[1], a[2])
    return None


def _controlled(u):
    gate = np.eye(4, dtype=np.complex64)
    gate[2:, 2:] = u
    return gate


def _two_qubit_matrix(name, params):
    """4x4 matrix on (first qubit, second qubit), with the first qubit as the most significant bit."""
    if name in ("cx", "CX"):
        return _controlled(_X)
    if name == "cz":
        return _controlled(_Z)
    if name == "cy":
        return _controlled(_Y)
    if name == "ch":
        return _controlled(_H)
    if name == "csx":
        return _controlled(_FIXED_1Q["sx"])
    if name == "swap":
        return _SWAP
    a = _angles(params) if params else [0.0]
    if name in ("cp", "cu1"):
        return _controlled(_phase(a[0]))
    if name == "crz":
        return _controlled(_rz(a[0]))
    if name == "crx":
        return _controlled(_rx(a[0]))
    if name == "cry":
        return _controlled(_ry(a[0]))
    if name in ("cu3", "cu") and len(a) >= 3:
        return _controlled(_u3(a[0], a[1], a[2]))
    if name in ("rzz", "rxx", "ryy"):
        pauli = {"rzz": _Z, "rxx": _X, "ryy": _Y}[name]
        kron = np.kron(pauli, pauli)
        c, s = math.cos(a[0] / 2), math.sin(a[0] / 2)
        return (c * np.eye(4) - 1j * s * kron).astype(np.complex64)
    return None


def _toffoli_sequence(a, b, c):
    return [("h", "", (c,)), ("cx", "", (b, c)), ("tdg", "", (c,)), ("cx", "", (a, c)), ("t", "", (c,)), ("cx", "", (b, c)),
            ("tdg", "", (c,)), ("cx", "", (a, c)), ("t", "", (b,)), ("t", "", (c,)), ("h", "", (c,)), ("cx", "", (a, b)),
            ("t", "", (a,)), ("tdg", "", (b,)), ("cx", "", (a, b))]


class _Chain:
    def __init__(self, n, chi):
        self.n = n
        self.chi = chi
        self.gammas = [np.array([[[1.0], [0.0]]], dtype=np.complex64) for _ in range(n)]
        self.lambdas = [np.ones(1, dtype=np.float32) for _ in range(n + 1)]
        self.pending = [None] * n
        self.updates = 0
        self.work = 0.0
        self.discarded = 0.0
        self.truncated_updates = 0
        self.at_cap_updates = 0
        self.entropy_growth = 0.0

    def one_qubit(self, q, matrix):
        pending = self.pending[q]
        self.pending[q] = matrix if pending is None else matrix @ pending

    def _absorb(self, gate, q):
        """Combine single-qubit gates waiting on sites q and q+1 into a 4x4 gate."""
        left, right = self.pending[q], self.pending[q + 1]
        if left is None and right is None:
            return gate
        a = _I if left is None else left
        b = _I if right is None else right
        self.pending[q] = self.pending[q + 1] = None
        return gate @ np.kron(a, b)

    def adjacent(self, q, gate):
        """Apply a 4x4 gate to sites q and q+1 with q as the most significant bit."""
        gate = self._absorb(gate, q)
        lam_l, lam_m, lam_r = self.lambdas[q], self.lambdas[q + 1], self.lambdas[q + 2]
        a = self.gammas[q] * lam_l[:, None, None] * lam_m[None, None, :]
        b = self.gammas[q + 1] * lam_r[None, None, :]
        left_dim, right_dim = a.shape[0], b.shape[2]
        theta = np.tensordot(a, b, axes=(2, 0)).reshape(left_dim, 4, right_dim)
        theta = np.tensordot(gate, theta, axes=(1, 1)).transpose(1, 0, 2).reshape(left_dim * 2, right_dim * 2)
        try:
            u, s, vh = np.linalg.svd(theta, full_matrices=False)
        except np.linalg.LinAlgError:
            return
        self.updates += 1
        self.work += 1.0 + 6.0 * (min(len(s), self.chi) / 16.0) ** 3
        total = float((s * s).sum())
        if total <= 0:
            return
        keep = int((s > TOLERANCE * s[0]).sum())
        if keep >= self.chi:
            self.at_cap_updates += 1
            if keep > self.chi:
                self.truncated_updates += 1
            keep = self.chi
        kept = float((s[:keep] ** 2).sum())
        self.discarded += max(0.0, 1.0 - kept / total)
        s = s[:keep] / math.sqrt(kept)
        self.lambdas[q + 1] = s.astype(np.float32)
        self.gammas[q] = (u[:, :keep].reshape(left_dim, 2, keep)) / np.maximum(lam_l, FLOOR)[:, None, None]
        self.gammas[q + 1] = (vh[:keep].reshape(keep, 2, right_dim)) / np.maximum(lam_r, FLOOR)[None, None, :]

    def two_qubit(self, a, b, gate):
        """Apply a 4x4 gate on chain positions a and b, moving a next to b with swaps if needed."""
        if a > b:
            order = np.array([0, 2, 1, 3])
            gate = gate[order][:, order]
            a, b = b, a
        path = []
        for q in range(a, b - 1):
            self.adjacent(q, _SWAP)
            path.append(q)
        self.adjacent(b - 1, gate)
        for q in reversed(path):
            self.adjacent(q, _SWAP)

    def entropies(self):
        out = []
        for lam in self.lambdas[1:self.n]:
            p = lam.astype(np.float64) ** 2
            p = p[p > 1e-14]
            out.append(float(-(p * np.log2(p)).sum()))
        return out

    def ranks(self):
        return [len(lam) for lam in self.lambdas[1:self.n]]


def mps_features(text, chi=8, max_work=MAX_WORK, time_limit=TIME_LIMIT, prefix="mps", indexed=None):
    """Simulate at bond dimension chi and summarize real entanglement. See the module docstring."""
    started = time.perf_counter()
    deadline = started + time_limit
    spans, main = indexed if indexed is not None else _index_gates(text)
    parser = _Parser(expand_gates=True)
    parser.gates = _LazyGates(text, spans, parser)
    chain = None
    position = [0]
    samples = []  # (updates done, mean entropy, max entropy) at intervals
    next_sample = 64
    n_ops = n_unknown = 0
    stopped_early = hit_time_limit = False
    peak_mean = peak_max = 0.0
    one_cache, two_cache = {}, {}

    def do_two(a, b, name, params):
        key = (name, params)
        matrix = two_cache.get(key)
        if matrix is None:
            matrix = _two_qubit_matrix(name, params)
            two_cache[key] = matrix if matrix is not None else False
        if matrix is False:
            return False
        chain.two_qubit(a, b, matrix)
        return True

    def do_one(q, name, params):
        key = (name, params)
        matrix = one_cache.get(key)
        if matrix is None:
            matrix = _one_qubit_matrix(name, params)
            one_cache[key] = matrix if matrix is not None else False
        if matrix is False:
            return False
        chain.one_qubit(q, matrix)
        return True

    for name, params, qubits, conditional in parser.ops(_lines_with_position(main, position)):
        if name in ("barrier", "measure", "reset"):
            continue
        if chain is None or chain.n != parser.n_qubits:
            if chain is None:
                chain = _Chain(max(parser.n_qubits, 2), chi)
        n_ops += 1
        width = len(qubits)
        if width == 1:
            if not do_one(qubits[0], name, params):
                n_unknown += 1
        elif width == 2:
            if not do_two(qubits[0], qubits[1], name, params):
                n_unknown += 1
        elif name in ("ccx", "rccx") and width == 3:
            for sub_name, sub_params, sub_qubits in _toffoli_sequence(*qubits):
                if len(sub_qubits) == 1:
                    do_one(sub_qubits[0], sub_name, sub_params)
                else:
                    do_two(sub_qubits[0], sub_qubits[1], sub_name, sub_params)
        elif name == "cswap" and width == 3:
            control, a, b = qubits
            do_two(b, a, "cx", "")
            for sub_name, sub_params, sub_qubits in _toffoli_sequence(control, a, b):
                if len(sub_qubits) == 1:
                    do_one(sub_qubits[0], sub_name, sub_params)
                else:
                    do_two(sub_qubits[0], sub_qubits[1], sub_name, sub_params)
            do_two(b, a, "cx", "")
        else:
            n_unknown += 1
            for q in qubits[1:]:
                do_two(qubits[0], q, "cx", "")
        if chain.updates >= next_sample:
            values = chain.entropies()
            mean_entropy, max_entropy = sum(values) / max(len(values), 1), max(values, default=0.0)
            samples.append((chain.updates, mean_entropy, max_entropy))
            peak_mean, peak_max = max(peak_mean, mean_entropy), max(peak_max, max_entropy)
            next_sample = chain.updates + max(64, chain.updates // 8)
        if chain.work >= max_work:
            stopped_early = True
            break
        if n_ops % 256 == 0 and time.perf_counter() > deadline:
            stopped_early = hit_time_limit = True
            break

    if chain is None:
        chain = _Chain(2, chi)
    values = chain.entropies()
    final_mean, final_max = sum(values) / max(len(values), 1), max(values, default=0.0)
    peak_mean, peak_max = max(peak_mean, final_mean), max(peak_max, final_max)
    ranks = chain.ranks()
    updates = max(chain.updates, 1)
    limit = math.log2(chi)
    # Growth over the last half of the run: is entanglement still climbing, or has it levelled off?
    midpoint = next((s for s in samples if s[0] >= chain.updates / 2), None)
    growth = final_mean - midpoint[1] if midpoint else 0.0
    return {
        f"{prefix}_final_mean_entropy": final_mean,
        f"{prefix}_final_max_entropy": final_max,
        f"{prefix}_peak_mean_entropy": peak_mean,
        f"{prefix}_peak_max_entropy": peak_max,
        f"{prefix}_final_mean_rank": sum(ranks) / max(len(ranks), 1),
        f"{prefix}_final_max_rank": max(ranks, default=1),
        f"{prefix}_frac_at_cap": sum(r >= chi for r in ranks) / max(len(ranks), 1),
        f"{prefix}_saturation": final_max / limit if limit else 0.0,
        f"{prefix}_truncated_share": chain.truncated_updates / updates,
        f"{prefix}_at_cap_share": chain.at_cap_updates / updates,
        f"{prefix}_discarded_total": chain.discarded,
        f"{prefix}_discarded_rate": chain.discarded / updates,
        f"{prefix}_entropy_growth": growth,
        f"{prefix}_updates": chain.updates,
        f"{prefix}_tracked_ops": n_ops,
        f"{prefix}_stopped_early": int(stopped_early),
        f"{prefix}_hit_time_limit": int(hit_time_limit),
        f"{prefix}_unknown_ops": n_unknown,
        f"{prefix}_seconds": time.perf_counter() - started,
    }
