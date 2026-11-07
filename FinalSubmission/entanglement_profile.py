#!/usr/bin/env python3
"""
entanglement_profile.py  --  estimate how a tensor-network state grows, without simulating it.

Treats the qubits as a chain and tracks, for every cut between neighbors, an estimate of
log2(bond dimension) as the circuit runs. Each estimate combines three views:

  structural   each gate adds its entangling power to the cuts it crosses. Gates that a
               product-state tracker proves can't entangle add nothing, and a gate that
               exactly undoes the previous gate on its qubits removes what that gate added.
  groups       qubits only count as linked once an entangling gate joins them. Measuring or
               resetting a qubit removes it from its group.
  stabilizer   a tableau of the circuit with each non-Clifford gate replaced by a similar
               Clifford. Unlike the others this can fall as a circuit undoes its own work.
               Exact for Clifford circuits, but it can badly underestimate the rest, because
               a replaced gate can make a real entangling sequence cancel itself.

The combined estimate at each cut is the lowest of the structural value, the group value, and
the stabilizer value plus a budget for the non-Clifford content so far. That budget is an
entropy-weighted count, so the combined value is an estimate rather than a strict bound.
Nothing here computes actual singular values.

Usage:
    from entanglement_profile import entanglement_profile, plot_entanglement_profile
    profile = entanglement_profile("train-circuits/0c0ce4a1.qasm.zst")
    plot_entanglement_profile("train-circuits/0c0ce4a1.qasm.zst", out="profile.png")

    python entanglement_profile.py train-circuits/0c0ce4a1.qasm.zst --out profile.png

Very large circuits stop after a fixed amount of work, so results don't depend on machine speed.
The remainder is counted without being tracked and extrapolated, and "coverage" reports the
fraction of operations actually tracked. A time limit remains only as a safety net.
"""

import argparse
import collections
import math
import re
import time
from pathlib import Path

from visualize_qasm import _GateDef, _human, _open_lines, _Parser, _statements

THRESHOLDS = (16, 64, 256, 512, 1024)
BOND_BINS = 12  # estimates of 11 bits or more share a bin; 2^11 exceeds every threshold
SMALL_ANGLE = 0.1  # radians; a smaller rotation is treated as the identity in the tableau
STACK_DEPTH = 32  # undo history kept per qubit
MAX_WORK = 600_000  # tracking stops here: one unit is about one operation, and 4 cuts crossed cost one more
TIME_LIMIT = 9.0  # seconds; a safety net for slow machines that normally never triggers

# Gate families for gate-mix features, so similar gates from different benchmark suites match.
GATE_FAMILIES = {
    "h": "h", "x": "pauli", "y": "pauli", "z": "pauli", "id": "pauli", "s": "s", "sdg": "s", "t": "t", "tdg": "t",
    "sx": "sx", "sxdg": "sx", "rx": "rx", "ry": "ry", "rz": "rz", "p": "rz", "u1": "rz", "u2": "u", "u3": "u", "u": "u",
    "U": "u", "cx": "cx", "CX": "cx", "cy": "cx", "cz": "cz", "swap": "swap", "cp": "cp", "cu1": "cp", "crz": "cp",
    "crx": "crx", "cry": "crx", "ch": "crx", "csx": "crx", "cu": "crx", "cu3": "crx", "rzz": "rzz", "rxx": "rzz",
    "ryy": "rzz", "ccx": "ccx", "rccx": "ccx", "cswap": "cswap", "measure": "measure", "reset": "reset",
}
FAMILY_NAMES = sorted(set(GATE_FAMILIES.values())) + ["other"]
_CLIFFORD_1Q = {"h", "x", "y", "z", "id", "s", "sdg", "sx", "sxdg"}
_DIAGONAL_ANGLE = {"t": "pi/4", "tdg": "-pi/4", "s": "pi/2", "sdg": "-pi/2"}
_TOFFOLI_TARGET = ("h", "c", "T", "c", "t", "c", "T", "c", "t", "h")
_TOFFOLI_TOKENS = {"h": "h", "t": "t", "tdg": "T"}

_NUMERIC = re.compile(r"[0-9eE.+\-*/() pi]*$")
_GATE_BLOCK = re.compile(r"^[ \t]*gate\s+(\w+)\s*(?:\(([^)]*)\))?\s*([^{]*)\{([^}]*)\}", re.MULTILINE)
_LEADING_WORD = re.compile(r"^[ \t]*([A-Za-z_]\w*)", re.MULTILINE)
_NOT_OPERATIONS = {"OPENQASM", "include", "qreg", "creg", "qubit", "bit", "gate", "barrier", "if",
                   "else", "int", "uint", "float", "angle", "bool", "const", "opaque", "pragma"}

ANGLE_2Q = {"cp", "cu1", "crx", "cry", "crz", "rzz", "rxx", "ryy"}
_ANGLE_GATES = ANGLE_2Q | {"rz", "rx", "ry", "p", "u1"}
_SELF_INVERSE = {"h", "x", "y", "z", "id", "cx", "CX", "cy", "cz", "ch", "swap", "ccx", "cswap"}
_INVERSE_NAME = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t", "sx": "sxdg", "sxdg": "sx"}

_signed_angles = {}


def signed_angle(params):
    """Value of a gate's first angle in radians. None if it can't be evaluated."""
    if params in _signed_angles:
        return _signed_angles[params]
    value = None
    text = params.split(",")[0]
    if text and _NUMERIC.match(text):
        try:
            # The regex limits this to arithmetic on numbers and pi.
            value = float(eval(text, {"__builtins__": {}}, {"pi": math.pi}))  # noqa: S307
        except (NameError, SyntaxError, ZeroDivisionError, TypeError):
            pass
    _signed_angles[params] = value
    return value


def angle_from_zero(params):
    """Distance of a gate's first angle from 0 mod 2 pi. None if it can't be evaluated."""
    theta = signed_angle(params)
    if theta is None:
        return None
    value = abs((theta + math.pi) % (2 * math.pi) - math.pi)
    # Rounding grows with the angle, and angles like 268435456*pi appear in these files.
    return 0.0 if value < 1e-9 + abs(theta) * 1e-15 else value


def entangling_power(name, params):
    """Entanglement a gate can add across a cut, in bits: the entropy of its operator-Schmidt weights."""
    if name == "swap":
        return 2.0
    if name not in ANGLE_2Q:
        return 1.0
    angle = angle_from_zero(params)
    if angle is None:
        return 1.0
    weight = math.sin(angle / (2 if name in ("rzz", "rxx", "ryy") else 4)) ** 2
    if weight <= 0 or weight >= 1:
        return 0.0
    return -weight * math.log2(weight) - (1 - weight) * math.log2(1 - weight)


# ---------------------------------------------------------------------- #
# Product-state tracker                                                  #
# ---------------------------------------------------------------------- #
# Single-qubit states the tracker can follow exactly while a qubit is unentangled.
Z0, Z1, X_PLUS, X_MINUS, GENERAL = range(5)
_FIXED_1Q = {
    "x": {Z0: Z1, Z1: Z0, X_PLUS: X_PLUS, X_MINUS: X_MINUS},
    "y": {Z0: Z1, Z1: Z0, X_PLUS: X_MINUS, X_MINUS: X_PLUS},
    "z": {Z0: Z0, Z1: Z1, X_PLUS: X_MINUS, X_MINUS: X_PLUS},
    "h": {Z0: X_PLUS, Z1: X_MINUS, X_PLUS: Z0, X_MINUS: Z1},
    "id": {Z0: Z0, Z1: Z1, X_PLUS: X_PLUS, X_MINUS: X_MINUS},
}
_DIAGONAL_1Q = {"s", "sdg", "t", "tdg", "rz", "p", "u1"}
_X_ROTATION_1Q = {"sx", "sxdg", "rx"}
_DIAGONAL_2Q = {"cz", "cp", "cu1", "crz", "rzz"}
_CONTROLLED_2Q = {"cy", "ch", "csx", "crx", "cry", "cu", "cu3"}
_IS_Z = (Z0, Z1)


def _apply_1q(states, qubit, name):
    state = states[qubit]
    if name == "reset":
        states[qubit] = Z0
    elif name == "measure":
        if state not in _IS_Z:
            states[qubit] = GENERAL
    elif state == GENERAL:
        pass
    elif name in _FIXED_1Q:
        states[qubit] = _FIXED_1Q[name][state]
    elif name in _DIAGONAL_1Q:
        if state not in _IS_Z:
            states[qubit] = GENERAL
    elif name in _X_ROTATION_1Q:
        if state in _IS_Z:
            states[qubit] = GENERAL
    else:
        states[qubit] = GENERAL


def _cx_entangles(states, control, target):
    if states[control] == Z0:
        return False
    if states[control] == Z1:
        _apply_1q(states, target, "x")
        return False
    if states[target] == X_PLUS:
        return False
    if states[target] == X_MINUS:
        _apply_1q(states, control, "z")
        return False
    states[control] = states[target] = GENERAL
    return True


def can_entangle(states, name, params, qubits, conditional):
    """Update per-qubit states. Return False if the gate provably leaves a product state unentangled."""
    if conditional:
        for q in qubits:
            states[q] = GENERAL
        return len(qubits) > 1
    if len(qubits) == 1:
        _apply_1q(states, qubits[0], name)
        return False
    if name in ANGLE_2Q:
        angle = angle_from_zero(params)
        if angle is not None and angle < 1e-9:
            return False
    if name in ("cx", "CX") and len(qubits) == 2:
        return _cx_entangles(states, *qubits)
    if name in _DIAGONAL_2Q and len(qubits) == 2:
        a, b = qubits
        other = "rz" if name != "cz" else "z"
        # A qubit in |0> or |1> reduces a diagonal two-qubit gate to a phase on the other qubit.
        # This covers rzz too: there the phase is an rz on the other qubit.
        if states[a] in _IS_Z or states[b] in _IS_Z:
            if name == "cz":
                if states[a] == Z1:
                    _apply_1q(states, b, "z")
                elif states[b] == Z1:
                    _apply_1q(states, a, "z")
            elif name == "rzz" or states[a] == Z1 or states[b] == Z1:
                if states[a] not in _IS_Z:
                    _apply_1q(states, a, other)
                if states[b] not in _IS_Z:
                    _apply_1q(states, b, other)
            return False
        states[a] = states[b] = GENERAL
        return True
    if name in _CONTROLLED_2Q and len(qubits) == 2:
        if states[qubits[0]] == Z0:
            return False
        states[qubits[0]] = states[qubits[1]] = GENERAL
        return True
    if name == "swap" and len(qubits) == 2:
        a, b = qubits
        moves_entanglement = states[a] == GENERAL or states[b] == GENERAL
        states[a], states[b] = states[b], states[a]
        return moves_entanglement
    if name in ("ccx", "rccx") and len(qubits) == 3:
        a, b, target = qubits
        if states[a] == Z0 or states[b] == Z0:
            return False
        if states[a] == Z1:
            return _cx_entangles(states, b, target)
        if states[b] == Z1:
            return _cx_entangles(states, a, target)
        if states[target] == X_PLUS:
            return False
        states[a] = states[b] = states[target] = GENERAL
        return True
    if name == "cswap" and len(qubits) == 3:
        control, a, b = qubits
        if states[control] == Z0:
            return False
        if states[control] == Z1:
            moves_entanglement = states[a] == GENERAL or states[b] == GENERAL
            states[a], states[b] = states[b], states[a]
            return moves_entanglement
        if states[a] == states[b] != GENERAL:
            return False
        states[control] = states[a] = states[b] = GENERAL
        return True
    for q in qubits:
        states[q] = GENERAL
    return True


# ---------------------------------------------------------------------- #
# Stabilizer tableau                                                     #
# ---------------------------------------------------------------------- #
class Tableau:
    """
    Stabilizer generators stored by column: bit g of x[q] is generator g's X part on qubit q.
    Signs are dropped because they never change entanglement, so Paulis are no-ops.
    """

    __slots__ = ("x", "z")

    def __init__(self, n_qubits):
        self.x = [0] * n_qubits
        self.z = [1 << q for q in range(n_qubits)]

    def add_qubits(self, n_qubits):
        for q in range(len(self.x), n_qubits):
            self.x.append(0)
            self.z.append(1 << q)

    def h(self, q):
        self.x[q], self.z[q] = self.z[q], self.x[q]

    def s(self, q):
        self.z[q] ^= self.x[q]

    def sx(self, q):
        self.x[q] ^= self.z[q]

    def cx(self, control, target):
        self.x[target] ^= self.x[control]
        self.z[control] ^= self.z[target]

    def cz(self, a, b):
        self.z[a] ^= self.x[b]
        self.z[b] ^= self.x[a]

    def swap(self, a, b):
        self.x[a], self.x[b] = self.x[b], self.x[a]
        self.z[a], self.z[b] = self.z[b], self.z[a]

    def measure(self, q):
        """Measure in the Z basis. The outcome doesn't affect entanglement, so it isn't chosen."""
        anticommuting = self.x[q]
        if not anticommuting:
            return
        pivot = anticommuting & -anticommuting
        others = anticommuting ^ pivot
        x, z = self.x, self.z
        keep = ~pivot
        for j in range(len(x)):
            if x[j] & pivot:
                x[j] = (x[j] ^ others) & keep
            if z[j] & pivot:
                z[j] = (z[j] ^ others) & keep
        z[q] |= pivot

    def cut_entropies(self):
        """Entanglement in bits across every cut: rank(generators on the left qubits) - qubits on the left."""
        basis = {}  # highest bit -> reduced column
        rank = 0
        out = []
        x, z = self.x, self.z
        for k in range(len(x) - 1):
            for vector in (x[k], z[k]):
                while vector:
                    top = vector.bit_length()
                    pivot = basis.get(top)
                    if pivot is None:
                        basis[top] = vector
                        rank += 1
                        break
                    vector ^= pivot
            out.append(rank - k - 1)
        return out

    def apply(self, name, params, qubits, effective):
        """Apply a gate, replacing a non-Clifford gate with a Clifford of the same general kind."""
        width = len(qubits)
        if width == 1:
            q = qubits[0]
            if name in ("x", "y", "z", "id"):
                return
            if name == "h":
                self.h(q)
            elif name in ("s", "sdg", "t", "tdg"):
                self.s(q)
            elif name in ("sx", "sxdg"):
                self.sx(q)
            elif name in _ANGLE_GATES:
                action = _clifford_rotation(params)
                if action == "quarter":
                    if name in ("rz", "p", "u1"):
                        self.s(q)
                    elif name == "rx":
                        self.sx(q)
                    else:
                        self.h(q)
            else:  # u2, u3 and other general single-qubit gates
                self.h(q)
                self.s(q)
            return
        if width == 2:
            a, b = qubits
            if name in ("cx", "CX", "cy", "ch", "csx", "crx", "cry", "cu", "cu3"):
                if name in ("crx", "cry") and _clifford_rotation(params) == "none":
                    return
                self.cx(a, b)
            elif name == "cz":
                self.cz(a, b)
            elif name == "swap":
                self.swap(a, b)
            elif name in ANGLE_2Q:
                if _clifford_rotation(params) == "quarter":
                    if name in ("rxx", "ryy"):
                        self.h(a)
                        self.h(b)
                    self.cz(a, b)
                    if name in ("rxx", "ryy"):
                        self.h(a)
                        self.h(b)
            elif effective:
                self.cx(a, b)
            return
        if not effective:
            return
        if name in ("ccx", "rccx") and width == 3:
            self.cx(qubits[0], qubits[2])
            self.cx(qubits[1], qubits[2])
        elif name == "cswap" and width == 3:
            control, a, b = qubits
            self.cx(b, a)
            self.cx(control, b)
            self.cx(a, b)
            self.cx(b, a)
        else:
            for q in qubits[1:]:
                self.cx(qubits[0], q)


def _binary_entropy(weight):
    if weight <= 0 or weight >= 1:
        return 0.0
    return -weight * math.log2(weight) - (1 - weight) * math.log2(1 - weight)


_CLIFFORD_GATES = {"h", "s", "sdg", "x", "y", "z", "id", "sx", "sxdg", "cx", "CX", "cz", "cy", "swap"}
_T_WEIGHT = _binary_entropy(math.sin(math.pi / 8) ** 2)
_ROTATION_WEIGHT = {}


def nonclifford_weight(name, params, effective, before):
    """How far a gate is from being Clifford, in bits. 0 means the tableau follows it exactly."""
    if name in _CLIFFORD_GATES:
        return 0.0
    if len(before) == 1:
        if name in _DIAGONAL_1Q and before[0] in _IS_Z:
            return 0.0
        if name in _X_ROTATION_1Q and before[0] in (X_PLUS, X_MINUS):
            return 0.0
    if name in ("t", "tdg"):
        return _T_WEIGHT
    if name in _ANGLE_GATES:
        key = (name, params)
        weight = _ROTATION_WEIGHT.get(key)
        if weight is None:
            theta = signed_angle(params)
            if theta is None:
                weight = 1.0
            elif name in ("cp", "cu1", "crz", "crx", "cry"):
                # Controlled rotations are Clifford only at multiples of pi.
                weight = _binary_entropy(math.sin(abs((theta + math.pi / 2) % math.pi - math.pi / 2) / 4) ** 2)
            else:
                # Plain rotations and rzz-style gates are Clifford at multiples of pi/2.
                weight = _binary_entropy(math.sin(abs((theta + math.pi / 4) % (math.pi / 2) - math.pi / 4) / 2) ** 2)
            if theta is not None and abs(theta) * 1e-15 > 1e-12 and weight < 1e-6:
                weight = 0.0
            _ROTATION_WEIGHT[key] = weight
        return weight
    if len(before) == 1:
        return 1.0
    return 1.0 if effective else 0.0


def _clifford_rotation(params):
    """'none' for a rotation that is roughly the identity or a Pauli here, otherwise 'quarter'."""
    theta = signed_angle(params)
    if theta is None:
        return "quarter"
    from_pi = abs((theta + math.pi / 2) % math.pi - math.pi / 2)
    tolerance = max(1e-9 + abs(theta) * 1e-15, SMALL_ANGLE)
    return "none" if from_pi < tolerance else "quarter"


# ---------------------------------------------------------------------- #
# Reading a circuit with lazily compiled gate definitions                #
# ---------------------------------------------------------------------- #
class _LazyGates(dict):
    """Gate definitions that are only compiled when a circuit uses them."""

    def __init__(self, text, spans, parser):
        super().__init__()
        self.text = text
        self.spans = spans
        self.parser = parser

    def get(self, name, default=None):
        gate = dict.get(self, name)
        if gate is not None:
            return gate
        span = self.spans.get(name)
        if span is None:
            return default
        formals_text, qargs_text, start, end = span
        formals = [p.strip() for p in formals_text.split(",") if p.strip()]
        qargs = [q.strip() for q in qargs_text.split(",") if q.strip()]
        gate = _GateDef(formals, {q: i for i, q in enumerate(qargs)}, [])
        for statement in _statements(self.text[start:end].splitlines()):
            self.parser._add_to_gate_body(gate, statement)
        self[name] = gate
        return gate


def _index_gates(text):
    """Find gate definitions cheaply. Returns name -> span, and the text with definitions removed."""
    spans = {}
    pieces = []
    position = 0
    for match in _GATE_BLOCK.finditer(text):
        spans[match.group(1)] = (match.group(2) or "", match.group(3), match.start(4), match.end(4))
        pieces.append(text[position:match.start()])
        position = match.end()
    if not spans:
        return spans, text
    pieces.append(text[position:])
    return spans, "\n".join(pieces)


def _lines_with_position(text, tracker):
    start = 0
    find = text.find
    while True:
        end = find("\n", start)
        if end < 0:
            tracker[0] = len(text)
            yield text[start:]
            return
        tracker[0] = end + 1
        yield text[start:end]
        start = end + 1


def _count_remaining(remaining_text, source_text, spans, deadline):
    """Count operations in untracked text, including custom gate expansion, without expanding them."""
    tokens = collections.Counter(_LEADING_WORD.findall(remaining_text))
    sizes = {}
    timed_out = False

    def size(name, depth=0):
        nonlocal timed_out
        if name not in spans:
            return 1
        if name in sizes:
            return sizes[name]
        if timed_out or depth > 64:
            return 1
        if time.perf_counter() > deadline:
            timed_out = True
            return 1
        _, _, start, end = spans[name]
        total = 0
        for token in _LEADING_WORD.findall(source_text, start, end):
            if token not in _NOT_OPERATIONS:
                total += size(token, depth + 1)
        sizes[name] = total
        return total

    total = 0
    for token, count in tokens.items():
        if token not in _NOT_OPERATIONS:
            total += count * size(token)
    return total, timed_out


# ---------------------------------------------------------------------- #
# The profiler                                                           #
# ---------------------------------------------------------------------- #
def _chi_from_bits(bits, threshold):
    return threshold if bits >= 11 else min(2.0 ** bits, threshold)


class _Profiler:
    def __init__(self, keep_snapshots):
        self.keep_snapshots = keep_snapshots
        self.n = 0
        self.states = []
        self.groups = []
        self.free_groups = []
        self.group_of = []
        self.stacks = []
        self.power = []
        self.crossing_diff = []
        self.total_crossings = []
        self.tableau = Tableau(0)
        self.counts = collections.Counter()
        self.ops = 0
        self.est_hist = [0.0] * BOND_BINS
        self.stab_hist = [0.0] * BOND_BINS
        self.power_hist = [0.0] * BOND_BINS
        self.snapshots = []
        self.next_snapshot = 16
        self.last_snapshot_ops = 0
        self.work = 0
        self.names = collections.Counter()
        self.level = []
        self.level_2q = []
        self.depth = 0
        self.depth_2q = 0
        self.pairs = set()
        self.sum_span = 0
        self.measured = set()
        self.after_measure = 0
        self.last_cp = []
        self.tokens = []
        self.peak_state = dict.fromkeys(THRESHOLDS, 0.0)
        self.state_area = dict.fromkeys(THRESHOLDS, 0.0)
        self.peak = {"est": 0.0, "stab": 0.0, "power": 0.0, "group": 0.0}
        self.peak_mean = {"est": 0.0, "stab": 0.0, "power": 0.0, "group": 0.0}
        self.largest_group = 1
        self.magic = 0.0
        self.last_est = []

    def resize(self, n):
        self.tableau.add_qubits(n)
        for q in range(self.n, n):
            self.states.append(Z0)
            self.groups.append({q})
            self.group_of.append(q)
            self.stacks.append([])
            self.power.append(0.0)
            self.crossing_diff.append(0)
            self.total_crossings.append(0)
            self.level.append(0)
            self.level_2q.append(0)
            self.last_cp.append(None)
            self.tokens.append(collections.deque(maxlen=len(_TOFFOLI_TARGET)))
        self.crossing_diff.append(0)
        self.n = n

    def _try_cancel(self, name, params, qubits):
        stack = self.stacks[qubits[0]]
        if not stack:
            return False
        record = stack[-1]
        if record[2] != qubits:
            return False
        for q in qubits:
            top = self.stacks[q]
            if not top or top[-1] is not record:
                return False
        previous = record[0]
        if name in _SELF_INVERSE:
            if previous != name:
                return False
        elif name in _INVERSE_NAME:
            if previous != _INVERSE_NAME[name]:
                return False
        elif name in _ANGLE_GATES and previous == name:
            first, second = signed_angle(record[1]), signed_angle(params)
            if first is None or second is None or abs(first + second) > 1e-9:
                return False
        else:
            return False
        for q, state in zip(qubits, record[3]):
            self.stacks[q].pop()
            self.states[q] = state
        self.magic = max(0.0, self.magic - record[5])
        added = record[4]
        if added:
            low = min(qubits)
            power = self.power
            for offset, amount in enumerate(added):
                power[low + offset] = max(0.0, power[low + offset] - amount)
        self.counts["n_cancelled"] += 2
        return True

    def _merge_cx_pair(self, qubits):
        """
        cx(a,b) rz(theta)[b] cx(a,b) is exactly rzz(theta), and cx(a,b) rx(theta)[a] cx(a,b) is rxx(theta).
        If this cx closes such a sequence, take back the first cx and return the true entangling power.
        """
        a, b = qubits
        stack_a, stack_b = self.stacks[a], self.stacks[b]
        if len(stack_b) >= 2 and stack_a and stack_b[-2] is stack_a[-1]:
            first, rotation, wanted = stack_a[-1], stack_b[-1], ("rz", "p", "u1", "t", "tdg", "s", "sdg")
        elif len(stack_a) >= 2 and stack_b and stack_a[-2] is stack_b[-1]:
            first, rotation, wanted = stack_b[-1], stack_a[-1], ("rx",)
        else:
            return None
        if first[0] not in ("cx", "CX") or first[2] != qubits or rotation[0] not in wanted:
            return None
        params = _DIAGONAL_ANGLE.get(rotation[0], rotation[1])
        self.counts["n_cx_rotation_cx"] += 1
        added = first[4]
        if added:
            low = min(qubits)
            for offset, amount in enumerate(added):
                self.power[low + offset] = max(0.0, self.power[low + offset] - amount)
                added[offset] = 0.0
        return entangling_power("rzz", params)

    def _note_phase_ladder(self, params, qubits):
        """Count controlled-phase gates whose angle halves or doubles along a shared qubit, as in a QFT."""
        theta = signed_angle(params)
        if not theta:
            return
        for q in qubits:
            previous = self.last_cp[q]
            if previous and (abs(previous / theta - 2) < 1e-6 or abs(previous / theta - 0.5) < 1e-6):
                self.counts["n_phase_ladder"] += 1
                break
        for q in qubits:
            self.last_cp[q] = theta
        if abs(theta) < SMALL_ANGLE:
            self.counts["n_cp_small"] += 1

    def _note_toffoli(self, name, qubits):
        """Count the standard Clifford+T Toffoli pattern as seen on its target qubit."""
        if name in ("cx", "CX") and len(qubits) == 2:
            self.tokens[qubits[0]].append(".")
            self.tokens[qubits[1]].append("c")
            return
        token = _TOFFOLI_TOKENS.get(name, ".")
        for q in qubits:
            history = self.tokens[q]
            history.append(token)
            if name == "h" and tuple(history) == _TOFFOLI_TARGET:
                self.counts["n_toffoli_motif"] += 1

    def update(self, op):
        name, params, qubits, conditional = op
        if name == "barrier":
            return
        self.ops += 1
        self.work += 1
        states = self.states
        width = len(qubits)
        self.names[name] += 1

        level = self.level
        column = max(level[q] for q in qubits)
        if conditional:
            column = max(column, self.after_measure)
        if name == "measure":
            self.after_measure = max(self.after_measure, column + 1)
        for q in qubits:
            level[q] = column + 1
        if column >= self.depth:
            self.depth = column + 1
        if self.measured and name != "measure":
            for q in qubits:
                if q in self.measured:
                    self.counts["n_ops_on_measured"] += 1
                    break

        if name == "measure" or name == "reset":
            self.counts["n_measure" if name == "measure" else "n_reset"] += 1
            q = qubits[0]
            if name == "measure":
                self.measured.add(q)
            self.work += self.n // 8  # collapsing the tableau touches every qubit
            self.stacks[q].clear()
            _apply_1q(states, q, name)
            self.tableau.measure(q)
            group = self.groups[self.group_of[q]]
            if len(group) > 1:
                group.discard(q)
                # A collapsed qubit starts its own group in an empty slot left by an earlier merge.
                slot = self.free_groups.pop() if self.free_groups else None
                if slot is None:
                    slot = len(self.groups)
                    self.groups.append(set())
                self.groups[slot] = {q}
                self.group_of[q] = slot
            return

        if conditional:
            self.counts["n_conditional"] += 1
        if width == 1:
            self.counts["n_1q"] += 1
            if name in ("t", "tdg"):
                self.counts["n_t"] += 1
            if name in _CLIFFORD_1Q:
                self.counts["n_1q_clifford"] += 1
        else:
            self.counts["n_multi"] += 1
            low, high = min(qubits), max(qubits)
            self.crossing_diff[low] += 1
            self.crossing_diff[high] -= 1
            self.work += (high - low) // 4
            self.sum_span += high - low
            if high - low == 1:
                self.counts["n_nearest_neighbor"] += 1
            self.pairs.add((low, high))
            level_2q = self.level_2q
            column_2q = max(level_2q[q] for q in qubits)
            for q in qubits:
                level_2q[q] = column_2q + 1
            if column_2q >= self.depth_2q:
                self.depth_2q = column_2q + 1
            if name in ("cp", "cu1", "crz"):
                self._note_phase_ladder(params, qubits)
        self._note_toffoli(name, qubits)

        if not conditional and self._try_cancel(name, params, qubits):
            self.tableau.apply(name, params, qubits, True)
            return

        saved = tuple(states[q] for q in qubits)
        effective = can_entangle(states, name, params, qubits, conditional)
        added = None
        if effective:
            self.counts["n_effective"] += 1
            low, high = min(qubits), max(qubits)
            gate_power = entangling_power(name, params)
            if name in ("cx", "CX") and width == 2:
                merged = self._merge_cx_pair(qubits)
                if merged is not None:
                    gate_power = merged
            n = self.n
            power = self.power
            added = []
            for cut in range(low, high):
                before = power[cut]
                limit = min(cut + 1, n - cut - 1)
                value = before + gate_power
                if value > limit:
                    value = limit
                power[cut] = value
                added.append(value - before)
            first = self.group_of[qubits[0]]
            for q in qubits[1:]:
                other = self.group_of[q]
                if other != first:
                    groups = self.groups
                    if len(groups[other]) > len(groups[first]):
                        first, other = other, first
                    for member in groups[other]:
                        self.group_of[member] = first
                    groups[first] |= groups[other]
                    groups[other] = set()
                    self.free_groups.append(other)
            if len(self.groups[first]) > self.largest_group:
                self.largest_group = len(self.groups[first])

        self.tableau.apply(name, params, qubits, effective)
        magic = nonclifford_weight(name, params, effective, saved)
        self.magic += magic
        record = (name, params, qubits, saved, added, magic)
        for q in qubits:
            stack = self.stacks[q]
            stack.append(record)
            if len(stack) > STACK_DEPTH:
                del stack[:STACK_DEPTH // 2]

    def snapshot(self):
        n = self.n
        if n < 2:
            return
        stab = self.tableau.cut_entropies()
        power = self.power[:n - 1]
        # Each group can only be cut as many ways as its smaller side has members.
        group_bound = []
        left = collections.Counter()
        sizes = [len(g) for g in self.groups]
        running = 0
        group_of = self.group_of
        for q in range(n - 1):
            g = group_of[q]
            before = min(left[g], sizes[g] - left[g])
            left[g] += 1
            running += min(left[g], sizes[g] - left[g]) - before
            group_bound.append(running)
        magic = self.magic
        est = [min(s + magic, p, b) for s, p, b in zip(stab, power, group_bound)]

        crossing = 0
        diff = self.crossing_diff
        histograms = (self.est_hist, est), (self.stab_hist, stab), (self.power_hist, power)
        for c in range(n - 1):
            crossing += diff[c]
            diff[c] = 0
            if crossing:
                self.total_crossings[c] += crossing
                for histogram, values in histograms:
                    value = values[c]
                    histogram[int(value) if value < BOND_BINS else BOND_BINS - 1] += crossing
        diff[n - 1] = 0
        if n < len(diff):
            diff[n] = 0

        delta = self.ops - self.last_snapshot_ops
        state_now = {}
        for threshold in THRESHOLDS:
            chi = [1.0] + [_chi_from_bits(e, threshold) for e in est] + [1.0]
            size = sum(chi[i] * 2 * chi[i + 1] for i in range(n))
            state_now[threshold] = size
            self.state_area[threshold] += size * delta
            if size > self.peak_state[threshold]:
                self.peak_state[threshold] = size
        for key, values in (("est", est), ("stab", stab), ("power", power), ("group", group_bound)):
            self.peak[key] = max(self.peak[key], max(values))
            self.peak_mean[key] = max(self.peak_mean[key], sum(values) / len(values))
        self.last_est = est
        self.last_stab = stab
        self.last_power = power
        self.last_group = group_bound
        if self.keep_snapshots:
            self.snapshots.append({"ops": self.ops, "est": est, "stabilizer": stab, "structural": list(power),
                                   "group": group_bound, "state_size": state_now})
        self.last_snapshot_ops = self.ops
        self.next_snapshot = self.ops + max(16, self.ops // 16)
        self.work += n * n // 32  # the rank calculation grows with the square of the qubit count


def profile_text(text, max_work=MAX_WORK, keep_snapshots=False, time_limit=TIME_LIMIT, indexed=None):
    """Profile OpenQASM source text. See entanglement_profile for the result."""
    started = time.perf_counter()
    tracking_deadline = started + time_limit * 0.75
    counting_deadline = started + time_limit

    spans, main = indexed if indexed is not None else _index_gates(text)
    parser = _Parser(expand_gates=True)
    parser.gates = _LazyGates(text, spans, parser)
    profiler = _Profiler(keep_snapshots)
    position = [0]
    stopped_early = False
    hit_time_limit = False
    next_clock_check = 4096
    for op in parser.ops(_lines_with_position(main, position)):
        if parser.n_qubits != profiler.n:
            profiler.resize(parser.n_qubits)
        profiler.update(op)
        ops = profiler.ops
        if ops >= profiler.next_snapshot:
            profiler.snapshot()
        if profiler.work >= max_work:
            stopped_early = True
            break
        if ops >= next_clock_check:
            next_clock_check = ops + 4096
            if time.perf_counter() > tracking_deadline:
                stopped_early = hit_time_limit = True
                break
    if parser.n_qubits != profiler.n:
        profiler.resize(parser.n_qubits)
    profiler.snapshot()

    tracked = profiler.ops
    remaining = 0
    count_estimated = False
    if stopped_early:
        # Count the whole circuit: tracking can stop partway through one large custom gate,
        # so the untracked text alone would miss the rest of that gate.
        whole, count_estimated = _count_remaining(main, text, spans, counting_deadline)
        remaining = max(whole - tracked, 0)
        # Assume the rest of the circuit crosses each cut as often as the tracked part did.
        scale = remaining / max(tracked, 1)
        for c, value in enumerate(profiler.last_est):
            extra = profiler.total_crossings[c] * scale
            profiler.est_hist[int(value) if value < BOND_BINS else BOND_BINS - 1] += extra
            stab_value = profiler.last_stab[c]
            profiler.stab_hist[int(stab_value) if stab_value < BOND_BINS else BOND_BINS - 1] += extra
            power_value = profiler.last_power[c]
            profiler.power_hist[int(power_value) if power_value < BOND_BINS else BOND_BINS - 1] += extra
        for threshold in THRESHOLDS:
            profiler.state_area[threshold] += sum(_edge_sizes(profiler.last_est, threshold)) * remaining

    total = tracked + remaining
    scale_counts = total / max(tracked, 1)
    counts = profiler.counts
    n = profiler.n
    degree = collections.Counter()
    for a, b in profiler.pairs:
        degree[a] += 1
        degree[b] += 1
    families = collections.Counter()
    for name, count in profiler.names.items():
        families[GATE_FAMILIES.get(name, "other")] += count
    result = {
        "n_qubits": n,
        "n_ops": total,
        "tracked_ops": tracked,
        "coverage": tracked / max(total, 1),
        "count_estimated": int(count_estimated),
        "n_gate_defs": len(spans),
        "n_1q": counts["n_1q"] * scale_counts,
        "n_multi": counts["n_multi"] * scale_counts,
        "n_effective": counts["n_effective"] * scale_counts,
        "n_cancelled": counts["n_cancelled"] * scale_counts,
        "n_t": counts["n_t"] * scale_counts,
        "n_measure": counts["n_measure"] * scale_counts,
        "n_reset": counts["n_reset"] * scale_counts,
        "n_conditional": counts["n_conditional"] * scale_counts,
        "largest_group": profiler.largest_group,
        "magic": profiler.magic,
        "hit_time_limit": int(hit_time_limit),
        "depth": profiler.depth * scale_counts,
        "depth_2q": profiler.depth_2q * scale_counts,
        "n_pairs": len(profiler.pairs),
        "max_degree": max(degree.values(), default=0),
        "mean_span": profiler.sum_span / max(counts["n_multi"], 1),
        "n_distinct_gates": len(profiler.names),
        "n_1q_clifford": counts["n_1q_clifford"] * scale_counts,
        "n_nearest_neighbor": counts["n_nearest_neighbor"] * scale_counts,
        "n_ops_on_measured": counts["n_ops_on_measured"] * scale_counts,
        "n_cx_rotation_cx": counts["n_cx_rotation_cx"] * scale_counts,
        "n_phase_ladder": counts["n_phase_ladder"] * scale_counts,
        "n_cp_small": counts["n_cp_small"] * scale_counts,
        "n_toffoli_motif": counts["n_toffoli_motif"] * scale_counts,
        "profile_seconds": 0.0,
    }
    for family in FAMILY_NAMES:
        result[f"family_{family}"] = families[family] * scale_counts
    for key in ("est", "stab", "power", "group"):
        result[f"peak_{key}"] = profiler.peak[key]
        result[f"peak_mean_{key}"] = profiler.peak_mean[key]
    if profiler.last_est:
        for key, values in (("est", profiler.last_est), ("stab", profiler.last_stab),
                            ("power", profiler.last_power), ("group", profiler.last_group)):
            result[f"final_max_{key}"] = max(values)
            result[f"final_mean_{key}"] = sum(values) / len(values)
    else:
        for key in ("est", "stab", "power", "group"):
            result[f"final_max_{key}"] = 0.0
            result[f"final_mean_{key}"] = 0.0
    for threshold in THRESHOLDS:
        result[f"peak_state_{threshold}"] = profiler.peak_state[threshold]
        result[f"mean_state_{threshold}"] = profiler.state_area[threshold] / max(total, 1)
    for prefix, histogram in (("est", profiler.est_hist), ("stab", profiler.stab_hist), ("power", profiler.power_hist)):
        for i, value in enumerate(histogram):
            result[f"{prefix}_updates_{i}"] = value
    if keep_snapshots:
        result["snapshots"] = profiler.snapshots
    result["skipped"] = dict(parser.skipped)
    result["profile_seconds"] = time.perf_counter() - started
    return result


def _edge_sizes(est, threshold):
    chi = [1.0] + [_chi_from_bits(e, threshold) for e in est] + [1.0]
    return [chi[i] * 2 * chi[i + 1] for i in range(len(chi) - 1)]


def entanglement_profile(path, max_work=MAX_WORK, keep_snapshots=False):
    """
    Estimate entanglement growth for a .qasm or .qasm.zst file.

    Returns a dictionary of summary values. With keep_snapshots=True, "snapshots" also lists
    the per-cut estimates as the circuit progresses:
        {"ops": 1200, "est": [...], "stabilizer": [...], "structural": [...], "group": [...],
         "state_size": {16: ..., 64: ..., ...}}

    Bond values are in bits, so bond dimension is about 2^value, capped by threshold. "state_size"
    counts the numbers a chain-shaped tensor network would store. max_work limits how much of a
    very large circuit is tracked; "coverage" below 1 means the end of it was extrapolated.
    """
    with _open_lines(path) as lines:
        text = "\n".join(line.rstrip("\n") for line in lines)
    return profile_text(text, max_work, keep_snapshots)


# ---------------------------------------------------------------------- #
# Plot                                                                   #
# ---------------------------------------------------------------------- #
def plot_entanglement_profile(path, out=None, max_work=MAX_WORK):
    """Plot how the estimated bonds and state size change as the circuit progresses."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    from visualize_qasm import BLUE_RAMP, GATE, INK, MUTED, NON_UNITARY, SECONDARY, SURFACE, WIRE, _title

    profile = entanglement_profile(path, max_work, keep_snapshots=True)
    snapshots = profile["snapshots"]
    if not snapshots:
        raise ValueError("The circuit needs at least two qubits and one operation to profile.")
    ops = np.array([s["ops"] for s in snapshots], dtype=float)
    edges = np.concatenate([[0.0], ops])
    grid = np.array([s["est"] for s in snapshots], dtype=float).T

    fig_height = 5.2
    fig = plt.figure(figsize=(14, fig_height), facecolor=SURFACE)
    layout = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], left=0.05, right=0.985, top=0.79, bottom=0.12, wspace=0.27)
    bonds, means, sizes = (fig.add_subplot(layout[i]) for i in range(3))

    ramp = LinearSegmentedColormap.from_list("bonds", [SURFACE] + BLUE_RAMP)
    top = max(1.0, grid.max())
    mesh = bonds.pcolormesh(edges, np.arange(grid.shape[0] + 1) + 0.5, np.ma.masked_equal(grid, 0),
                            cmap=ramp, vmin=0, vmax=top, shading="flat")
    bonds.invert_yaxis()
    bonds.set_title("Estimated entanglement at each cut", color=INK, fontsize=10.5, loc="left")
    bonds.set_ylabel("Cut after qubit", color=SECONDARY, fontsize=9.5)
    bar = fig.colorbar(mesh, ax=bonds, pad=0.015, fraction=0.04)
    bar.set_label("Bits (bond dimension = 2^bits)", color=SECONDARY, fontsize=8.5)
    bar.outline.set_edgecolor(WIRE)
    bar.ax.tick_params(colors=MUTED, labelsize=8.5, length=0)

    series = (("structural", "Structural bound", GATE), ("stabilizer", "Stabilizer", NON_UNITARY), ("group", "Group bound", MUTED))
    for key, label, color in series:
        values = [sum(s[key]) / len(s[key]) for s in snapshots]
        means.plot(ops, values, color=color, linewidth=2, label=label)
    means.set_title("Mean bits per cut, by method", color=INK, fontsize=10.5, loc="left")
    means.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY)

    shades = {16: BLUE_RAMP[0], 256: BLUE_RAMP[2], 1024: BLUE_RAMP[3]}
    endpoints = collections.defaultdict(list)
    for threshold, color in shades.items():
        values = [s["state_size"][threshold] for s in snapshots]
        sizes.plot(ops, values, color=color, linewidth=2)
        endpoints[round(math.log10(max(values[-1], 1)), 1)].append((threshold, values[-1]))
    # Thresholds that end at the same size share a label, because their lines overlap.
    for matching in endpoints.values():
        names = ", ".join(str(threshold) for threshold, _ in matching)
        label = f"threshold{'s' if len(matching) > 1 else ''} {names}"
        sizes.annotate(label, (ops[-1], matching[-1][1]), xytext=(-4, 5), textcoords="offset points",
                       ha="right", color=SECONDARY, fontsize=8.5)
    sizes.set_yscale("log")
    sizes.set_title("Estimated numbers stored", color=INK, fontsize=10.5, loc="left")

    for ax in (bonds, means, sizes):
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("Operations completed", color=SECONDARY, fontsize=9.5)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _human(int(value))))
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for spine in ax.spines.values():
            spine.set_color(WIRE)
            spine.set_linewidth(0.8)
    for ax in (means, sizes):
        ax.grid(color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)

    name = Path(path).name.split(".")[0]
    subtitle = (f"{profile['n_qubits']} qubits  ·  {_human(int(profile['n_ops']))} operations  ·  "
                f"{profile['n_cancelled'] / max(profile['n_ops'], 1):.0%} cancelled  ·  largest group {profile['largest_group']}  ·  "
                f"{profile['coverage']:.0%} tracked  ·  {profile['profile_seconds']:.1f} s")
    _title(fig, name, subtitle, fig_height)
    if out is not None:
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot estimated entanglement growth for a circuit.")
    ap.add_argument("path")
    ap.add_argument("--out", help="Image file to save. Without this, opens a window.")
    ap.add_argument("--max-work", type=int, default=MAX_WORK, help="Operations and cut crossings to track.")
    args = ap.parse_args()
    plot_entanglement_profile(args.path, args.out, args.max_work)
    if args.out:
        print(f"Saved {args.out}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
