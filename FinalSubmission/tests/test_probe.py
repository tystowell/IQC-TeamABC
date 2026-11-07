"""Check the MPS probe against exact statevector answers."""
import os
import random
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from mps_probe import _Chain, _one_qubit_matrix, _two_qubit_matrix, mps_features  # noqa: E402


def exact_state(n, gates):
    state = np.zeros(2 ** n, dtype=np.complex128)
    state[0] = 1
    state = state.reshape([2] * n)
    for name, params, qubits in gates:
        if len(qubits) == 1:
            m = _one_qubit_matrix(name, params).astype(np.complex128)
            state = np.moveaxis(np.tensordot(m, state, axes=(1, qubits[0])), 0, qubits[0])
        else:
            m = _two_qubit_matrix(name, params).astype(np.complex128).reshape(2, 2, 2, 2)
            state = np.moveaxis(np.tensordot(m, state, axes=((2, 3), qubits)), (0, 1), qubits)
    return state


def exact_entropies(state, n):
    out = []
    for cut in range(1, n):
        s = np.linalg.svd(state.reshape(2 ** cut, -1), compute_uv=False)
        p = s ** 2
        p = p[p > 1e-12]
        out.append(float(-(p * np.log2(p)).sum()))
    return out


def chain_entropies(n, gates, chi):
    chain = _Chain(n, chi)
    for name, params, qubits in gates:
        if len(qubits) == 1:
            chain.one_qubit(qubits[0], _one_qubit_matrix(name, params))
        else:
            chain.two_qubit(qubits[0], qubits[1], _two_qubit_matrix(name, params))
    return chain.entropies(), chain


# Exact answers: Bell pair, GHZ, and undoing entanglement.
values, _ = chain_entropies(3, [("h", "", (0,)), ("cx", "", (0, 1))], 8)
assert np.allclose(values, [1, 0], atol=1e-4), values
values, _ = chain_entropies(4, [("h", "", (0,)), ("cx", "", (0, 1)), ("cx", "", (0, 2)), ("cx", "", (0, 3))], 8)
assert np.allclose(values, [1, 1, 1], atol=1e-4), values
values, _ = chain_entropies(2, [("h", "", (0,)), ("cx", "", (0, 1)), ("cx", "", (0, 1)), ("h", "", (0,))], 8)
assert np.allclose(values, [0], atol=1e-4), values

# Random circuits with every gate type, including long range and reversed qubit order, at a chi large enough to be exact.
random.seed(1)
one = [("h", ""), ("t", ""), ("s", ""), ("sx", ""), ("rx", "0.7"), ("ry", "1.1"), ("rz", "0.4"), ("p", "0.9"), ("u3", "0.3,0.5,0.8")]
two = [("cx", ""), ("cz", ""), ("cp", "0.6"), ("rzz", "0.8"), ("rxx", "0.5"), ("swap", ""), ("crz", "0.9"), ("cry", "0.4"), ("cu3", "0.3,0.2,0.6")]
n = 8
worst = 0.0
for trial in range(5):
    gates = []
    for _ in range(40):
        if random.random() < .5:
            g = random.choice(one)
            gates.append((g[0], g[1], (random.randrange(n),)))
        else:
            g = random.choice(two)
            a, b = random.sample(range(n), 2)
            gates.append((g[0], g[1], (a, b)))
    values, chain = chain_entropies(n, gates, 32)
    truth = exact_entropies(exact_state(n, gates), n)
    worst = max(worst, max(abs(x - y) for x, y in zip(values, truth)))
print(f"Largest entropy error against exact statevector, 5 random 8-qubit circuits, chi=32: {worst:.4f} bits")
assert worst < 0.02, worst

# A low cap can't exceed its own limit, and it reports truncation.
values, chain = chain_entropies(n, gates, 2)
assert max(values) <= 1.0001 and chain.truncated_updates > 0, (max(values), chain.truncated_updates)

# Merging: a cx pair cancels with no SVD, and a split-up entangler on one pair takes one update.
_, chain = chain_entropies(3, [("h", "", (0,)), ("cx", "", (0, 1)), ("cx", "", (0, 1)), ("h", "", (0,))], 8)
assert chain.updates == 0, chain.updates
values, chain = chain_entropies(3, [("h", "", (0,)), ("cx", "", (0, 1)), ("rz", "0.7", (1,)), ("cx", "", (1, 0)), ("ry", "0.4", (0,)), ("cx", "", (0, 1))], 8)
assert chain.updates == 1, chain.updates
# A rank-1 merged gate leaves its single-qubit effect behind: x on one qubit, written as cx then a local fix.
values, chain = chain_entropies(2, [("h", "", (0,)), ("rz", "0.3", (0,)), ("cz", "", (0, 1)), ("cz", "", (0, 1))], 8)
assert chain.updates == 0 and np.allclose(values, [0], atol=1e-4)
# Merging must match exact answers on a qubit pair that keeps being interrupted by other pairs.
gates = [("h", "", (0,)), ("h", "", (3,)), ("cx", "", (0, 1)), ("cx", "", (3, 2)), ("t", "", (1,)), ("cx", "", (1, 0)),
         ("cx", "", (1, 2)), ("rz", "0.5", (2,)), ("cx", "", (0, 3)), ("cx", "", (2, 1))]
values, _ = chain_entropies(4, gates, 32)
assert np.allclose(values, exact_entropies(exact_state(4, gates), 4), atol=1e-3), values
# Swapping through unentangled qubits takes no SVD.
_, chain = chain_entropies(6, [("h", "", (0,)), ("cx", "", (0, 5))], 8)
assert chain.updates <= 5, chain.updates

# The budget covers queued gates: long-range gates that never touch each other can't run far over it.
pairs = "".join(f"rx(0.7) q[{i}];\nrx(0.7) q[{i + 100}];\ncx q[{i}],q[{i + 100}];\n" for i in range(99))
queued = mps_features(f"OPENQASM 2.0;\nqreg q[200];\n{pairs}", chi=16, max_work=2000)
assert queued["mps_stopped_early"] == 1 and queued["mps_updates"] < 1500, queued["mps_updates"]

# The text entry point: same input gives identical features, except elapsed time.
qasm = 'OPENQASM 2.0;\nqreg q[4];\nh q[0];\ncx q[0],q[3];\nccx q[0],q[1],q[2];\nrz(0.3) q[1];\n' * 20
a, b = mps_features(qasm), mps_features(qasm)
a.pop("mps_seconds"), b.pop("mps_seconds")
assert a == b and a["mps_unknown_ops"] == 0, a
# The product-state shortcut (gates between unentangled qubits, applied without swaps) must stay exact.
random.seed(3)
worst, shortcuts = 0.0, 0
for trial in range(40):
    n = 9
    gates = [("x", "", (q,)) for q in range(n) if random.random() < 0.5]
    for _ in range(60):
        a, b = random.sample(range(n), 2)
        r = random.random()
        if r < 0.35:
            gates.append((random.choice(["cx", "cz", "swap"]), "", (a, b)))
        elif r < 0.6:
            gates.append(("cp", f"0.{random.randint(1, 9)}", (a, b)))
        elif r < 0.8:
            gates.append(random.choice([("h", "", (a,)), ("t", "", (a,)), ("rx", "0.7", (a,))]))
        else:
            gates.append(("rz", "0.4", (b,)))
    state = exact_state(n, gates)
    chain = _Chain(n, 64)
    for name, params, qubits in gates:
        if len(qubits) == 1:
            chain.one_qubit(qubits[0], _one_qubit_matrix(name, params))
        else:
            chain.two_qubit(qubits[0], qubits[1], _two_qubit_matrix(name, params))
    values = chain.entropies()
    shortcuts += chain.product_gates
    for cut in range(1, n):
        p = np.linalg.svd(state.reshape(2 ** cut, -1), compute_uv=False) ** 2
        p = p[p > 1e-14]
        worst = max(worst, abs(float(-(p * np.log2(p)).sum()) - values[cut - 1]))
assert shortcuts > 100 and worst < 1e-3, (shortcuts, worst)
print(f"Product-state shortcut: {shortcuts} gates, largest entropy error {worst:.5f} bits")

print("all probe checks passed")

