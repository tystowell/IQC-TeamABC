"""Known-answer checks for entanglement_profile."""
import os
import sys
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from entanglement_profile import Tableau, profile_text  # noqa: E402


def profile(body, n=4):
    return profile_text(f'OPENQASM 2.0;\nqreg q[{n}];\n{body}\n', keep_snapshots=True)


def last(result, key):
    return result["snapshots"][-1][key]


# Tableau exact answers: Bell pair on (0,1), GHZ across every cut, entanglement falls on undoing.
t = Tableau(4)
assert t.cut_entropies() == [0, 0, 0]
t.h(0)
t.cx(0, 1)
assert t.cut_entropies() == [1, 0, 0]
t.cx(0, 1)
t.h(0)
assert t.cut_entropies() == [0, 0, 0]
t = Tableau(4)
t.h(0)
for q in range(1, 4):
    t.cx(0, q)
assert t.cut_entropies() == [1, 1, 1]
# Two Bell pairs across the middle cut give 2 bits there.
t = Tableau(4)
t.h(0)
t.cx(0, 3)
t.h(1)
t.cx(1, 2)
assert t.cut_entropies() == [1, 2, 1]
# Measuring one half of a Bell pair removes its entanglement.
t = Tableau(2)
t.h(0)
t.cx(0, 1)
t.measure(0)
assert t.cut_entropies() == [0]
# Measuring a GHZ qubit collapses all of it, and a deterministic measurement changes nothing.
t = Tableau(3)
t.h(0)
t.cx(0, 1)
t.cx(0, 2)
t.measure(1)
assert t.cut_entropies() == [0, 0]
t = Tableau(2)
t.h(0)
t.cx(0, 1)
t.cz(0, 1)
before = t.cut_entropies()
t.measure(0)
assert before == [1] and t.cut_entropies() == [0]

# GHZ: the structural bound climbs but the stabilizer stays at 1 bit, and the estimate takes the lower.
r = profile("h q[0];\ncx q[0],q[1];\ncx q[0],q[2];\ncx q[0],q[3];")
assert last(r, "stabilizer") == [1, 1, 1], last(r, "stabilizer")
assert last(r, "structural") == [1.0, 2.0, 1.0], last(r, "structural")
assert last(r, "est") == [1.0, 1.0, 1.0], last(r, "est")

# No superposition: nothing can entangle, however many gates there are.
r = profile("cx q[0],q[1];\nccx q[0],q[1],q[2];\ncp(pi/2) q[0],q[3];")
assert r["n_effective"] == 0 and last(r, "est") == [0, 0, 0]

# A gate followed directly by its inverse cancels, and cascades through nested layers.
r = profile("h q[0];\ncx q[0],q[1];\nrz(0.7) q[1];\nrz(-0.7) q[1];\ncx q[0],q[1];\nh q[0];")
assert last(r, "structural") == [0.0, 0.0, 0.0] and r["n_cancelled"] == 6, (last(r, "structural"), r["n_cancelled"])
assert last(r, "est") == [0, 0, 0]

# Independent pairs: the group bound stops a long chain of updates counting as one block.
r = profile("h q[0];\ncx q[0],q[1];\nh q[2];\ncx q[2],q[3];", n=4)
assert last(r, "group") == [1, 0, 1], last(r, "group")
assert r["largest_group"] == 2

# Measuring removes a qubit from its group; reset too.
r = profile("h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\nmeasure q[2];", n=3)
assert last(r, "group") == [1, 0], last(r, "group")

# Angle weighting: a small rzz adds far less than a large one, and the state size follows.
small = profile("h q[0];\nh q[1];\nrx(1.0) q[0];\nrx(1.0) q[1];\nrzz(0.05) q[0],q[1];", n=2)
large = profile("h q[0];\nh q[1];\nrx(1.0) q[0];\nrx(1.0) q[1];\nrzz(1.5) q[0],q[1];", n=2)
assert last(small, "structural")[0] < 0.02 < 0.9 < last(large, "structural")[0]

# A pure Clifford circuit adds no budget, so the exact tableau sets the estimate.
r = profile("h q[0];\ns q[0];\ncx q[0],q[1];\ncz q[1],q[2];\nsdg q[0];\nswap q[2],q[3];")
assert r["magic"] == 0.0, r["magic"]
assert last(r, "est") == [min(a, b) for a, b in zip(last(r, "stabilizer"), last(r, "structural"))]

# The failure this budget fixes: a small rotation inside a cx pair is real entanglement, but the
# skeleton sees the two cx cancel. The estimate must follow the structural value up, not stay at 0.
fsim_like = ("h q[0];\nh q[1];\nrz(0.3) q[0];\nrz(0.3) q[1];\n" + "cx q[0],q[1];\nrx(0.09) q[0];\ncx q[0],q[1];\nrz(0.7) q[0];\nrz(0.9) q[1];\n" * 6)
r = profile(fsim_like, n=2)
assert last(r, "stabilizer") == [0], last(r, "stabilizer")
assert r["magic"] > 1 and last(r, "est")[0] > 0.1, (r["magic"], last(r, "est"))
# T on a basis state is only a phase, so it adds nothing; T on a superposition adds about 0.6 bits.
assert profile("t q[0];")["magic"] == 0.0
assert 0.59 < profile("h q[0];\nt q[0];")["magic"] < 0.61
# Exact multiples of pi are Clifford even with huge coefficients; t and tdg cancel and restore the budget.
assert profile("h q[0];\nh q[1];\ncp(268435456*pi) q[0],q[1];")["magic"] == 0.0
assert profile("h q[0];\nt q[0];\ntdg q[0];")["magic"] == 0.0

# Custom gates expand, and untracked counting handles nesting: 3 * 2 * 4 operations.
nested = ('OPENQASM 2.0;\nqreg q[2];\ngate a x,y { cx x,y; cx y,x; cx x,y; cx y,x; }\n'
          'gate b x,y { a x,y; a y,x; }\nb q[0],q[1];\nb q[1],q[0];\nb q[0],q[1];\n')
r = profile_text(nested)
assert r["n_ops"] == 24 and r["coverage"] == 1.0, (r["n_ops"], r["coverage"])

big = 'OPENQASM 2.0;\nqreg q[3];\n' + 'h q[0];\ncx q[0],q[1];\nrz(0.3) q[1];\n' * 400000
started = time.perf_counter()
r = profile_text(big, max_work=50000)
elapsed = time.perf_counter() - started
assert r["coverage"] < 1 and r["n_ops"] == 1200000 and r["tracked_ops"] < 1200000, (r["coverage"], r["n_ops"])
assert sum(r["est_updates_%d" % i] for i in range(12)) > 0
assert r["hit_time_limit"] == 0
# Same input gives identical features apart from the elapsed time.
again = profile_text(big, max_work=50000)
r.pop("profile_seconds"), again.pop("profile_seconds")
assert r == again, "features should be deterministic"

# cx rz cx is exactly rzz: full power only at rzz(pi/2), and small angles add little.
half = profile("h q[0];\nh q[1];\nrx(1) q[0];\nrx(1) q[1];\ncx q[0],q[1];\nrz(pi/2) q[1];\ncx q[0],q[1];", n=2)
assert half["n_cx_rotation_cx"] == 1 and 0.99 < last(half, "structural")[0] <= 1.0, last(half, "structural")
tiny = profile("h q[0];\nh q[1];\nrx(1) q[0];\nrx(1) q[1];\ncx q[0],q[1];\nrz(0.01) q[1];\ncx q[0],q[1];", n=2)
assert tiny["n_cx_rotation_cx"] == 1 and last(tiny, "structural")[0] < 0.02, last(tiny, "structural")
# A rotation on the control is rxx, and an unrelated gate between the cx pair doesn't merge.
rxx = profile("ry(0.7) q[0];\nry(0.7) q[1];\ncx q[0],q[1];\nrx(0.01) q[0];\ncx q[0],q[1];", n=2)
assert rxx["n_cx_rotation_cx"] == 1
other = profile("ry(0.7) q[0];\nry(0.7) q[1];\ncx q[0],q[1];\nrx(0.01) q[1];\ncx q[0],q[1];", n=2)
assert other["n_cx_rotation_cx"] == 0

# Stopping inside one large custom gate must not lose the rest of that gate from the count.
large_gate = "OPENQASM 2.0;\nqreg q[2];\ngate g x,y {\n" + "cx x,y;\nrz(0.3) y;\n" * 5000 + "}\ng q[0],q[1];\nh q[0];\n"
inside = profile_text(large_gate, max_work=500)
assert inside["coverage"] < 1 and inside["n_ops"] == 10001, (inside["coverage"], inside["n_ops"])

# Depth, depth of multi-qubit gates, and gate families.
d = profile("h q[0];\ncx q[0],q[1];\ncx q[1],q[2];\nh q[3];\nu1(0.2) q[3];", n=4)
assert d["depth"] == 3 and d["depth_2q"] == 2 and d["n_pairs"] == 2 and d["family_cx"] == 2 and d["family_rz"] == 1 and d["family_h"] == 2

# A QFT's phase ladder is recognized; unrelated angles aren't.
qft = profile("h q[0];\ncp(pi/2) q[1],q[0];\ncp(pi/4) q[2],q[0];\ncp(pi/8) q[3],q[0];", n=4)
assert qft["n_phase_ladder"] == 2, qft["n_phase_ladder"]
assert profile("cp(0.3) q[1],q[0];\ncp(0.7) q[2],q[0];")["n_phase_ladder"] == 0

# The standard Clifford+T Toffoli pattern is recognized on its target.
toffoli = ("h q[2];\ncx q[1],q[2];\ntdg q[2];\ncx q[0],q[2];\nt q[2];\ncx q[1],q[2];\ntdg q[2];\ncx q[0],q[2];"
           "\nt q[1];\nt q[2];\nh q[2];\ncx q[0],q[1];\nt q[0];\ntdg q[1];\ncx q[0],q[1];")
assert profile(toffoli, n=3)["n_toffoli_motif"] == 1
# Regression: a qubit used as a control earlier must still match when it later becomes a target.
busy = "cx q[2],q[0];\n" * 50 + toffoli
assert profile(busy, n=3)["n_toffoli_motif"] == 1
assert profile(toffoli + "\n" + toffoli, n=3)["n_toffoli_motif"] == 2
print("all profile checks passed; 1.2M ops at low work limit took", round(elapsed, 2), "s, coverage", round(again["coverage"], 3))
