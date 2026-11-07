"""Read one small circuit into the standard dictionary, draw it, and open it in a window."""

from pathlib import Path

import matplotlib.pyplot as plt

from visualize_qasm import parse_qasm, visualize_qasm

CIRCUIT = Path(__file__).parent / "train-circuits" / "7337baba.qasm.zst"

circuit = parse_qasm(CIRCUIT)
print(f"{circuit['num_qubits']} qubits, {circuit['num_operations']} operations")
print("First operation:", circuit["operations"][0])

visualize_qasm(circuit, out="7337baba.png")
print("Saved 7337baba.png")

plt.show()
