#!/usr/bin/env python3
"""
visualize_qasm.py  --  read and draw the quantum circuit in a .qasm or .qasm.zst file.

Holds the OpenQASM 2.0/3.0 parser (_Parser) that the predictor uses: entanglement_profile.py and
mps_probe.py both read circuits through it. parse_qasm() turns a file into one standard dictionary
of operations (step 1 in pipeline.py), and visualize_qasm() draws it.

Needs:  pip install matplotlib numpy zstandard
        ('zstandard' is optional if the zstd command-line tool is installed.)

From Python:
    from visualize_qasm import parse_qasm, visualize_qasm

    # Draw a file directly
    fig = visualize_qasm("train-circuits/0071a13b.qasm.zst", out="circuit.png")

    # Or get the standardized dictionary first (see parse_qasm), then draw that
    circuit = parse_qasm("train-circuits/7337baba.qasm.zst")
    fig = visualize_qasm(circuit, out="circuit.png")

From the command line:
    python visualize_qasm.py train-circuits/0071a13b.qasm.zst --out circuit.png
    python visualize_qasm.py train-circuits/0071a13b.qasm.zst --mode diagram --start 500 --max-ops 100

Two views, because most circuits here are far too large for a wire diagram:
  * diagram:  the usual wires-and-gates picture, for up to `max_ops` operations.
  * overview: where multi-qubit gates fall over circuit depth, plus which qubits
              interact. Built from a stream, so very large files use little memory.

The drawing functions are for looking at circuits; only the parser is needed for prediction.
"""

import argparse
import collections
import io
import itertools
import math
import re
import subprocess
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
WIRE = "#c3c2b7"
GATE = "#2a78d6"
NON_UNITARY = "#eb6834"
# Empty cells show the surface, so the lowest count starts dark enough to stand out against it.
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ORANGE_RAMP = ["#f3a585", "#eb6834", "#b4471c", "#6e2a10"]

OVERVIEW_BINS = 600
UNIT_IN = 0.55  # inches per diagram column unit
ROW_IN = 0.5  # inches per qubit wire

# gate name -> (number of control qubits, label drawn on the target)
CONTROLLED = {
    "cx": (1, "x"), "CX": (1, "x"), "cy": (1, "y"), "cz": (1, "z"), "ch": (1, "h"),
    "csx": (1, "sx"), "cp": (1, "p"), "cu1": (1, "u1"), "cu3": (1, "u3"), "cu": (1, "u"),
    "crx": (1, "rx"), "cry": (1, "ry"), "crz": (1, "rz"),
    "ccx": (2, "x"), "ccz": (2, "z"), "c3x": (3, "x"), "c4x": (4, "x"),
}
NOT_DRAWN_AS_GATES = {"measure", "reset", "barrier"}

_IGNORED = {"OPENQASM", "include", "creg", "bit", "int", "uint", "float", "angle",
            "bool", "const", "opaque", "pragma", "defcalgrammar"}
_WORD = re.compile(r"[A-Za-z_]\w*")
_CALL = re.compile(r"([A-Za-z_]\w*)\s*(?:\((.*)\))?\s*(.*)$", re.DOTALL)
_ARG = re.compile(r"([A-Za-z_]\w*)(?:\s*\[\s*(\d+)\s*\])?$")
_GATE_DEF = re.compile(r"gate\s+(\w+)\s*(?:\(([^)]*)\))?\s*([^{]*)\{$", re.DOTALL)
_QREG = re.compile(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]$")
_QUBIT = re.compile(r"qubit\s*(?:\[\s*(\d+)\s*\])?\s*(\w+)$")
_MEASURE = re.compile(r"(?:[\w\[\]\s]+=\s*)?measure\s+([^-]+?)(?:\s*->.*)?$", re.DOTALL)
_IF_PREFIX = re.compile(r"if\s*\(.*?\)\s*(.+)$", re.DOTALL)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_NUMBER = re.compile(r"\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+")
_ANGLE_EXPRESSION = re.compile(r"[0-9eE.+\-*/() pi]+$")

# The same gate can have different names across QASM versions and benchmark suites.
GATE_ALIASES = {"CX": "cx", "cnot": "cx", "U": "u3", "u": "u3", "u1": "p", "phase": "p",
                "cphase": "cp", "cu1": "cp"}


class Op(NamedTuple):
    name: str
    params: str
    qubits: tuple
    conditional: bool = False


def _qubit_label(registers, index):
    for name, (offset, size) in registers.items():
        if offset <= index < offset + size:
            return f"{name}[{index - offset}]"
    return f"q{index}"


class _CircuitInfo(NamedTuple):
    """The circuit details the drawing code needs when working from a dictionary."""
    n_qubits: int
    registers: dict

    def qubit_label(self, index):
        return _qubit_label(self.registers, index)


class _GateDef(NamedTuple):
    formals: list
    index: dict
    body: list


@contextmanager
def _open_lines(path):
    """Yield an iterator of text lines without loading the whole file into memory."""
    path = Path(path)
    if path.suffix != ".zst":
        with open(path, encoding="utf-8", errors="replace") as f:
            yield f
        return
    try:
        import zstandard
    except ImportError:
        proc = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, errors="replace")
        try:
            yield proc.stdout
        finally:
            proc.stdout.close()
            proc.kill()
            proc.wait()
        return
    with open(path, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as raw:
        yield io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def _statements(lines):
    """Split source lines into statements. Block openers keep a trailing '{'; closers are '}'."""
    buffer = ""
    in_comment = False
    for line in lines:
        if in_comment:
            end = line.find("*/")
            if end < 0:
                continue
            line = line[end + 2:]
            in_comment = False
        if "/*" in line:
            line = _BLOCK_COMMENT.sub(" ", line)
            start = line.find("/*")
            if start >= 0:
                line, in_comment = line[:start], True
        if "//" in line:
            line = line.split("//", 1)[0]
        text = line.strip()
        if not text:
            continue
        if not buffer and text[-1] == ";" and text.count(";") == 1 and "{" not in text and "}" not in text:
            yield text[:-1].strip()
            continue
        for piece in re.split(r"([;{}])", text):
            if piece == ";":
                if buffer.strip():
                    yield buffer.strip()
                buffer = ""
            elif piece == "{":
                yield (buffer + " {").strip()
                buffer = ""
            elif piece == "}":
                if buffer.strip():
                    yield buffer.strip()
                buffer = ""
                yield "}"
            elif piece:
                buffer += piece + " "


class _Parser:
    """Streaming OpenQASM 2/3 reader: yields one Op per gate, measure, reset or barrier."""

    def __init__(self, expand_gates=True):
        self.expand_gates = expand_gates
        self.version = None
        self.registers = {}  # name -> (offset, size)
        self.n_qubits = 0
        self.gates = {}
        self.skipped = collections.Counter()
        self._resolved = {}

    def qubit_label(self, index):
        return _qubit_label(self.registers, index)

    def _declare(self, name, size):
        self.registers[name] = (self.n_qubits, size)
        self.n_qubits += size

    def _resolve(self, arg):
        """'q[3]' -> (qubit,), 'q' -> every qubit in the register, unknown -> None."""
        cached = self._resolved.get(arg)
        if cached is not None:
            return cached
        match = _ARG.match(arg.strip())
        register = self.registers.get(match.group(1)) if match else None
        if register is None:
            return None
        offset, size = register
        if match.group(2) is None:
            result = tuple(range(offset, offset + size))
        elif int(match.group(2)) < size:
            result = (offset + int(match.group(2)),)
        else:
            return None
        self._resolved[arg] = result
        return result

    @staticmethod
    def _parse_call(statement):
        modifiers = ""
        if "@" in statement:
            modifiers, statement = statement.rsplit("@", 1)
            modifiers = modifiers.strip() + " @ "
        match = _CALL.match(statement.strip())
        if not match or not match.group(3):
            return None
        args = [a.strip() for a in match.group(3).split(",")]
        return modifiers + match.group(1), match.group(2) or "", args

    def _emit(self, name, params, qubits, conditional):
        gate = self.gates.get(name) if self.expand_gates else None
        if gate is None or len(gate.index) != len(qubits):
            yield Op(name, params, qubits, conditional)
            return
        substitutions = dict(zip(gate.formals, params.split(","))) if gate.formals else None
        for inner_name, inner_params, indices in gate.body:
            if substitutions and inner_params:
                inner_params = _WORD.sub(lambda m: substitutions.get(m.group(), m.group()), inner_params)
            yield from self._emit(inner_name, inner_params, tuple(qubits[i] for i in indices), conditional)

    def _add_to_gate_body(self, gate, statement):
        call = self._parse_call(statement)
        try:
            gate.body.append((call[0], call[1], [gate.index[a] for a in call[2]]))
        except (TypeError, KeyError):
            self.skipped[_first_word(statement)] += 1

    def _statement_ops(self, statement, conditional):
        if statement.startswith("if") and _IF_PREFIX.match(statement):
            statement = _IF_PREFIX.match(statement).group(1)
            conditional = True
        if "measure" in statement:
            match = _MEASURE.match(statement)
            if match:
                qubits = self._resolve(match.group(1))
                if qubits is None:
                    self.skipped["measure"] += 1
                    return
                for qubit in qubits:
                    yield Op("measure", "", (qubit,), conditional)
                return
        call = self._parse_call(statement)
        resolved = [self._resolve(a) for a in call[2]] if call else [None]
        if any(r is None for r in resolved):
            self.skipped[_first_word(statement)] += 1
            return
        name, params = call[0], call[1]
        if name == "barrier":
            yield Op(name, "", tuple(q for r in resolved for q in r), conditional)
            return
        width = max(len(r) for r in resolved)
        if width == 1:
            yield from self._emit(name, params, tuple(r[0] for r in resolved), conditional)
        elif all(len(r) in (1, width) for r in resolved):  # register broadcast, e.g. `h q;`
            for i in range(width):
                qubits = tuple(r[i] if len(r) > 1 else r[0] for r in resolved)
                yield from self._emit(name, params, qubits, conditional)
        else:
            self.skipped[name] += 1

    def ops(self, lines):
        blocks = []
        conditional_depth = 0
        gate = None
        for statement in _statements(lines):
            if statement == "}":
                if blocks:
                    kind = blocks.pop()
                    if kind == "if":
                        conditional_depth -= 1
                    elif kind == "gate":
                        gate = None
                continue
            if gate is not None:
                if statement.endswith("{"):
                    blocks.append("other")
                else:
                    self._add_to_gate_body(gate, statement)
                continue
            word = _first_word(statement)
            if word == "OPENQASM":
                self.version = statement.split()[-1]
                continue
            if word in _IGNORED:
                continue
            if word == "gate":
                match = _GATE_DEF.match(statement)
                if match:
                    formals = [p.strip() for p in (match.group(2) or "").split(",") if p.strip()]
                    qargs = [q.strip() for q in match.group(3).split(",") if q.strip()]
                    gate = _GateDef(formals, {q: i for i, q in enumerate(qargs)}, [])
                    self.gates[match.group(1)] = gate
                    blocks.append("gate")
                else:
                    self.skipped["gate"] += 1
                    blocks.append("other")
                continue
            if word == "qreg" or word == "qubit":
                match = _QREG.match(statement) if word == "qreg" else _QUBIT.match(statement)
                if not match:
                    self.skipped[word] += 1
                elif word == "qreg":
                    self._declare(match.group(1), int(match.group(2)))
                else:
                    self._declare(match.group(2), int(match.group(1) or 1))
                continue
            if statement.endswith("{"):
                if word in ("if", "else"):
                    blocks.append("if")
                    conditional_depth += 1
                else:
                    blocks.append("other")
                    self.skipped[word] += 1
                continue
            yield from self._statement_ops(statement, conditional_depth > 0)


def _first_word(statement):
    match = _WORD.match(statement)
    return match.group() if match else statement[:12]


def _short_params(params):
    text = _NUMBER.sub(lambda m: f"{float(m.group()):.2g}", params).replace("pi", "π").replace(" ", "")
    return text if len(text) <= 14 else text[:13] + "…"


def _human(count):
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 10_000:
        return f"{count / 1000:.0f}k"
    return f"{count:,}"


def _title(fig, name, subtitle, height_in):
    fig.text(0.1 / fig.get_figwidth(), 1 - 0.28 / height_in, name, color=INK, fontsize=13,
             fontweight="bold", va="center")
    fig.text(0.1 / fig.get_figwidth(), 1 - 0.54 / height_in, subtitle, color=SECONDARY, fontsize=9.5, va="center")


# ---------------------------------------------------------------------- #
# Diagram view                                                            #
# ---------------------------------------------------------------------- #
def _gate_box(ax, x, y_low, y_high, width, name, params, color, conditional):
    height = y_high - y_low + 0.76
    ax.add_patch(FancyBboxPatch((x - width / 2, y_low - 0.38), width, height,
                                boxstyle="round,pad=0,rounding_size=0.06", mutation_aspect=0.9,
                                facecolor=color, edgecolor="none", zorder=3))
    middle = (y_low + y_high) / 2
    if params:
        ax.text(x, middle - 0.11, name, color="white", ha="center", va="center", fontsize=9.5, zorder=4)
        ax.text(x, middle + 0.17, params, color="white", ha="center", va="center", fontsize=6.5, zorder=4)
    else:
        ax.text(x, middle, name, color="white", ha="center", va="center", fontsize=9.5, zorder=4)
    if conditional:
        ax.text(x, y_low - 0.45, "if", color=SECONDARY, ha="center", va="bottom", fontsize=7,
                style="italic", zorder=4)


def _draw_op(ax, op, x, rows, width):
    ys = [rows[q] for q in op.qubits]
    low, high = min(ys), max(ys)
    params = _short_params(op.params)
    if op.name == "barrier":
        ax.plot([x, x], [low - 0.4, high + 0.4], color=MUTED, linewidth=1.5, zorder=2)
    elif op.name == "measure":
        _gate_box(ax, x, low, low, width, "M", "", NON_UNITARY, op.conditional)
    elif op.name == "reset":
        _gate_box(ax, x, low, low, width, "|0⟩", "", NON_UNITARY, op.conditional)
    elif op.name == "swap" and len(ys) == 2:
        ax.plot([x, x], [low, high], color=GATE, linewidth=1.5, zorder=2)
        ax.plot([x, x], ys, color=GATE, linestyle="none", marker="x", markersize=8, markeredgewidth=2, zorder=3)
    elif op.name in CONTROLLED and len(ys) == CONTROLLED[op.name][0] + 1:
        n_controls, target = CONTROLLED[op.name]
        ax.plot([x, x], [low, high], color=GATE, linewidth=1.5, zorder=2)
        dots = ys if target == "z" else ys[:n_controls]
        ax.plot([x] * len(dots), dots, color=GATE, linestyle="none", marker="o", markersize=6.5, zorder=3)
        if target == "x":
            radius = 0.19  # matches the 15 pt marker at UNIT_IN inches per column
            ax.plot([x], [ys[-1]], color=GATE, linestyle="none", marker="o", markersize=15,
                    markerfacecolor=SURFACE, markeredgewidth=1.5, zorder=3)
            ax.plot([x, x], [ys[-1] - 0.21, ys[-1] + 0.21], color=GATE, linewidth=1.5, zorder=4)
            ax.plot([x - radius, x + radius], [ys[-1], ys[-1]], color=GATE, linewidth=1.5, zorder=4)
        elif target != "z":
            _gate_box(ax, x, ys[-1], ys[-1], width, target, params, GATE, False)
        if op.conditional:
            ax.text(x, low - 0.45, "if", color=SECONDARY, ha="center", va="bottom", fontsize=7,
                    style="italic", zorder=4)
    else:
        _gate_box(ax, x, low, high, width, op.name, params, GATE, op.conditional)
        if len(ys) > 1:
            ax.plot([x - width / 2 + 0.09] * len(ys), ys, color="white", linestyle="none", marker="o",
                    markersize=3.5, zorder=4)


def _draw_diagram(ops, parser, name, start, truncated):
    if not ops:
        raise ValueError(f"No operations to draw at start={start}.")
    # Multi-qubit ops reserve every wire they cross so that gates never overlap.
    next_free = collections.defaultdict(int)
    placed = []
    after_measure = 0
    for op in ops:
        low, high = min(op.qubits), max(op.qubits)
        column = max(next_free[q] for q in range(low, high + 1))
        if op.conditional:
            column = max(column, after_measure)
        if op.name == "measure":
            after_measure = max(after_measure, column + 1)
        for q in range(low, high + 1):
            next_free[q] = column + 1
        placed.append((op, column))

    # Keep every wire on small circuits; drop idle wires when a 260-qubit register would dominate.
    used = sorted({q for op in ops for q in range(min(op.qubits), max(op.qubits) + 1)})
    shown = list(range(parser.n_qubits)) if parser.n_qubits <= 30 else used
    rows = {q: i for i, q in enumerate(shown)}

    widths = collections.defaultdict(lambda: 0.8)
    for op, column in placed:
        if op.name in NOT_DRAWN_AS_GATES:
            continue
        # Gate names use a larger font than parameters, so they take more width per character.
        text_width = max(0.16 * len(op.name), 0.1 * len(_short_params(op.params)))
        widths[column] = max(widths[column], text_width + 0.25)
    n_columns = max(column for _, column in placed) + 1
    edges = np.concatenate([[0.0], np.cumsum([widths[c] + 0.2 for c in range(n_columns)])])
    centers = (edges[:-1] + edges[1:]) / 2
    plot_width = edges[-1]

    labels = {q: parser.qubit_label(q) for q in shown}
    margin = max(len(label) for label in labels.values()) * 0.14 + 0.35
    title_in = 0.85
    fig_width = min(400.0, max(7.0, (plot_width + margin + 0.3) * UNIT_IN))
    fig_height = min(400.0, len(shown) * ROW_IN + title_in + 0.2)

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1 - title_in / fig_height])
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_xlim(-margin, -margin + fig_width / UNIT_IN)
    ax.set_ylim(len(shown) - 0.5 + 0.2 / ROW_IN, -0.5)

    for q, row in rows.items():
        ax.plot([-0.1, plot_width + 0.1], [row, row], color=WIRE, linewidth=1, zorder=1)
        ax.text(-0.2, row, labels[q], color=SECONDARY, ha="right", va="center", fontsize=9)
    for op, column in placed:
        _draw_op(ax, op, centers[column], rows, widths[column])

    n_gates = sum(op.name != "barrier" for op in ops)
    window = f"operations {start:,}–{start + len(ops) - 1:,} of more" if truncated or start else f"{n_gates:,} operations"
    qubits = f"{parser.n_qubits} qubits" if len(shown) == parser.n_qubits else \
        f"{len(shown)} of {parser.n_qubits} qubits shown"
    _title(fig, name, f"{qubits}  ·  {window}", fig_height)

    if any(op.name in ("measure", "reset") for op in ops):
        handles = [Patch(facecolor=GATE, label="Gate"), Patch(facecolor=NON_UNITARY, label="Measure / reset")]
        fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=9, ncols=2,
                   labelcolor=SECONDARY, handlelength=1, handleheight=1)
    return fig


# ---------------------------------------------------------------------- #
# Overview view                                                           #
# ---------------------------------------------------------------------- #
def _collect_overview(ops):
    """Stream ops into fixed-size summaries. Bins double in width as the circuit gets deeper."""
    multi = collections.defaultdict(lambda: [0] * OVERVIEW_BINS)
    every = collections.defaultdict(lambda: [0] * OVERVIEW_BINS)
    pairs = collections.Counter()
    names = collections.Counter()
    level = collections.defaultdict(int)
    bin_size = 1
    depth = 0
    n_ops = 0
    n_multi = 0
    n_conditional = 0
    after_measure = 0

    def merge(table):
        for q, row in table.items():
            table[q] = [row[i] + row[i + 1] for i in range(0, OVERVIEW_BINS, 2)] + [0] * (OVERVIEW_BINS // 2)

    for op in ops:
        if op.name == "barrier":
            continue
        n_ops += 1
        names[GATE_ALIASES.get(op.name, op.name)] += 1
        qubits = op.qubits
        column = max(level[q] for q in qubits)
        # A conditional gate has to wait for a measurement result. The bit it reads isn't
        # tracked, so this assumes the most recent measurement, which may overstate depth.
        if op.conditional:
            n_conditional += 1
            column = max(column, after_measure)
        if op.name == "measure":
            after_measure = max(after_measure, column + 1)
        for q in qubits:
            level[q] = column + 1
        if column >= depth:
            depth = column + 1
        while column // bin_size >= OVERVIEW_BINS:
            merge(multi)
            merge(every)
            bin_size *= 2
        slot = column // bin_size
        is_multi = len(qubits) > 1
        for q in qubits:
            every[q][slot] += 1
            if is_multi:
                multi[q][slot] += 1
        if is_multi:
            n_multi += 1
            if len(qubits) == 2:
                a, b = qubits
                pairs[(a, b) if a < b else (b, a)] += 1
            else:
                for a, b in itertools.combinations(sorted(qubits), 2):
                    pairs[(a, b)] += 1
    return multi, every, pairs, names, bin_size, depth, n_ops, n_multi, n_conditional


def _to_matrix(table, n_qubits, n_bins):
    matrix = np.zeros((n_qubits, n_bins))
    for q, row in table.items():
        matrix[q] = row[:n_bins]
    return np.ma.masked_equal(matrix, 0)


def _heatmap_norm(matrix):
    return LogNorm(vmin=1, vmax=max(2.0, float(matrix.max())))


def _colorbar(fig, image, ax, label, pad):
    bar = fig.colorbar(image, ax=ax, pad=pad, fraction=0.04)
    top = image.norm.vmax
    decades = [10 ** k for k in range(int(np.log10(top)) + 1) if 10 ** k * 1.5 < top]
    ticks = decades + [top]
    bar.set_ticks(ticks, labels=[f"{tick:,.0f}" for tick in ticks])
    bar.minorticks_off()
    bar.ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    bar.outline.set_edgecolor(WIRE)
    bar.set_label(label, color=SECONDARY, fontsize=8.5)


def _draw_overview(ops, parser, name):
    multi, every, pairs, names, bin_size, depth, n_ops, n_multi, n_conditional = _collect_overview(ops)
    if n_ops == 0:
        raise ValueError("The circuit contains no operations.")
    n_qubits = parser.n_qubits
    has_multi = n_multi > 0
    n_bins = max(1, -(-depth // bin_size))
    density = _to_matrix(multi if has_multi else every, n_qubits, n_bins)

    fig_height = 5.6
    fig = plt.figure(figsize=(13.5, fig_height), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.4, 1], left=0.05, right=0.925, top=0.80, bottom=0.11, wspace=0.2)
    left, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    blues = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    oranges = LinearSegmentedColormap.from_list("oranges", ORANGE_RAMP)
    # Qubit 0 sits at the top, matching the wire diagram.
    image = left.imshow(density, aspect="auto", interpolation="nearest", cmap=blues,
                        norm=_heatmap_norm(density), extent=(0, n_bins * bin_size, n_qubits - 0.5, -0.5))
    left.xaxis.set_major_formatter(FuncFormatter(lambda value, _: _human(int(value))))
    left.set_xlabel("Circuit depth (layers)", color=SECONDARY, fontsize=9.5)
    left.set_ylabel("Qubit index", color=SECONDARY, fontsize=9.5)
    kind = "Multi-qubit gates" if has_multi else "Gates (no multi-qubit gates in this circuit)"
    left.set_title(f"{kind} per qubit over depth", color=INK, fontsize=10.5, loc="left")
    span = "layer" if bin_size == 1 else f"{bin_size:,} layers"
    _colorbar(fig, image, left, f"Gates per {span}", pad=0.012)

    if pairs:
        matrix = np.zeros((n_qubits, n_qubits))
        for (a, b), count in pairs.items():
            matrix[a, b] = matrix[b, a] = count
        matrix = np.ma.masked_equal(matrix, 0)
        image = right.imshow(matrix, interpolation="nearest", cmap=oranges, norm=_heatmap_norm(matrix))
        _colorbar(fig, image, right, "Gates on this pair", pad=0.03)
    else:
        right.set_xlim(-0.5, n_qubits - 0.5)
        right.set_ylim(n_qubits - 0.5, -0.5)
        right.set_aspect("equal")
        message = "No multi-qubit gates"
        if n_conditional:
            message += "\nQubits are linked only through\nmeasurement-conditioned gates"
        right.text(0.5, 0.5, message, transform=right.transAxes, ha="center", va="center",
                   color=SECONDARY, fontsize=10)
    right.set_title("Which qubits interact", color=INK, fontsize=10.5, loc="left")
    right.set_xlabel("Qubit index", color=SECONDARY, fontsize=9.5)
    right.set_ylabel("Qubit index", color=SECONDARY, fontsize=9.5)

    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for spine in ax.spines.values():
            spine.set_color(WIRE)
            spine.set_linewidth(0.8)

    top = "  ".join(f"{gate} {count / n_ops:.0%}" for gate, count in names.most_common(6))
    conditional = f"{_human(n_conditional)} conditional gates  ·  " if n_conditional else ""
    subtitle = (f"{n_qubits} qubits  ·  {_human(n_ops)} operations  ·  depth {_human(depth)}  ·  "
                f"{_human(n_multi)} multi-qubit gates  ·  {conditional}{top}")
    _title(fig, name, subtitle, fig_height)
    return fig


# ---------------------------------------------------------------------- #
# Standardized dictionary format                                         #
# ---------------------------------------------------------------------- #
def _angle_value(text):
    """Evaluate an angle such as 'pi/4'. Returns None if it refers to a gate's own parameter."""
    if not _ANGLE_EXPRESSION.match(text):
        return None
    try:
        # The regex limits this to arithmetic on numbers and pi.
        return float(eval(text, {"__builtins__": {}}, {"pi": math.pi}))  # noqa: S307
    except (NameError, SyntaxError, ZeroDivisionError, TypeError):
        return None


def _op_to_dict(op):
    params = [p.strip() for p in op.params.split(",")] if op.params else []
    return {
        "name": GATE_ALIASES.get(op.name, op.name),
        "qubits": list(op.qubits),
        "params": params,
        "angles": [_angle_value(p) for p in params],
        "conditional": op.conditional,
    }


def _dict_to_op(op):
    return Op(op["name"], ",".join(op["params"]), tuple(op["qubits"]), op["conditional"])


def _warn_skipped(name, skipped):
    if skipped:
        warnings.warn(f"{name}: left out statements the parser could not read: {dict(skipped)}")


def _read_circuit(path, expand_gates, max_ops):
    path = Path(path)
    parser = _Parser(expand_gates)
    with _open_lines(path) as lines:
        limit = None if max_ops is None else max_ops + 1
        operations = [_op_to_dict(op) for op in itertools.islice(parser.ops(lines), limit)]
    truncated = max_ops is not None and len(operations) > max_ops
    if truncated:
        operations.pop()
    return {
        "source": path.name,
        "qasm_version": parser.version,
        "num_qubits": parser.n_qubits,
        "registers": {name: {"offset": offset, "size": size}
                      for name, (offset, size) in parser.registers.items()},
        "custom_gates": sorted(parser.gates),
        "num_operations": len(operations),
        "truncated": truncated,
        "skipped": dict(parser.skipped),
        "operations": operations,
    }


def parse_qasm(path, expand_gates=True, max_ops=1_000_000):
    """
    Read a .qasm or .qasm.zst file into one standard dictionary, for OpenQASM 2.0 or 3.0.

    {
      "source": "7337baba.qasm.zst",
      "qasm_version": "2.0",
      "num_qubits": 2,                   # all quantum registers, numbered as one list
      "registers": {"q": {"offset": 0, "size": 2}},
      "custom_gates": [],                # names of `gate` definitions in the file
      "num_operations": 7,
      "truncated": False,                # True if max_ops cut the circuit short
      "skipped": {},                     # statements the parser could not read, by keyword
      "operations": [
        {"name": "rz",                   # standard name: cnot and CX become cx, u1 becomes p
         "qubits": [0],                  # positions in the combined qubit list
         "params": ["pi/4"],             # angles as written in the file
         "angles": [0.785...],           # the same angles in radians
         "conditional": False},          # True inside an `if`
      ],
    }

    Operations are "measure", "reset" and "barrier" as well as gates. Classical bits aren't
    kept: "conditional" says a gate depends on a measurement, not which one. Gates that use
    a `gate` definition's own parameters can have None in "angles".

    expand_gates  True replaces custom gates with the gates inside them.
    max_ops       stop after this many operations. A very large circuit can need tens of
                  gigabytes as dictionaries, so None (no limit) needs care. A truncated
                  result sets "truncated" and warns.
    """
    circuit = _read_circuit(path, expand_gates, max_ops)
    name = circuit["source"]
    if circuit["truncated"]:
        warnings.warn(f"{name}: stopped at {max_ops:,} operations. Pass max_ops=None to read all of them.")
    _warn_skipped(name, circuit["skipped"])
    return circuit


# ---------------------------------------------------------------------- #
# Public function                                                        #
# ---------------------------------------------------------------------- #
def visualize_qasm(circuit, out=None, mode="auto", start=0, max_ops=200, expand_gates=True):
    """
    Draw a circuit and return a matplotlib Figure.

    circuit       a circuit file (OpenQASM 2.0 or 3.0, plain or zstd-compressed), or a
                  dictionary from parse_qasm.
    out           optional image path to save to, such as "circuit.png".
    mode          "auto" draws a diagram if the circuit has at most max_ops operations,
                  and an overview otherwise. "diagram" and "overview" force a view.
    start, max_ops  which operations a diagram shows. Setting start > 0 implies a diagram.
    expand_gates  True replaces custom `gate` definitions with the gates inside them.
                  False draws each custom gate as one labelled box. A dictionary already
                  made this choice in parse_qasm, so this is ignored for one.

    Files are read through the dictionary format for diagrams. Overviews of files are
    streamed instead, because the largest circuits won't fit in memory as dictionaries.
    """
    if mode not in ("auto", "diagram", "overview"):
        raise ValueError(f"mode must be auto, diagram or overview, not {mode!r}")
    if start < 0 or max_ops < 1:
        raise ValueError("start must be >= 0 and max_ops must be >= 1")

    given_dict = isinstance(circuit, dict)
    parsed = None
    if given_dict:
        parsed = circuit
    elif mode != "overview":
        parsed = _read_circuit(circuit, expand_gates, start + max_ops)

    more_ops = parsed is not None and (parsed["truncated"] or len(parsed["operations"]) > start + max_ops)
    source = parsed["source"] if parsed else Path(circuit).name
    name = source.split(".")[0]

    if mode == "overview" or (mode == "auto" and more_ops and start == 0):
        if given_dict:
            if parsed["truncated"]:
                warnings.warn(f"{name}: this dictionary was truncated, so the overview covers only part of the circuit.")
            info = _CircuitInfo(parsed["num_qubits"], {})
            fig = _draw_overview(map(_dict_to_op, parsed["operations"]), info, name)
            _warn_skipped(name, parsed["skipped"])
        else:
            parser = _Parser(expand_gates)
            with _open_lines(circuit) as lines:
                fig = _draw_overview(parser.ops(lines), parser, name)
            _warn_skipped(name, parser.skipped)
    else:
        registers = {reg: (spec["offset"], spec["size"]) for reg, spec in parsed["registers"].items()}
        info = _CircuitInfo(parsed["num_qubits"], registers)
        window = [_dict_to_op(op) for op in parsed["operations"][start:start + max_ops]]
        fig = _draw_diagram(window, info, name, start, more_ops)
        _warn_skipped(name, parsed["skipped"])

    if out is not None:
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    return fig


def main():
    ap = argparse.ArgumentParser(description="Draw the quantum circuit in a .qasm or .qasm.zst file.")
    ap.add_argument("path")
    ap.add_argument("--out", help="Image file to save. Without this, opens a window.")
    ap.add_argument("--mode", choices=["auto", "diagram", "overview"], default="auto")
    ap.add_argument("--start", type=int, default=0, help="First operation to show in a diagram.")
    ap.add_argument("--max-ops", type=int, default=200, help="Most operations to show in a diagram.")
    ap.add_argument("--no-expand", action="store_true", help="Draw custom gates as single boxes.")
    args = ap.parse_args()
    visualize_qasm(args.path, out=args.out, mode=args.mode, start=args.start, max_ops=args.max_ops,
                   expand_gates=not args.no_expand)
    if args.out:
        print(f"Saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
