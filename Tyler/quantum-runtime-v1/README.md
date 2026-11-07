# Quantum Circuit Runtime Predictor (v1)

Our entry for the [Quantum Rings Circuit Runtime Prediction Challenge](https://github.com/Quantum-Rings/quantumrings-challenge-iqc-2026): given an OpenQASM circuit and a simulator `threshold`, predict how many seconds the simulation takes.

The automated score is `max(0, 1 − |log10(predicted / actual)|)`, averaged over every run, so being off by 10× scores 0. It's worth 1/3 of the final score. Everyone on the team must be able to explain the whole submission, so please read [How it works](#how-it-works).

**v1 scores (preliminary):** about 88% on random held-out training circuits, about 85.5% on groups of similar circuits, and about 65% on algorithm families the model has never seen. See [How much to trust the scores](#how-much-to-trust-the-scores).

## Quick start
Tested with Python 3.13. The included `analysis/profile_features.csv` holds features for the training circuits, so training doesn't need them recalculated.

```bash
# 1. Install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Get the training data from the challenge repo (it isn't stored here)
git clone git@github.com:Quantum-Rings/quantumrings-challenge-iqc-2026.git ../challenge
cp ../challenge/train-runtime-data.csv .
cp -r ../challenge/train-circuits .

# 3. Check that everything works
python tests/test_profile.py
python tests/test_probe.py

# 4. Train (under a minute)
python analysis/train_model.py
```

Step 4 writes `submission-harness/artifacts/runtime_model.joblib`. This file isn't committed because it only loads reliably with the `scikit-learn` version that created it, so each person trains their own copy.

## Making a submission
The hold-out circuits arrive in the challenge repo during the final hours of the event.

```bash
git -C ../challenge pull
cd submission-harness
python run.py --team "Your Team Name" --circuits ../../challenge/holdout-circuits
```

This writes `submission.csv`, which is sent to the organizers as a DM. Before sending it:
- Check the end of the output for a `WARNING` about the 15 s limit.
- Close other heavy programs while it runs, because the file records how long each circuit took.
- Train on all circuits first. `analysis/train_model.py` does this by default.

## How it works
Runtime depends far less on line count or qubit count than on **how much entanglement a circuit builds**, capped by `threshold`. Circuits with mid-circuit measurement or `reset` are a separate, much slower case.

```text
QASM text
  ├─ Structural profile: estimates entanglement at each cut between neighboring qubits
  ├─ Small simulations: measure real entanglement at bond dimensions 8 and 16
  └─ Counts: gate mix, depth, interaction graph, recognized algorithm blocks
        ↓
  model inputs for each circuit and threshold
        ↓
  average, in log space, of:
    ├─ trees predicting runtime directly
    └─ a physics formula + trees predicting its error
        ↓
  predicted seconds
```

- **Structural profile ([`entanglement_profile.py`](entanglement_profile.py)):** reads the circuit as a stream and combines three estimates of entanglement at each cut. A structural bound adds each gate's entangling power, a group bound tracks which qubits have been linked, and a stabilizer tableau follows Clifford gates exactly. It also spots gates that can't entangle, such as `cx` with a `|0⟩` control, and gates that undo each other.
- **Small simulations ([`mps_probe.py`](mps_probe.py)):** a compact matrix-product-state simulation at very low bond dimension. It doesn't try to reproduce the simulator's results. It measures singular values, which structure alone can't predict.
- **Prediction ([`submission-harness/model.py`](submission-harness/model.py)):** the physics formula below is fitted to the training data, and trees learn what it misses.

```text
seconds = startup + a × ops + b × ops (dynamic circuits only)
        + c × Σ gate crossings × min(2^bits, threshold)^p
```

## What's in this repo

| Path | Purpose |
|---|---|
| `submission-harness/model.py` | **The model:** features and prediction |
| `submission-harness/run.py`, `score.py` | Organizers' harness and scorer; don't edit |
| `entanglement_profile.py` | Structural entanglement estimate, plus `plot_entanglement_profile()` |
| `mps_probe.py` | Small simulations |
| `visualize_qasm.py` | QASM parser, plus `visualize_qasm()` and `parse_qasm()` for looking at circuits |
| `visualize_example.py` | Short example of reading and drawing a circuit |
| `analysis/train_model.py` | Trains the final model |
| `analysis/profile_features.py` | Recalculates `analysis/profile_features.csv` |
| `analysis/profile_features.csv` | Saved features for the 532 training circuits |
| `analysis/evaluate_model.py` | Measures each model change on held-out circuits (slow) |
| `tests/` | Known-answer checks |

**Changing the model:** if you change `entanglement_profile.py`, `mps_probe.py`, or the feature code in `model.py`, recalculate the features (`python analysis/profile_features.py`, about 15 minutes) and retrain. Models and saved features only work if they come from the same code.

## Looking at a circuit

```python
from entanglement_profile import plot_entanglement_profile
from visualize_qasm import visualize_qasm

visualize_qasm("train-circuits/0071a13b.qasm.zst", out="circuit.png")
plot_entanglement_profile("train-circuits/0071a13b.qasm.zst", out="profile.png")
```

## How much to trust the scores
Never measure accuracy on circuits the model trained on: this model scores about 97% there, which means nothing. `analysis/evaluate_model.py` uses three tests, from easiest to hardest:

| Test | What's held out | v1 score |
|---|---|---:|
| Random circuits | Individual circuits | about 88.4% |
| Similar circuits | Groups of circuits of similar size | about 85.5% |
| Unseen families | All circuits sharing a gate set | about 65.0% |

These scores are preliminary: they were measured with larger simulation budgets than v1 uses, so v1 may score slightly lower. To confirm them, run `python analysis/evaluate_model.py` (about 30 minutes).

For comparison, the organizers report 82.0% cross-validated and 77.9% on the hold-out. Those aren't direct comparisons because splits may differ.

## Known limits
- **Open rules question:** the challenge says to predict runtime "without running it". The small simulations aren't the real simulator, but they do simulate the circuit. Check this with the organizers before relying on them. To train without them, run `python analysis/train_model.py --no-probes`, which is less accurate but purely structural.
- **New algorithm types:** accuracy falls sharply for circuits unlike anything in training.
- **Very large circuits:** tracking stops after a fixed amount of work and the rest is extrapolated. These predictions are less accurate.
- **Slow machines:** timing was measured on one Mac, with a slowest circuit of about 11 s while other jobs were running. Features stay repeatable unless a machine is slow enough to hit the safety time limits.
- **Internals are inferred:** we don't have documentation of what `threshold` does. Treating it as a bond-dimension cap is our reading of the timing data.

## Credits
`submission-harness/run.py` and `score.py` come from the [challenge repo](https://github.com/Quantum-Rings/quantumrings-challenge-iqc-2026), which is also the source of the circuits and runtime labels.
