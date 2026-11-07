# Quantum Circuit Runtime Predictor (v2, final)

Our entry for the [Quantum Rings Circuit Runtime Prediction Challenge](https://github.com/Quantum-Rings/quantumrings-challenge-iqc-2026): given an OpenQASM circuit and a simulator `threshold`, predict how many seconds the simulation takes.

The automated score is `max(0, 1 − |log10(predicted / actual)|)`, averaged over every run, so being off by 10× scores 0. It's worth 1/3 of the final score. Everyone on the team must be able to explain the whole submission, so please read [How it works](#how-it-works).

**v2 scores:** about 90.3% on random held-out training circuits, about 87.8% on groups of similar circuits, and about 71% on algorithm families the model has never seen. See [How much to trust the scores](#how-much-to-trust-the-scores).

## What's new in v2
| Change | Why | Effect |
|---|---|---|
| **Shot-sampling term** in the physics formula | Runtime of shallow circuits grew with *measured qubits*, not gates: drawing 1024 shots costs about 7 ms per measured qubit, about 35 ms once the final state is entangled | Formula alone: 58% → 69% |
| **Swap-cost term** for long-range gates | Wide circuits with no entanglement cost 13–370 µs per gate depending on how far qubits must move | Included in the line above |
| Six new features: measured qubits, sampling cost, total swap distance, and the largest bond the qubit count allows | Lets the trees use the same findings | +0.3 to +0.7 |
| **Probe shortcut:** a gate between two unentangled qubits is applied directly, without swaps, when the result stays unentangled | The probe used to spend its budget moving qubits on wide circuits | Exact; covers up to 50× more of unentangled wide circuits; score about the same |
| Gates dropped when the probe's budget runs out now mark the run as stopped early | A hidden case where the probe reported finishing when it hadn't | Correctness |
| **Four text-scan features:** share of operations on the busiest 10% of qubits, how unevenly qubits are used, median rotation angle, mean two-qubit rotation angle | Found by studying long circuits the model misjudged | +0.3 on similar circuits, +0.7 on long circuits |
| **Dynamic cost grows with entanglement:** the extra cost per dynamic operation rises about 1.9× for each bit the probe measures | Dynamic circuits run once per shot, and entangled ones run each shot slowly | Formula alone on dynamic circuits: 64% → 78% |
| **Gated neural network:** one-third of the prediction comes from a small neural network, only when the circuit is *familiar*: its gate-set family appears in training **and** it is close to a training circuit in feature space | A network everywhere gained about 0.6 on familiar circuits but lost up to 6 points on new algorithm families | +0.6 on random and similar circuits, unchanged on unseen families |
| **Timeout rule** (from a teammate): runs on large dynamic circuits, and on long, dense, all-to-all rotation circuits above a threshold, are set straight to the cap | The trees under-predict a few families that always time out | Catches 48 of 57 training timeouts (3 false alarms, all over 4,400 s); +0.3 on random and similar circuits, +1.5 on unseen families |
| **Predictions capped at 21,600 s** | The scorer counts a timeout as exactly 21,600 s, and a run that finishes is always faster, so a prediction above the cap can only lose points | Free; rarely matters |

| Test | v1 | v2 |
|---|---:|---:|
| Official `run.py` + `score.py`, 108 held-out circuits | 86.75% | **88.49%** |
| Random circuits held out | 88.2% | **90.3%** |
| Similar circuits held out | 85.5% | **87.8%** |
| Unseen gate-set families | 65.1% | **70.9%** |
| Slowest circuit to featurize (limit 15 s) | 8.4 s | 8.7 s |
| Slowest prediction (limit 15 s) | 0.2 s | 0.3 s |

The random, similar and family scores come from out-of-fold predictions of the full v2 model, including the network and the timeout rule (whose thresholds were set on all training data, so its share of the gain may be slightly optimistic). `analysis/evaluate_model.py` scores trees and physics only (89.4% / 86.9% / 69.2%).

Tried and **not** adopted: probes started partway through the circuit (no gain); a network used everywhere (loses up to 6 points on new families); swap cost growing with qubit count (no gain); calibrating predictions under 1 s (no gain: those errors are per circuit, not a shared bias); probes at bond dimension 64–128 for short circuits (too slow, and our worst short circuit has exactly 4 bits of entanglement even at bond dimension 128 while its runtime still grows 1,600× with threshold).

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

# 3. Check that everything works (known-answer tests and the 15 s timing check, about 3 minutes)
python pipeline.py check

# 4. Train (about 1 minute)
python pipeline.py train

# 5. Try it on one circuit
python pipeline.py predict train-circuits/0071a13b.qasm.zst
```

Step 4 writes `submission-harness/artifacts/runtime_model.joblib`. This file isn't committed because it only loads reliably with the `scikit-learn` version that created it, so each person trains their own copy. Add `--no-network` to train without the neural network (trees and physics only).

## The pipeline
Everything goes through [`pipeline.py`](pipeline.py), one command per step. Each command is a short function that calls the module doing the work, so the file doubles as a map of the code.

| Step | Command | What it does | Code |
|---|---|---|---|
| 1. Parse | `python pipeline.py parse CIRCUIT` | Reads a `.qasm` or `.qasm.zst` file (OpenQASM 2 or 3) into a standard dictionary of operations and summarizes it | `visualize_qasm.parse_qasm` |
| 2. Features | `python pipeline.py features CIRCUIT` | Computes the 163 features for one circuit, exactly as the harness's `featurize` does (must finish within 15 s) | `model.circuit_features` |
| | `python pipeline.py features --all` | Recomputes `analysis/profile_features.csv` for all training circuits (about 5 minutes with 4 workers) | `analysis/profile_features.py` |
| 3. Train | `python pipeline.py train` | Fits the physics formula, trees and gated network; writes `submission-harness/artifacts/runtime_model.joblib` (about 1 minute) | `analysis/train_model.py` |
| 4. Predict | `python pipeline.py predict CIRCUIT` | Predicts one circuit at all five thresholds | `model.RuntimeModel` |
| | `python pipeline.py predict --folder DIR --team NAME` | Predicts a whole folder with the organizers' unmodified `run.py` and writes `submission-harness/submission.csv` | `submission-harness/run_parallel.py` → `run.py` |
| Score | `python pipeline.py score --pred FILE --labels FILE` | Scores a submission with the organizers' `score.py` | `submission-harness/score.py` |
| Check | `python pipeline.py check` | Known-answer tests and the 15 s timing check | `tests/`, `analysis/timing_check.py` |

Step 2 `--all` only needs re-running if the feature code changes, because `analysis/profile_features.csv` ships with the repo. Step 3 needs running once per machine.

## How timeouts are handled
Runs that don't finish within 6 hours are killed and recorded as timeouts. The official `score.py` scores them like this:
- **Run timed out:** the actual value is taken as exactly 21,600 s and the prediction is capped at 21,600 s, so any prediction of 21,600 s or more gets full credit.
- **Run finished:** the prediction is *not* capped. Predicting 50,000 s for a run that took 10,000 s is scored against 50,000 s.

So a prediction above 21,600 s can only lose points, and `model.py` caps every prediction at 21,600 s. In training, timeouts use a target 0.6 decades above the cap (about 86,000 s). This pushes predictions for likely timeouts firmly to the cap instead of just below it, and the cap is applied afterwards.

## Making a submission
The hold-out circuits arrive in the challenge repo during the final hours of the event.

```bash
git -C ../challenge pull
python pipeline.py predict --folder ../challenge/holdout-circuits --team "Your Team Name" --workers 4
```

This writes `submission-harness/submission.csv`, which is sent to the organizers as a DM. It runs the organizers' unmodified `run.py` on 4 batches at once and merges the results; on our test its predictions matched a single `run.py` run exactly. To use `run.py` directly instead: `cd submission-harness && python run.py --team "Your Team Name" --circuits ../../challenge/holdout-circuits`.

Before sending it:
- Check the end of the output for a `WARNING` about the 15 s limit. If `run_parallel.py` warns, try fewer workers.
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
  if the circuit is familiar: blend in a neural network (1/3)
        ↓
  if the timeout rule fires: the 6-hour cap
        ↓
  predicted seconds, at most 21,600 (the timeout cap)
```

- **Structural profile ([`entanglement_profile.py`](entanglement_profile.py)):** reads the circuit as a stream and combines three estimates of entanglement at each cut. A structural bound adds each gate's entangling power, a group bound tracks which qubits have been linked, and a stabilizer tableau follows Clifford gates exactly. It also spots gates that can't entangle, such as `cx` with a `|0⟩` control, and gates that undo each other.
- **Small simulations ([`mps_probe.py`](mps_probe.py)):** a compact matrix-product-state simulation at very low bond dimension. It doesn't try to reproduce the simulator's results. It measures singular values, which structure alone can't predict. A gate between two qubits that are both unentangled is applied directly, without moving qubits, whenever the result stays unentangled. This is exact, and lets the simulation get much further through wide circuits.
- **Prediction ([`submission-harness/model.py`](submission-harness/model.py)):** the physics formula below is fitted to the training data, and trees learn what it misses.

```text
seconds = startup + a × ops + b × ops × (1 + g × probe entropy)   (dynamic circuits only)
        + c × Σ gate crossings × min(2^bits, threshold)^p
        + d × measured qubits × (1 + e × final state entangled)
        + f × total swap distance of long-range gates
```

The last two terms came from studying the worst predictions. Drawing 1024 shots costs about 7 ms per measured qubit, or about 35 ms once the final state is entangled. Long-range gates cost about 1.4 µs per swap. The fitted formula alone rose from 58% to 69% with them, and they raised the full model's score on unseen families by about 4 points.

## What's in this repo

| Path | Purpose |
|---|---|
| `pipeline.py` | **Start here:** parse, features, train, predict, score and check, one command each |
| `submission-harness/model.py` | **The model:** features and prediction |
| `submission-harness/run.py`, `score.py` | Organizers' harness and scorer; don't edit |
| `submission-harness/run_parallel.py` | Runs `run.py` on batches at once, then merges the results |
| `entanglement_profile.py` | Structural entanglement estimate, plus `plot_entanglement_profile()` |
| `mps_probe.py` | Small simulations |
| `visualize_qasm.py` | The QASM parser the predictor uses, plus `parse_qasm()` and `visualize_qasm()` for looking at circuits |
| `analysis/train_model.py` | Trains the final model |
| `analysis/profile_features.py` | Recalculates `analysis/profile_features.csv` |
| `analysis/profile_features.csv` | Saved features for the 532 training circuits |
| `analysis/evaluate_model.py` | Scores the model on held-out circuits; `--full` compares each model change (about 30 minutes) |
| `analysis/timing_check.py` | Times the slowest circuits against the 15 s limit (about 2 minutes) |
| `tests/` | Known-answer checks |

**Changing the model:** if you change `entanglement_profile.py`, `mps_probe.py`, or the feature code in `model.py`, recalculate the features (`python analysis/profile_features.py --workers 4`, about 5 minutes) and retrain. Models and saved features only work if they come from the same code. If you only change model settings, skip that step, retrain, and run `analysis/evaluate_model.py`.

## Looking at a circuit

```python
from entanglement_profile import plot_entanglement_profile
from visualize_qasm import visualize_qasm

visualize_qasm("train-circuits/0071a13b.qasm.zst", out="circuit.png")
plot_entanglement_profile("train-circuits/0071a13b.qasm.zst", out="profile.png")
```

## How much to trust the scores
Never measure accuracy on circuits the model trained on: this model scores about 97% there, which means nothing. We use three tests, from easiest to hardest:

| Test | What's held out | v2 score |
|---|---|---:|
| Random circuits | Individual circuits | 90.3% |
| Similar circuits | Groups of circuits of similar size | 87.8% |
| Unseen families | All circuits sharing a gate set | 70.9% |

`python analysis/evaluate_model.py` reproduces them for trees and physics (5–10 minutes, one seed); the network adds about 0.5 on the first two, and the timeout rule about 0.3 more (1.5 on unseen families). The official `run.py` and `score.py` also gave 88.49% on 108 held-out circuits, which is one split of the data.

For comparison, the organizers report 82.0% cross-validated and 77.9% on the hold-out. Those aren't direct comparisons because splits may differ.

## Where we still lose points
With random circuits held out (the way the hold-out set was drawn), the final model scores 90.3%, so 9.7 points are lost. About 1.4 of those are measurement noise: repeated runs of identical work differ by 3–5%.

| Group (each run counted once) | Points lost |
|---|---:|
| Runs under 1 s (mostly predicted a little too slow) | 2.89 |
| Probe saw under half the circuit, predicted too fast | 1.95 |
| Probe saw under half the circuit, predicted too slow | 1.15 |
| Probe saw most of the circuit, not maxed out | 1.11 |
| Probe saw most of the circuit but maxed out at bond dimension 16 | 0.94 |
| Dynamic circuits (`reset` / `if`, simulated once per shot) | 0.82 |
| Near the 6-hour cap | 0.62 |
| Noisy labels (runtime jumps against its own trend) | 0.24 |

## Known limits
- **Open rules question:** the challenge says to predict runtime "without running it". The small simulations aren't the real simulator, but they do simulate the circuit. Check this with the organizers before relying on them. To train without them, run `python analysis/train_model.py --no-probes`, which is less accurate but purely structural.
- **New algorithm types:** accuracy falls sharply for circuits unlike anything in training.
- **Very large circuits:** tracking stops after a fixed amount of work and the rest is extrapolated. These predictions are less accurate.
- **Slow machines:** timing was measured on one Mac, with a slowest circuit of 8.7 s. Run `python analysis/timing_check.py` to test your own machine. Features stay repeatable unless a machine is slow enough to hit the safety time limits.
- **Internals are inferred:** we don't have documentation of what `threshold` does. Treating it as a bond-dimension cap is our reading of the timing data.

## Credits
`submission-harness/run.py` and `score.py` come from the [challenge repo](https://github.com/Quantum-Rings/quantumrings-challenge-iqc-2026), which is also the source of the circuits and runtime labels.
