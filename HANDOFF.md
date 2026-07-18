# A100 handoff — neuroforecast

Self-contained. Everything below runs from the repo root.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch will pick up CUDA on the cluster
export PYTHONPATH=.
```

## What is already validated (CPU, in this repo)
- `tests/test_linear.py` — the linear-Gaussian CDI estimator recovers the
  **analytic** conditional directed information to <0.005 bits, reads ~0 on a
  null channel, certifies a real channel. PASSED.
- `tests/test_neural.py` — the **neural** CDI estimator (causal TCN + held-out
  residual variance) recovers the analytic CDI to <0.01 bits and rejects the
  null. PASSED.
- `experiments/demo_gut_brain.py` — detection-power calibration + figure. The
  estimator's certified floor at n=8000 is ~0.007 bits.

## What the cluster run adds (needs data + GPU)
```bash
DATA_DIR=/path/to/sleep-edfx/sleep-cassette ./run_a100.sh
```
`run_a100.sh` re-runs both validations, regenerates the figure, then runs
`experiments/run_sleep_edf.py --neural`, which produces the **first certified
directed-information number on real EEG**:

    CDI = I( future delta power ; EEG band-power trajectory | delta's own past, time-of-night )

reported in bits with a subject-cluster-bootstrap 95% lower bound, linear and
neural. This is the estimand that is *not* pinned by an oracle current stage, so
it is where residual directed EEG information can actually appear.

## The scientific/collaboration target
The same instrument drops onto **simultaneous gut-brain recordings**: set
`Y_future = future brain/sleep state`, `X_past = gastric channel history`,
`Z = (brain history, circadian)`, `clusters = subject`. It then certifies how
many bits the gastric channel contributes about the future state, conditioned on
the brain's own past — the causal, leakage-controlled version of "adding the gut
improves the forecast," with a floor that proves the contribution is real rather
than underpowered.

## Knobs
- `run_sleep_edf.py`: `--max-subjects`, `--bootstrap`, `--neural`.
- `neural.neural_conditional_directed_information`: `epochs`, `lr`, `embed_dim`,
  `n_splits`, `batch_size`, `device` (auto-detects CUDA).
- Scale: increase `epochs` and `n_splits` on GPU; the estimator is
  embarrassingly parallel over the bootstrap.

## Honest status
Estimators validated against ground truth; real-data numbers are the cluster
step. Neural DI point estimates must be read against the detection-power
calibration (`calibration.py` / the demo) — the floor is what makes a null or a
small positive interpretable. That calibration IS the contribution.
