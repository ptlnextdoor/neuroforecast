# neuroforecast — certified directed information for neural forecasting

**One question, answered honestly:** *does the causal history of an added channel
`X` carry information about the future of a neural signal `Y`, beyond `Y`'s own
past and known nuisances `Z` — and is that contribution statistically real or
merely underpowered?*

That quantity is the causally-conditioned directed information (transfer entropy)

```
CDI = I( Y_future ; X_past | Z ),     Z = ( Y_past , nuisance ).
```

It is exactly what a claim like *"adding the gastric channel improves prediction
of a future sleep/brain state"* needs to become rigorous: not an AUC or an R²,
but a certified number of **bits**, with a proven floor below which a null is
uninformative.

## Why this exists

Finite-sample directed-information estimators are biased and overstate coupling;
deep models on EEG routinely inflate performance through subject leakage. A
"prediction improved" result can be real signal, extra parameters, shared
nuisance, or an underpowered artifact — and those are not distinguished by
standard metrics. `neuroforecast` distinguishes them, with two commitments:

1. **Cross-fitting** — CDI is the *held-out* predictive-log-likelihood gap of a
   model given `(X, Z)` over a model given `Z` alone. An uninformative channel
   contributes ≈0, not a positive bias. For a fixed model class the estimate is a
   *lower bound* on the true CDI — the honest direction.
2. **A certified detection floor** — a subject-cluster bootstrap gives a one-sided
   95% lower bound; a calibration injects *known* directed information at graded
   strengths and reports the smallest true CDI whose bound clears zero. That is
   the instrument's floor at the given sample size.

Both the linear and the neural estimator are checked against the **closed-form**
linear-Gaussian CDI, so the numbers are trustworthy before any real data.

## Two estimators

| module | estimator | when |
|---|---|---|
| `neuroforecast.linear` | cross-fitted ridge plug-in | linear-Gaussian coupling; instant; the analytic-validated baseline |
| `neuroforecast.neural` | causal TCN + heteroscedastic Gaussian head, cross-fitted | nonlinear coupling on raw causal windows / band-power trajectories; the GPU workload |

The neural estimator encodes the raw causal *trajectory* of `X` with a dilated
causal 1-D convolutional network (a TCN) — preserving the fine-timescale
precursor structure that a single window-average destroys — and reports a
heteroscedastic predictive density so its log-likelihood is a proper score.

## Validation (against analytic ground truth)

```
strength |  true CDI |   est CDI |   95% LCB | certified
    0.00 |   +0.0000 |   -0.0003 |   -0.0004 | no
    0.10 |   +0.0072 |   +0.0054 |   +0.0034 | YES
    0.35 |   +0.0834 |   +0.0845 |   +0.0748 | YES
    0.80 |   +0.3568 |   +0.3507 |   +0.3348 | YES
```

![detection-power](experiments/fig_cdi_detection_power.png)

## Real-data application (Sleep-EDF)

`experiments/run_sleep_edf.py` asks whether the causal EEG **band-power
trajectory** carries directed information about **future delta power** (a
continuous target that is *not* pinned by an oracle current stage — the estimand
where residual directed EEG information can actually live), beyond delta's own
recent past and time-of-night. Certified CDI, linear and neural.

## Layout

```
neuroforecast/          linear.py  neural.py  calibration.py
experiments/            demo_gut_brain.py  run_sleep_edf.py
figures/                plot_detection_power.py
tests/                  test_linear.py  test_neural.py   (validate vs analytic)
run_a100.sh             cluster entry point
```

## Run

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python tests/test_linear.py          # linear CDI vs analytic
python tests/test_neural.py          # neural CDI vs analytic (must match)
python experiments/demo_gut_brain.py # calibration + detection-floor figure
# real data:
python experiments/run_sleep_edf.py --data-dir <sleep-cassette> --neural
# or the whole thing on a cluster:
DATA_DIR=<sleep-cassette> ./run_a100.sh
```

## Design notes

- **Directed, not merely conditional.** `Z` must include `Y`'s own past for CDI
  to be transfer entropy rather than a static conditional MI.
- **Lower bound, stated as such.** A richer model class can only raise the
  estimate; a certified positive is therefore conservative.
- **Cluster bootstrap = the leakage control.** Resampling whole subjects (not
  samples) is what makes the lower bound honest across people.
