# neuroforecast

**Certified causally-conditioned directed-information graphs for multi-organ
neural coupling.** A finite-sample, leakage-controlled implementation of directed
information graphs [Quinn, Kiyavash & Coleman, *IEEE TIT* 2015] that separates
**direct** from **mediated** coupling and certifies each edge with a bootstrap
lower bound and a detection-power floor.

---

## Why this exists

Recent work on simultaneous stomach–brain electrophysiology in human sleep
(Rao et al., "Dynamic stomach-brain electrical coupling during human sleep," 2026)
states its own methodological gap directly:

> *"These analyses are inherently correlational, precluding inference about
> directionality. Future work can employ passive causal inference methods on
> simultaneously recorded waveforms … such as Granger causality or directed
> information."*

The coupling results there are cross-correlation and phase-amplitude coupling.
This repository is that named next step — directed information — built to be
**finite-sample-certified** and **multi-organ**, so a directed edge comes with an
honest statement of whether it is real or merely underpowered.

## The estimand

For simultaneously recorded signals $\{X_1,\dots,X_m\}$, the directed edge
$i \to j$ is the causally-conditioned directed information

$$\mathrm{DI}(i \to j \,\|\, \text{rest}) \;=\; I\big(X_j(t)\,;\,X_i(\text{past}) \,\big|\, X_j(\text{past}),\, X_{k\neq i,j}(\text{past})\big).$$

Conditioning on **all other organs' pasts** is what distinguishes a *direct* edge
from one *mediated* through a third organ: if $X_i$ influences $X_j$ only via
$X_k$, then conditioning on $X_k$'s past drives $\mathrm{DI}(i\to j\,\|\,\text{rest})$
to zero — while a pairwise correlation, PAC, or even a pairwise transfer entropy
still lights up.

Two honesty guarantees:

1. **Cross-fitted estimation** — each edge is the held-out predictive
   log-likelihood gain of a model given $(X_i,\text{rest})$ over one given
   $\text{rest}$ alone, so an uninformative channel contributes $\approx 0$, not a
   positive bias. For a fixed model class the estimate is a *lower bound* on the
   true DI.
2. **A certified detection floor** — a subject/record-cluster bootstrap gives a
   one-sided 95% lower bound; a calibration injects *known* DI at graded strengths
   and reports the smallest true value whose bound clears zero. Below that floor a
   null is uninformative; above it, a null is a real negative.

Both linear (cross-fitted ridge plug-in) and neural (causal TCN + held-out
residual variance) estimators are provided; both are checked against the
closed-form linear-Gaussian DI.

---

## It works — validated against analytic ground truth

The estimator recovers the closed-form DI of a linear-Gaussian VAR, reads $\approx 0$
on a null channel, and its detection floor is explicit:

![detection-power calibration](experiments/fig_cdi_detection_power.png)

On a chain $A\to B\to C$ it recovers the two direct edges and **drives the
mediated $A\to C$ edge to zero**, while a pairwise (correlation-style) view
certifies $A\to C$ *spuriously*:

![method: direct vs mediated](experiments/fig_multiorgan_digraph.png)

---

## It runs on real multi-organ recordings

**Autonomic organs during sleep** (MIT-BIH Polysomnographic Database, `slpdb`:
EEG, heart rate, arterial BP, respiration; 18 recordings, ~308k anchors). The
certified graph recovers *known* autonomic couplings **with direction** —
respiration→heart-rate (respiratory sinus arrhythmia), heart-rate→BP — and finds
a certified **top-down EEG→heart-rate** edge while heart-rate→EEG does **not**
clear the floor. That directional asymmetry is exactly what correlation and PAC
cannot resolve.

![directed information: autonomic organs](experiments/fig_slpdb_digraph_heatmap.png)

*(rows = source / causal past, columns = destination / future; red outline =
certified, bootstrap 95% LCB > 0; diagonal masked.)*

For reference, the same pipeline on Sleep-EDF channels (frontal/occipital EEG,
EOG, EMG; 78 subjects, 6.4M anchors) — a within-brain/eye/muscle system:

![directed information: Sleep-EDF](experiments/fig_sleep_edf_digraph_heatmap.png)

---

## Direct line to the stomach–brain data

The estimator is signal-agnostic. Point it at a stomach–brain recording with
`Y_future = future cortical state`, `X_past = gastric (EGG) history`,
`Z = (cortical history, other organs, time)`, and `clusters = subject`, and it
certifies **how many bits the gastric rhythm contributes to the future cortical
state, conditioned on the cortex's own past** — the directional statement the 2026
paper's Section 3.6 leaves open. The same estimand applies to the cephalic-phase
efferent (brain→gut) question in the lab's Parkinson's program.

## Honest scope

- Effects on the public data above are **small** (sub-0.01 bits); at hundreds of
  thousands of anchors "certified" is an easy bar. The contribution is the
  **directionality + the direct-vs-mediated distinction + the certification**, not
  the magnitude.
- Directed information is not new, and the graph formulation is Quinn–Kiyavash–
  Coleman (2015). What is here: a **finite-sample-certified, multi-organ**
  implementation with a detection-power floor, validated end-to-end.
- The stomach–brain result requires the corresponding recordings; this repo
  demonstrates the method and its validation, not that result.

## Run it

```bash
pip install -r requirements.txt
export PYTHONPATH=.
python tests/test_linear.py     # linear DI vs analytic ground truth
python tests/test_graph.py      # graph: recovers structure, kills the mediated edge
python experiments/run_slpdb_graph.py --max-records 18   # the autonomic graph above
```

Layout: `neuroforecast/` (`linear.py`, `neural.py`, `graph.py`, `calibration.py`),
`experiments/` (runners + figures), `tests/` (analytic-validation gates).

## Reference

Quinn, Kiyavash, Coleman. *Directed Information Graphs.* IEEE Transactions on
Information Theory, 61(12), 2015.
