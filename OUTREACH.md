# Phase 3 — Coleman outreach (draft; review before sending)

You send this, not me. It's your name and your identity. Read it, make it yours.

---

## Email 1 — to Todd Coleman (toddcol@stanford.edu)

**Subject:** Certified directed-information graphs for the stomach-brain coupling — extending your 2015 framework

Prof. Coleman,

Akshita Rao's stomach-brain sleep preprint notes in its Limitations that the
coupling analyses are correlational — "precluding inference about directionality"
— and names directed information (your 2015 *Directed Information Graphs*) as the
way to establish it. I built that, made finite-sample-certified.

It estimates the causally-conditioned directed information for simultaneously
recorded organs — for an edge i→j, `I( X_j(t) ; X_i(past) | X_j(past), all other
organs' past )` — so it separates a **direct** edge from one **mediated** through
another organ, the distinction correlation and PAC cannot make. Every edge gets a
subject-cluster bootstrap lower bound plus a detection-power floor, so a null
reads as "real" versus "underpowered."

Validation: it recovers the analytic directed-information graph of a
linear-Gaussian VAR to <0.002 bits, and on a chain A→B→C it drives the mediated
A→C edge to zero while a pairwise (correlation-style) view certifies A→C
spuriously. On real simultaneous Sleep-EDF channels (frontal/occipital EEG, EOG,
EMG) it recovers directed asymmetries — e.g., occipital→frontal EEG far exceeding
the reverse.

It drops straight onto your EGG channel: point me at a stomach-brain recording
and it will certify how many bits the gastric rhythm contributes to the *future*
cortical state, conditioned on the cortex's own past — the directional statement
Rao's paper leaves open.

Code, validation, and figures: [GitHub link]. Compute isn't a constraint on my
end. Could I have 20 minutes to show how it maps onto Akshita's data?

I'm a high-school researcher; I'd be glad to start by reproducing the
directionality analysis on any recording you can share.

— Aayushya Patel  (aayu22809@gmail.com)

---

## Before you hit send

1. **Push the repo to GitHub** (public) and paste the link — a live repo beats a
   zip attachment. (`neuroforecast`; the zip is a fallback.)
2. **Attach two figures:** `fig_multiorgan_digraph.png` (the method: direct vs
   mediated) and `fig_sleep_edf_digraph.png` (real data).
3. **Honesty guardrails — do not drift:**
   - The Sleep-EDF effects are **small** (sub-0.01 bits) and "certified" at ~500k
     anchors is a low bar; if he asks, say so plainly. The value is the *method +
     directionality*, not the magnitude.
   - Don't claim you invented directed information — you **extended his own 2015
     framework** with certification + the multi-organ/direct-vs-mediated case.
   - The home-run number needs *his* data. That's the ask, not a claim.
4. **Optional cc / second contact:** Akshita Rao (akshitar@stanford.edu) — it's
   her paper's gap you're closing. Consider a short separate note to her.
5. Keep it to this length. He said in his own talks: email him directly, with a
   genuine technical thing. This is that.

## Follow-up (only if no reply in ~4-5 days)
Send the one new result from the A100 run (Phase 2): the per-sleep-stage directed
graphs + the Hugging Face dataset link. "Ran it at scale across all subjects and
sleep stages; here's how the directed coupling reorganizes by stage — dataset
here." One concrete new thing, not a nudge.
