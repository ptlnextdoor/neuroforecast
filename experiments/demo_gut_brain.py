"""Demo: certifying whether an added channel improves forecasting.

Mirrors the structure of a gut-brain sleep-forecasting question --

    Y_future : a future brain/sleep state to forecast
    Z        : the autoregressive/nuisance baseline (Y's own past + time-of-night)
    X        : a candidate *added* channel (e.g. the gastric signal)

and asks: does X carry directed information about Y_future beyond Z, and is that
improvement certified (bootstrap LCB > 0) or merely underpowered?

The demo does two things Coleman's subfield rarely does together:
  1. Recovers the *true* directed information (linear-Gaussian ground truth).
  2. Builds a detection-power curve: the estimator's certified floor, i.e. the
     smallest true CDI whose bootstrap LCB clears zero at this sample size.

Produces `fig_cdi_detection_power.png/.pdf`.

    python demo_gut_brain.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuroforecast.linear import analytic_cdi_gaussian, conditional_directed_information

# Coupling strengths of the candidate channel X -> future of Y.
STRENGTHS = (0.0, 0.1, 0.2, 0.35, 0.55, 0.8)
N = 8000            # samples (a realistic single-cohort size)
N_CLUSTERS = 40     # e.g. subjects, for the cluster bootstrap
BASE_Z = 0.8        # baseline (autoregressive/nuisance) coupling


def simulate(strength: float, seed: int):
    """Transfer-entropy structure of the gut-brain forecasting question:

        brain_future = PHI * brain_past + strength * gut_past + C * circadian + noise

    The baseline Z = (brain_past, circadian) is the brain's OWN autoregressive
    past plus time-of-night; X = gut_past is the candidate added channel. Then
    CDI = I(brain_future ; gut_past | brain_past, circadian) is exactly transfer
    entropy from gut to brain -- the certified, causal version of "does adding
    the gut improve the forecast."

    Returns (brain_future, gut_past, Z, clusters) and the analytic CDI.
    """
    rng = np.random.default_rng(seed)
    PHI, C = 0.6, 0.5
    brain_past = rng.standard_normal(N)
    gut_past = rng.standard_normal(N)
    circadian = rng.standard_normal(N)
    brain_future = (PHI * brain_past + strength * gut_past
                    + C * circadian + rng.standard_normal(N))
    Z = np.column_stack([brain_past, circadian])
    clusters = rng.integers(0, N_CLUSTERS, size=N)
    # Joint covariance in order [brain_future, gut_past, brain_past, circadian].
    var_bf = PHI**2 + strength**2 + C**2 + 1.0
    cov = np.array([
        [var_bf, strength, PHI, C],
        [strength, 1.0, 0.0, 0.0],
        [PHI, 0.0, 1.0, 0.0],
        [C, 0.0, 0.0, 1.0],
    ])
    true = analytic_cdi_gaussian(cov, iy=0, ix=[1], iz=[2, 3])
    return (brain_future, gut_past, Z, clusters), true


def main() -> int:
    rows = []
    print(f"{'strength':>8} | {'true CDI':>9} | {'est CDI':>9} | {'95% LCB':>9} | certified")
    for s in STRENGTHS:
        (y, x, z, clusters), true = simulate(s, seed=100 + int(s * 100))
        res = conditional_directed_information(y, x, z, clusters=clusters,
                                               n_boot=2000, seed=0)
        rows.append((s, true, res.cdi_bits, res.lcb95_bits))
        print(f"{s:>8.2f} | {true:>+9.4f} | {res.cdi_bits:>+9.4f} | "
              f"{res.lcb95_bits:>+9.4f} | {'YES' if res.is_certified_positive else 'no'}")

    floor = next((t for _, t, _, lcb in rows if lcb > 0), None)
    print(f"\nCertified detection floor at this sample size: "
          f"true CDI ~ {floor:.3f} bits (smallest true CDI with LCB>0).")
    print("\nThe certificate a gut-brain forecasting claim needs, in one line:")
    print("  'the gastric channel contributes CDI = <est> bits [LCB <lcb>] about the")
    print("   future brain state, conditioned on the brain's own past -- above the")
    print(f"   {floor:.3f}-bit detection floor, so the contribution is real, not underpowered.'")

    _plot(rows, floor)
    return 0


def _plot(rows, floor):
    try:
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib unavailable; skipping figure)")
        return
    true = [r[1] for r in rows]
    est = [r[2] for r in rows]
    lcb = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    lo = min(min(lcb), 0) - 0.02
    ax.axhspan(lo, 0, color="0.92", zorder=0)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.plot(true, true, ":", color="0.5", lw=1.0, label="analytic CDI (truth)")
    ax.plot(true, est, "-o", color="#1f77b4", label="estimated CDI")
    ax.fill_between(true, lcb, [max(e, l) for e, l in zip(est, lcb)],
                    color="#1f77b4", alpha=0.2)
    ax.plot(true, lcb, "--", color="#1f77b4", lw=1.0, label="95% LCB (cluster bootstrap)")
    if floor is not None:
        ax.axvline(floor, color="#d62728", lw=1.0, ls="--")
        ax.text(floor, ax.get_ylim()[1], " detection floor",
                color="#d62728", fontsize=8, va="top")
    ax.set_xlabel("true directed information  I(Y_future ; X_past | Z)  [bits]")
    ax.set_ylabel("recovered CDI [bits]")
    ax.set_title("Certified directed-information detection floor")
    ax.legend(fontsize=7, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    out = Path(__file__).parent / "fig_cdi_detection_power"
    fig.savefig(f"{out}.png", dpi=200)
    fig.savefig(f"{out}.pdf")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
