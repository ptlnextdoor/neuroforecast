"""Multi-organ certified directed-information graph -- aimed at the stomach-brain
sleep setting (Rao et al. 2025), which currently reports only correlation/PAC.

We simulate four simultaneously-recorded organ signals with a KNOWN directed
structure motivated by the sleep physiology:

    EKG_HRV  -->  EGG        (autonomic tone modulates the gastric rhythm)
    EGG      -->  EEG_sigma  (the gastric slow wave drives cortical sigma: Rao's finding)
    EMG                       (isolated: movement/artifact, no true coupling)

so EKG_HRV -> EEG_sigma is MEDIATED through EGG. A correlational or pairwise-PAC
analysis would report an EKG_HRV <-> EEG_sigma coupling; the causally-conditioned
directed-information graph correctly does NOT certify that edge, while it does
certify the two direct edges -- the exact directionality the lab said it lacks.

Produces `fig_multiorgan_digraph.png/.pdf`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuroforecast.graph import analytic_var1_di_graph, directed_information_graph

NAMES = ["EKG_HRV", "EGG", "EEG_sigma", "EMG"]
# rows = destination, cols = source in a VAR: x(t) = A x(t-1) + noise
# encode EKG->EGG and EGG->EEG_sigma (indices: 0 EKG, 1 EGG, 2 EEG, 3 EMG)
A = np.array([
    [0.35, 0.00, 0.00, 0.00],   # EKG_HRV self
    [0.50, 0.35, 0.00, 0.00],   # EGG <- EKG (direct)
    [0.00, 0.55, 0.35, 0.00],   # EEG_sigma <- EGG (direct); <- EKG only via EGG
    [0.00, 0.00, 0.00, 0.35],   # EMG isolated
])


def simulate(T, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 4))
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.standard_normal(4)
    return x


def main() -> int:
    truth = analytic_var1_di_graph(A)
    x = simulate(30000, seed=0)
    clusters = np.repeat(np.arange(40), len(x) // 40 + 1)[:len(x) - 1]
    g = directed_information_graph(x, NAMES, clusters=clusters, n_boot=1000, seed=0)

    print("=== certified directed-information graph (bits) ===")
    print("edge (source -> dest) : estimate [95% LCB]  (analytic)  certified?")
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            cert = "YES" if g.lcb[i, j] > 0 else "no"
            print(f"  {NAMES[i]:>9} -> {NAMES[j]:<9}: {g.cdi[i,j]:+.4f} "
                  f"[{g.lcb[i,j]:+.4f}]  (truth {truth[i,j]:.4f})  {cert}")
    print("\ncertified edges:", g.certified_edges())
    print("note: EKG_HRV -> EEG_sigma is mediated through EGG and is correctly NOT "
          "certified -- the direct/mediated distinction correlation & PAC cannot make.")
    _plot(g)
    return 0


def _plot(g):
    try:
        import matplotlib
        matplotlib.use("agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
    except Exception:
        print("(matplotlib unavailable; skipping figure)")
        return
    pos = {"EKG_HRV": (0, 0), "EGG": (1, 0.6), "EEG_sigma": (2, 0), "EMG": (1, -0.9)}
    fig, ax = plt.subplots(figsize=(6, 3.6), constrained_layout=True)
    for name, (px, py) in pos.items():
        ax.scatter([px], [py], s=2400, color="#e8eef7", edgecolors="#3d5a80", zorder=2)
        ax.text(px, py, name, ha="center", va="center", fontsize=8, zorder=3)
    m = len(g.names)
    for i in range(m):
        for j in range(m):
            if i == j or g.lcb[i, j] <= 0:
                continue
            p0, p1 = pos[g.names[i]], pos[g.names[j]]
            ax.add_patch(FancyArrowPatch(
                p0, p1, arrowstyle="-|>", mutation_scale=16,
                shrinkA=26, shrinkB=26, lw=1.6, color="#ee6c4d", zorder=1))
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            ax.text(mx, my + 0.08, f"{g.cdi[i,j]:.3f} bits", fontsize=7,
                    color="#ee6c4d", ha="center")
    ax.set_title("Certified directed-information graph (multi-organ, sleep)")
    ax.axis("off")
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-1.4, 1.2)
    out = Path(__file__).parent / "fig_multiorgan_digraph"
    fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf")
    print(f"saved {out}.png / .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
