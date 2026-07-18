"""Plot the certified directed-information graph from run_sleep_edf_graph.py output."""
import sys
from pathlib import Path

import numpy as np

npz = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/sleep_edf_digraph.npz")
d = np.load(npz, allow_pickle=True)
cdi, lcb, names = d["cdi"], d["lcb"], list(d["names"])
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

pos = {"EEG_front": (0, 1), "EEG_occip": (2, 1), "EOG": (0, 0), "EMG": (2, 0)}
fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
for nm, (x, y) in pos.items():
    ax.scatter([x], [y], s=2600, color="#e8eef7", edgecolors="#3d5a80", zorder=3)
    ax.text(x, y, nm, ha="center", va="center", fontsize=8, zorder=4)
mx = max(cdi[i, j] for i in range(len(names)) for j in range(len(names)) if i != j)
for i, si in enumerate(names):
    for j, sj in enumerate(names):
        if i == j or lcb[i, j] <= 0:
            continue
        p0, p1 = pos[si], pos[sj]
        lw = 0.8 + 4.0 * cdi[i, j] / mx
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                     shrinkA=30, shrinkB=30, lw=lw, color="#ee6c4d",
                     connectionstyle="arc3,rad=0.12", zorder=2))
ax.set_title(f"Certified directed-information graph — real Sleep-EDF\n"
             f"({int(d['n'])} anchors, {int(d['n_subjects'])} subjects; edge width ~ bits)")
ax.axis("off"); ax.set_xlim(-0.8, 2.8); ax.set_ylim(-0.6, 1.6)
out = npz.with_name("fig_sleep_edf_digraph")
fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf")
print(f"saved {out}.png")
