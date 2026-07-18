"""Generic certified directed-information graph plot from a *_digraph.npz."""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

npz = Path(sys.argv[1]); title = sys.argv[2] if len(sys.argv) > 2 else "Certified directed-information graph"
d = np.load(npz, allow_pickle=True)
cdi, lcb, names = d["cdi"], d["lcb"], list(d["names"])
m = len(names)
# circular layout
ang = np.linspace(0, 2*np.pi, m, endpoint=False) + np.pi/2
pos = {names[i]: (np.cos(ang[i]), np.sin(ang[i])) for i in range(m)}
mx = max(cdi[i, j] for i in range(m) for j in range(m) if i != j and lcb[i, j] > 0)
fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
for nm, (x, y) in pos.items():
    ax.scatter([x], [y], s=3200, color="#e8eef7", edgecolors="#3d5a80", zorder=3, linewidths=1.5)
    ax.text(x, y, nm, ha="center", va="center", fontsize=9, zorder=4, weight="bold")
for i in range(m):
    for j in range(m):
        if i == j or lcb[i, j] <= 0:
            continue
        p0, p1 = pos[names[i]], pos[names[j]]
        lw = 0.8 + 4.5 * cdi[i, j] / mx
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                     shrinkA=32, shrinkB=32, lw=lw, color="#ee6c4d",
                     connectionstyle="arc3,rad=0.15", zorder=2))
        mxp, myp = 0.5*(p0[0]+p1[0]), 0.5*(p0[1]+p1[1])
        ax.text(mxp*1.12, myp*1.12, f"{cdi[i,j]:.4f}", fontsize=6.5, color="#c44", ha="center", zorder=5)
ax.set_title(f"{title}\n{int(d['n'])} anchors, {int(d.get('n_records', d.get('n_subjects', 0)))} recordings; edge width ~ bits", fontsize=10)
ax.axis("off"); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.6)
out = npz.with_name("fig_" + npz.stem)
fig.savefig(f"{out}.png", dpi=200); fig.savefig(f"{out}.pdf")
print(f"saved {out}.png")
