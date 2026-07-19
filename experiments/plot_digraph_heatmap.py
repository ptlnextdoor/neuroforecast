"""Directed-information graph as a matrix heatmap — the unambiguous, overlap-free
rendering an information theorist expects. Rows = source, columns = destination;
cell = certified directed information (bits); certified edges (bootstrap LCB > 0)
are outlined in red. Diagonal is masked (no self-edges).

    python experiments/plot_digraph_heatmap.py experiments/slpdb_digraph.npz "title"
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / ".claude/skills/scientific-figures/scripts"))
import figure_style as fs  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

npz = Path(sys.argv[1])
title = sys.argv[2] if len(sys.argv) > 2 else "Certified directed-information graph"
d = np.load(npz, allow_pickle=True)
cdi, lcb, names = np.array(d["cdi"], float), np.array(d["lcb"], float), list(d["names"])
m = len(names)

M = cdi.copy()
np.fill_diagonal(M, np.nan)
vmax = np.nanmax(M)

fs.use_paper_style()
fig, ax = fs.new_figure(cols=2, aspect=0.5)
im = ax.imshow(M, cmap=fs.SEQUENTIAL, vmin=0, vmax=vmax, aspect="equal")

ax.set_xticks(range(m)); ax.set_yticks(range(m))
ax.set_xticklabels(names); ax.set_yticklabels(names)
ax.set_xlabel("destination (future)"); ax.set_ylabel("source (causal past)")
n_rec = int(d.get("n_records", d.get("n_subjects", 0)))
ax.set_title(f"{title}\n{int(d['n'])} anchors · {n_rec} recordings · "
             "red outline = certified (95% LCB > 0)", fontsize=9)

for i in range(m):
    for j in range(m):
        if i == j:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#e6e6e6", edgecolor="none"))
            continue
        val = cdi[i, j]
        cert = lcb[i, j] > 0
        tc = "white" if val < vmax * 0.55 else "black"
        ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7,
                color=tc, weight="bold" if cert else "normal")
        if cert:
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor=fs.PALETTE[3], lw=2.2))

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("directed information (bits)")

fs.assert_no_clip(fig)
out = npz.with_name("fig_" + npz.stem + "_heatmap")
fs.save(fig, str(out))
print(f"saved {out}.png")
