"""Reproduce Rao et al.'s figure grammar (multitaper spectrogram+hypnogram,
raincloud + LMM stage-effect) on our own real data (Sleep-EDF).

Panel C-style: overnight EEG spectrogram stacked above the real hypnogram.
Panel E/H-style: raincloud plot of EEG band power by sleep stage, with an LMM
(random intercept per subject) Wald test + Cohen's d + FDR correction across
pairwise stage contrasts -- the same statistical machinery as the paper.

    PYTHONPATH=. python experiments/plot_rao_style_panels.py --data-dir <cassette> --max-subjects 15
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / ".claude/skills/scientific-figures/scripts"))
import figure_style as fs  # noqa: E402

STAGE_MAP = {"Sleep stage W": "Wake", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
             "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]
_PSG = re.compile(r"^(SC4\d{3}[A-Z0-9]{2})-PSG\.edf$")


# --------------------------------------------------------------------------- #
# Panel C: multitaper spectrogram + hypnogram
# --------------------------------------------------------------------------- #
def make_spectrogram_hypnogram_panel(psg_path: Path, hyp_path: Path, out: Path):
    import edfio
    from mne.time_frequency import psd_array_multitaper

    e = edfio.read_edf(str(psg_path))
    eeg = next(s for s in e.signals if "EEG Fpz" in s.label)
    fs_hz = eeg.sampling_frequency
    sig = np.asarray(eeg.data, dtype=np.float64)

    win_s, step_s = 30.0, 30.0
    n_win = int(len(sig) / fs_hz / step_s)
    freqs = None
    spec = []
    for w in range(n_win):
        seg = sig[int(w * step_s * fs_hz):int((w * step_s + win_s) * fs_hz)]
        if len(seg) < fs_hz * win_s * 0.9:
            break
        psd, f = psd_array_multitaper(seg, sfreq=fs_hz, fmin=0.5, fmax=30,
                                      bandwidth=1.0, verbose=False)
        spec.append(10 * np.log10(psd + 1e-20))
        freqs = f
    spec = np.array(spec).T  # (freq, time)

    h = edfio.read_edf(str(hyp_path))
    hyp = np.array(["?"] * n_win, dtype=object)
    for ann in h.annotations:
        st = STAGE_MAP.get(ann.text)
        if st is None:
            continue
        a = int(ann.onset / step_s)
        b = int((ann.onset + max(ann.duration, 1)) / step_s)
        hyp[a:min(b, n_win)] = st
    t_hr = np.arange(n_win) * step_s / 3600.0

    fs.use_paper_style()
    fig, (ax1, ax2) = fs.new_figure(cols=2, aspect=0.55, panels=2)
    im = ax1.pcolormesh(t_hr, freqs, spec, shading="auto", cmap=fs.SEQUENTIAL,
                        vmin=np.percentile(spec, 5), vmax=np.percentile(spec, 95))
    ax1.set_ylabel("frequency (Hz)")
    ax1.set_title("Overnight EEG spectrogram (real Sleep-EDF)")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.01)
    cbar.set_label("power (dB)")

    stage_y = {s: i for i, s in enumerate(STAGE_ORDER)}
    stage_y["?"] = np.nan
    y = np.array([stage_y[s] for s in hyp], dtype=float)
    ax2.plot(t_hr, y, color=fs.PALETTE[0], lw=1.0, drawstyle="steps-post")
    ax2.set_yticks(range(len(STAGE_ORDER))); ax2.set_yticklabels(STAGE_ORDER)
    ax2.invert_yaxis()
    ax2.set_xlabel("time (hours)"); ax2.set_ylabel("stage")
    ax2.set_title("Hypnogram")
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False); ax2.spines[spine].set_visible(False)

    fs.assert_no_clip(fig)
    fs.save(fig, str(out))
    print(f"saved {out}.png")


# --------------------------------------------------------------------------- #
# Panel E/H: raincloud + LMM stage effect
# --------------------------------------------------------------------------- #
def _raincloud(ax, data_by_group, colors, labels):
    """Half-violin + box + jittered strip -- the raincloud combination."""
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(data_by_group.items()):
        vp = ax.violinplot([vals], positions=[i], widths=0.8, showextrema=False)
        for b in vp["bodies"]:
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_facecolor(colors[i]); b.set_alpha(0.55)
        bp = ax.boxplot([vals], positions=[i + 0.18], widths=0.12, patch_artist=True,
                        showfliers=False, zorder=3)
        for patch in bp["boxes"]:
            patch.set_facecolor("white"); patch.set_edgecolor("0.3")
        jitter = rng.normal(-0.35, 0.03, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=4, color=colors[i],
                  alpha=0.5, zorder=2)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)


def make_raincloud_panel(data_dir: Path, max_subjects: int, out: Path):
    import edfio
    from scipy.signal import welch
    from scipy.stats import ttest_ind
    from statsmodels.formula.api import mixedlm
    from statsmodels.stats.multitest import multipletests

    files = {p.name for p in data_dir.glob("*.edf")}
    psgs = sorted(f for f in files if _PSG.match(f))
    rows = []
    seen = set()
    for psg in psgs:
        rec = _PSG.match(psg).group(1)
        subj = rec[3:5]
        if subj in seen or len(seen) >= max_subjects:
            if len(seen) >= max_subjects:
                break
            continue
        hyp_name = next((f for f in files if f.startswith(rec[:6]) and "Hypnogram" in f), None)
        if hyp_name is None:
            continue
        try:
            e = edfio.read_edf(str(data_dir / psg))
            eeg = next(s for s in e.signals if "EEG Fpz" in s.label)
            fs_hz = eeg.sampling_frequency
            sig = np.asarray(eeg.data, dtype=np.float64)
            h = edfio.read_edf(str(data_dir / hyp_name))
        except Exception:
            continue
        n_epochs = int(len(sig) / fs_hz / 30)
        stage = np.array(["?"] * n_epochs, dtype=object)
        for ann in h.annotations:
            st = STAGE_MAP.get(ann.text)
            if st is None:
                continue
            a, b = int(ann.onset / 30), int((ann.onset + max(ann.duration, 1)) / 30)
            stage[a:min(b, n_epochs)] = st
        for ep in range(n_epochs):
            if stage[ep] not in ("NREM_UNUSED",) and stage[ep] != "?":
                seg = sig[int(ep * 30 * fs_hz):int((ep + 1) * 30 * fs_hz)]
                if len(seg) < fs_hz * 30 * 0.9:
                    continue
                f, pxx = welch(seg, fs=fs_hz, nperseg=int(min(4 * fs_hz, len(seg))))
                delta = np.log(pxx[(f >= 0.5) & (f < 4.0)].mean() + 1e-12)
                rows.append({"subject": subj, "stage": stage[ep], "delta": delta})
        seen.add(subj)

    df = pd.DataFrame(rows)
    df = df[df.stage.isin(["Wake", "N2", "N3", "REM"])]
    # normalize within subject (z-score) so cross-subject amplitude offsets don't dominate
    df["delta_z"] = df.groupby("subject")["delta"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))

    # LMM: delta_z ~ stage, random intercept per subject (mirrors the paper's LMM approach)
    df["stage"] = pd.Categorical(df["stage"], categories=["Wake", "N2", "N3", "REM"])
    model = mixedlm("delta_z ~ C(stage, Treatment('N2'))", df, groups=df["subject"]).fit()

    order = ["Wake", "N2", "N3", "REM"]
    data_by_group = {s: df.loc[df.stage == s, "delta_z"].values for s in order}
    colors = [fs.PALETTE[i] for i in range(len(order))]

    fs.use_paper_style()
    fig, ax = fs.new_figure(cols=1, aspect=0.8)
    _raincloud(ax, data_by_group, colors, order)
    ax.set_ylabel("within-subject z(log delta power)")
    ax.set_title("EEG delta power by sleep stage (real Sleep-EDF)")

    # pairwise contrasts vs N3-N2 (the paper's headline contrast), FDR-corrected
    pairs = [("N3", "N2"), ("Wake", "N2"), ("REM", "N2")]
    pvals = []
    for a, b in pairs:
        t, p = ttest_ind(data_by_group[a], data_by_group[b], equal_var=False)
        pvals.append(p)
    _, p_fdr, _, _ = multipletests(pvals, method="fdr_bh")

    def cohend(a, b):
        na, nb = len(a), len(b)
        pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
        return (a.mean() - b.mean()) / pooled

    # Sort contrasts by span width so narrower brackets sit low and wider ones
    # stack above them -- avoids a wide bracket crossing a narrower one's label.
    contrasts = []
    for k, (a, b) in enumerate(pairs):
        ia, ib = order.index(a), order.index(b)
        d = cohend(data_by_group[a], data_by_group[b])
        contrasts.append((abs(ib - ia), min(ia, ib), max(ia, ib), d, p_fdr[k]))
    contrasts.sort(key=lambda c: c[0])

    y0 = max(np.percentile(v, 97) for v in data_by_group.values())
    step = 1.0 * (np.ptp(np.concatenate(list(data_by_group.values()))) / 8)
    texts = []
    for k, (_, ia, ib, d, p) in enumerate(contrasts):
        yb = y0 + step * (k + 1)
        ax.plot([ia, ib], [yb, yb], color="0.3", lw=1.0, zorder=4)
        stars = "****" if p < 1e-4 else "***" if p < 1e-3 else \
            "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        texts.append(ax.annotate(f"{stars}  d={d:.2f}", ((ia + ib) / 2, yb),
                                 fontsize=6.5, ha="center", va="bottom", zorder=5))
    fs.place_labels(ax, texts, arrow=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fs.assert_no_clip(fig)
    fs.save(fig, str(out))
    print(f"saved {out}.png")
    print(model.summary())


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--max-subjects", type=int, default=15)
    args = ap.parse_args()

    files = {p.name for p in args.data_dir.glob("*.edf")}
    psgs = sorted(f for f in files if _PSG.match(f))
    rec = _PSG.match(psgs[0]).group(1)
    hyp = next(f for f in files if f.startswith(rec[:6]) and "Hypnogram" in f)
    make_spectrogram_hypnogram_panel(
        args.data_dir / psgs[0], args.data_dir / hyp,
        Path("experiments/fig_rao_style_spectrogram_hypnogram"))

    make_raincloud_panel(args.data_dir, args.max_subjects,
                         Path("experiments/fig_rao_style_raincloud"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
