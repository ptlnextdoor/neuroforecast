"""PHASE 1 (real data, local): certified directed-information graph across the
real simultaneous physiological channels in Sleep-EDF.

Sleep-EDF cassette records four organs at once: frontal EEG (Fpz-Cz), occipital
EEG (Pz-Oz), eye (EOG horizontal), and muscle (EMG submental). We build a
per-second log-power envelope for each, then estimate the certified
causally-conditioned directed-information graph among them -- the same estimand
Rao et al. said their stomach-brain analysis lacked, here demonstrated on real
multi-channel sleep recordings. Lags never cross subject boundaries; the bootstrap
resamples whole subjects.

    PYTHONPATH=. python experiments/run_sleep_edf_graph.py --data-dir <cassette> --max-subjects 20
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch

from neuroforecast.linear import conditional_directed_information

# channel label substring -> node name ; band (Hz) for the per-second envelope
CHANNELS = [
    ("EEG Fpz-Cz", "EEG_front", (0.5, 30.0)),
    ("EEG Pz-Oz", "EEG_occip", (0.5, 30.0)),
    ("EOG", "EOG", (0.5, 8.0)),
    ("EMG", "EMG", (10.0, 40.0)),
]
LAG = 5           # seconds of history
_PSG = re.compile(r"^(SC4\d{3}[A-Z0-9]{2})-PSG\.edf$")


def _per_second_logpower(sig, fs, band, n_sec):
    """Per-second log feature. Band-power via welch when the rate resolves the
    band; otherwise (e.g. Sleep-EDF's 1 Hz EMG) fall back to log mean-square
    amplitude so low-rate processed channels still contribute."""
    out = np.full(n_sec, np.nan)
    lo, hi = band
    use_welch = fs >= 2 * hi and fs >= 16
    for s in range(n_sec):
        seg = sig[int(s * fs):int((s + 1) * fs)]
        if len(seg) == 0:
            continue
        if use_welch and len(seg) >= 16:
            f, pxx = welch(seg, fs=fs, nperseg=int(min(fs, len(seg))))
            out[s] = np.log(pxx[(f >= lo) & (f < hi)].mean() + 1e-12)
        else:
            out[s] = np.log(np.mean(seg ** 2) + 1e-12)
    return out


def build_subject_series(psg_path: Path):
    """Return (n_sec, 4) log-power envelope series for the 4 channels, or None."""
    import edfio
    e = edfio.read_edf(str(psg_path))
    cols = []
    for label_sub, _name, band in CHANNELS:
        sig = next((s for s in e.signals if label_sub in s.label), None)
        if sig is None:
            return None
        fs = sig.sampling_frequency
        data = np.asarray(sig.data, dtype=np.float64)
        n_sec = int(len(data) / fs)
        cols.append(_per_second_logpower(data, fs, band, n_sec))
    n = min(len(c) for c in cols)
    series = np.column_stack([c[:n] for c in cols])
    # z-score each channel within subject (envelope scale differs per channel/person)
    series = (series - np.nanmean(series, 0)) / (np.nanstd(series, 0) + 1e-9)
    return series


def lagged(series, lag):
    """future (n, m), past (n, m, lag) within a single subject (no boundary cross)."""
    T, m = series.shape
    future = series[lag:]
    past = np.stack([series[lag - k - 1:T - k - 1] for k in range(lag)], axis=-1)
    ok = ~(np.isnan(future).any(1) | np.isnan(past).any((1, 2)))
    return future[ok], past[ok]


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--max-subjects", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("experiments/sleep_edf_digraph.npz"))
    args = ap.parse_args()

    files = {p.name for p in args.data_dir.glob("*.edf")}
    psgs = sorted(f for f in files if _PSG.match(f))
    names = [n for _, n, _ in CHANNELS]
    m = len(names)

    fut_all, past_all, clu_all = [], [], []
    seen = set()
    for psg in psgs:
        subj = _PSG.match(psg).group(1)[3:5]
        if subj in seen or len(seen) >= args.max_subjects:
            if len(seen) >= args.max_subjects:
                break
            continue
        try:
            series = build_subject_series(args.data_dir / psg)
            if series is None:
                continue
            fut, past = lagged(series, LAG)
        except Exception as exc:
            print(f"  skip {psg}: {type(exc).__name__}: {exc}")
            continue
        fut_all.append(fut)
        past_all.append(past)
        clu_all.append(np.full(len(fut), subj))
        seen.add(subj)
        print(f"  {psg}: {len(fut)} anchors (subj {subj}, total {len(seen)})")

    future = np.concatenate(fut_all)
    past = np.concatenate(past_all)
    clusters = np.concatenate(clu_all)
    n = len(future)
    print(f"\n{n} anchors, {len(seen)} subjects, channels={names}")

    cdi = np.zeros((m, m))
    lcb = np.zeros((m, m))
    for j in range(m):
        y = future[:, j]
        for i in range(m):
            if i == j:
                continue
            others = [k for k in range(m) if k not in (i, j)]
            x_src = past[:, i, :]
            z = np.hstack([past[:, j, :]] + [past[:, k, :] for k in others])
            res = conditional_directed_information(
                y, x_src, z, clusters=clusters, n_boot=args.bootstrap)
            cdi[i, j], lcb[i, j] = res.cdi_bits, res.lcb95_bits

    print("\n=== certified directed-information graph (bits), real Sleep-EDF ===")
    print("source -> dest : CDI [95% LCB]  certified?")
    edges = []
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            c = "YES" if lcb[i, j] > 0 else "no"
            if lcb[i, j] > 0:
                edges.append((names[i], names[j], cdi[i, j]))
            print(f"  {names[i]:>9} -> {names[j]:<9}: {cdi[i,j]:+.4f} [{lcb[i,j]:+.4f}]  {c}")
    print("\ncertified directed edges:", sorted(edges, key=lambda e: -e[2]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, cdi=cdi, lcb=lcb, names=names, n=n, n_subjects=len(seen))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
