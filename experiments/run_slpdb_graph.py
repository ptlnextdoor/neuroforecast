"""PHASE 2 stretch (Todd-relevant): certified directed-information graph across
autonomic organs during sleep — brain, heart, blood pressure, respiration.

MIT-BIH Polysomnographic Database (slpdb, PhysioNet, open) records EEG (C4-A1),
ECG, arterial BP, and respiration simultaneously. This is the closest public
proxy to Coleman's gut-brain / autonomic-coupling program: instead of
brain-eye-muscle (Sleep-EDF), the nodes are real visceral/autonomic organs.

Per-second node signals:
  EEG   : log band-power (0.5-30 Hz)               -- cortical state
  HR    : instantaneous heart rate from QRS beats  -- cardiac autonomic drive
  BP    : log power envelope                        -- vascular tone
  Resp  : log power envelope                        -- respiratory drive

Then the certified causally-conditioned directed-information graph asks, e.g.:
does cardiac autonomic state directionally drive cortex, or is an apparent
BP<->EEG coupling mediated through heart rate / respiration? -- the directional,
direct-vs-mediated statement correlation and PAC cannot make.

    PYTHONPATH=. python experiments/run_slpdb_graph.py --max-records 12
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch

from neuroforecast.linear import conditional_directed_information

NODES = ["EEG", "HR", "BP", "Resp"]
LAG = 5
EEG_BAND = (0.5, 30.0)


def _sig(record, want):
    """Return (fs, data) for the first channel whose name contains `want`."""
    for k, name in enumerate(record.sig_name):
        if want.lower() in name.lower():
            return record.fs, np.asarray(record.p_signal[:, k], dtype=np.float64)
    return None


def _per_sec_power(data, fs, band, n_sec):
    """Per-second log band-power. Sub-1 Hz bands (BP, Resp) are unresolvable in a
    1 s window at these rates, so fall back to log mean-square amplitude, which
    still tracks the envelope of those slow organ signals."""
    lo, hi = band
    out = np.full(n_sec, np.nan)
    for s in range(n_sec):
        seg = data[int(s * fs):int((s + 1) * fs)]
        if len(seg) < 16:
            continue
        f, pxx = welch(seg, fs=fs, nperseg=int(min(fs, len(seg))))
        mask = (f >= lo) & (f < hi)
        out[s] = (np.log(pxx[mask].mean() + 1e-12) if mask.any()
                  else np.log(np.mean(seg ** 2) + 1e-12))
    return out


def _per_sec_hr(ecg, fs, n_sec):
    """Instantaneous heart rate (bpm) on a per-second grid from QRS detection.
    Falls back to log ECG power if detection is too sparse."""
    try:
        import wfdb.processing as wp
        beats = wp.xqrs_detect(sig=ecg, fs=fs, verbose=False)
        if len(beats) > 10:
            t_beat = beats / fs
            rr = np.diff(t_beat)
            hr = 60.0 / np.clip(rr, 0.3, 2.0)               # bpm at each beat
            t_mid = t_beat[1:]
            grid = np.arange(n_sec)
            return np.interp(grid, t_mid, hr, left=hr[0], right=hr[-1])
    except Exception:
        pass
    return _per_sec_power(ecg, fs, (0.5, 40.0), n_sec)      # fallback envelope


def load_record(rec: str):
    import wfdb
    r = wfdb.rdrecord(rec, pn_dir="slpdb")
    eeg = _sig(r, "EEG"); ecg = _sig(r, "ECG"); bp = _sig(r, "BP"); resp = _sig(r, "Resp")
    if not all([eeg, ecg, bp, resp]):
        return None
    n_sec = int(min(len(eeg[1]) / eeg[0], len(ecg[1]) / ecg[0],
                    len(bp[1]) / bp[0], len(resp[1]) / resp[0]))
    cols = [
        _per_sec_power(eeg[1], eeg[0], EEG_BAND, n_sec),
        _per_sec_hr(ecg[1], ecg[0], n_sec),
        _per_sec_power(bp[1], bp[0], (0.0001, 5.0), n_sec),
        _per_sec_power(resp[1], resp[0], (0.05, 1.0), n_sec),
    ]
    series = np.column_stack(cols)
    return (series - np.nanmean(series, 0)) / (np.nanstd(series, 0) + 1e-9)


def lagged(series, lag):
    T, m = series.shape
    future = series[lag:]
    past = np.stack([series[lag - k - 1:T - k - 1] for k in range(lag)], axis=-1)
    ok = ~(np.isnan(future).any(1) | np.isnan(past).any((1, 2)))
    return future[ok], past[ok]


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-records", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("experiments/slpdb_digraph.npz"))
    args = ap.parse_args()

    import wfdb
    recs = wfdb.get_record_list("slpdb")[: args.max_records]
    fut_all, past_all, clu_all = [], [], []
    for rec in recs:
        try:
            series = load_record(rec)
            if series is None:
                print(f"  skip {rec}: missing a required organ channel")
                continue
            fut, past = lagged(series, LAG)
        except Exception as exc:
            print(f"  skip {rec}: {type(exc).__name__}: {exc}")
            continue
        fut_all.append(fut); past_all.append(past)
        clu_all.append(np.full(len(fut), rec))
        print(f"  {rec}: {len(fut)} anchors")

    if not fut_all:
        raise SystemExit("no records loaded")
    future = np.concatenate(fut_all); past = np.concatenate(past_all)
    clusters = np.concatenate(clu_all)
    m = len(NODES)
    print(f"\n{len(future)} anchors, {len(set(clusters))} records, nodes={NODES}")

    cdi = np.zeros((m, m)); lcb = np.zeros((m, m))
    for j in range(m):
        y = future[:, j]
        for i in range(m):
            if i == j:
                continue
            others = [k for k in range(m) if k not in (i, j)]
            z = np.hstack([past[:, j, :]] + [past[:, k, :] for k in others])
            res = conditional_directed_information(y, past[:, i, :], z,
                                                   clusters=clusters, n_boot=args.bootstrap)
            cdi[i, j], lcb[i, j] = res.cdi_bits, res.lcb95_bits

    print("\n=== certified directed-information graph (bits) — autonomic organs, slpdb ===")
    edges = []
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            c = "YES" if lcb[i, j] > 0 else "no"
            if lcb[i, j] > 0:
                edges.append((NODES[i], NODES[j], cdi[i, j]))
            print(f"  {NODES[i]:>5} -> {NODES[j]:<5}: {cdi[i,j]:+.4f} [{lcb[i,j]:+.4f}]  {c}")
    print("\ncertified directed edges:", sorted(edges, key=lambda e: -e[2]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, cdi=cdi, lcb=lcb, names=NODES, n=len(future), n_records=len(set(clusters)))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
