"""Real-data certificate on Sleep-EDF: does the causal EEG band-power trajectory
carry directed information about FUTURE delta power, beyond delta's own recent
past and time-of-night?

Why this target. Forecasting a coarse stage annotation conditions on an oracle
current stage, which removes most of the forecastable structure by construction.
Forecasting a *continuous* future band-power is not oracle-pinned, so it is the
estimand where residual directed EEG information can actually live. The certified
CDI answers it honestly:

    Y_future : delta-band power at issue_time + lead
    Z        : (recent delta-power level = delta's own past, time-of-night)
    X_past   : per-second band-power TRAJECTORY over the last WINDOW_S seconds
               (channels = bands, time = seconds) -- preserves the precursor
               dynamics a single average destroys.

    CDI = I(Y_future ; X_past | Z)  -- certified above the detection floor.

Run:
    PYTHONPATH=. python experiments/run_sleep_edf.py --data-dir <cassette> [--neural]
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch

from neuroforecast.linear import conditional_directed_information

BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (11.0, 16.0), (13.0, 30.0)]  # d th a sigma b
WINDOW_S = 120           # causal trajectory length (seconds)
LEAD_S = 180             # forecast horizon (3 min)
STEP_S = 30              # anchor cadence
_PSG = re.compile(r"^(SC4\d{3}[A-Z0-9]{2})-PSG\.edf$")


def _per_second_bandpowers(sig: np.ndarray, fs: float, n_sec: int) -> np.ndarray:
    """(n_bands, n_sec) log band-power, one column per second."""
    out = np.empty((len(BANDS), n_sec), dtype=np.float32)
    for s in range(n_sec):
        seg = sig[int(s * fs):int((s + 1) * fs)]
        f, pxx = welch(seg, fs=fs, nperseg=int(min(fs, len(seg))))
        for bi, (lo, hi) in enumerate(BANDS):
            out[bi, s] = np.log(pxx[(f >= lo) & (f < hi)].mean() + 1e-12)
    return out


def build_design(psg_path: Path, subj: str):
    """Return per-anchor (y_future, x_traj, z, subject) for one recording."""
    import edfio
    e = edfio.read_edf(str(psg_path))
    eeg = next((s for s in e.signals if "EEG" in s.label), None)
    if eeg is None:
        return []
    fs = eeg.sampling_frequency
    sig = np.asarray(eeg.data, dtype=np.float64)
    total_s = int(len(sig) / fs)
    bp = _per_second_bandpowers(sig, fs, total_s)          # (bands, total_s)
    delta = bp[0]                                          # delta trajectory
    rows = []
    t = WINDOW_S
    while t + LEAD_S < total_s:
        x_traj = bp[:, t - WINDOW_S:t]                     # (bands, WINDOW_S)
        y_future = float(delta[t + LEAD_S])                # future delta power
        z = np.array([delta[t - 1],                        # delta's own recent past
                      delta[t - WINDOW_S:t].mean(),        # recent delta level
                      min(t / (3600.0), 12.0)],            # hours into night
                     dtype=np.float64)
        rows.append((y_future, x_traj.astype(np.float32), z, subj))
        t += STEP_S
    return rows


def discover(data_dir: Path):
    files = {p.name for p in data_dir.glob("*.edf")}
    out = []
    for psg in sorted(f for f in files if _PSG.match(f)):
        rec = _PSG.match(psg).group(1)
        out.append((rec, rec[3:5], data_dir / psg))
    return out


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--neural", action="store_true", help="also run the TCN neural CDI")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    recs = discover(args.data_dir)
    rows, seen = [], set()
    for rec, subj, psg in recs:
        if args.max_subjects and subj not in seen and len(seen) >= args.max_subjects:
            continue
        try:
            rows.extend(build_design(psg, subj))
            seen.add(subj)
            print(f"  {rec}: total anchors now {len(rows)}")
        except Exception as exc:
            print(f"  skip {rec}: {type(exc).__name__}: {exc}")
    if not rows:
        raise SystemExit("no anchors built")

    y = np.array([r[0] for r in rows])
    x_traj = np.stack([r[1] for r in rows])                # (n, bands, WINDOW_S)
    z = np.stack([r[2] for r in rows])
    clusters = np.array([r[3] for r in rows])
    n_subj = len(set(clusters))
    print(f"\n{len(rows)} anchors, {n_subj} subjects, X_past {x_traj.shape[1:]}")

    # Linear CDI: collapse the trajectory to summary features (level+slope per band).
    lvl = x_traj[:, :, -1]
    slope = x_traj[:, :, -1] - x_traj[:, :, 0]
    x_feats = np.hstack([lvl, slope])
    lin = conditional_directed_information(y, x_feats, z, clusters=clusters,
                                           n_boot=args.bootstrap)
    print("\n=== linear CDI (band-power level+slope) ===")
    print(f"CDI = {lin.cdi_bits:+.4f} bits | 95% LCB = {lin.lcb95_bits:+.4f} | "
          f"{'CERTIFIED' if lin.is_certified_positive else 'not certified'}")

    if args.neural:
        from neuroforecast.neural import neural_conditional_directed_information
        neu = neural_conditional_directed_information(
            y, x_traj, z, clusters=clusters, epochs=60, n_boot=args.bootstrap)
        print("\n=== neural CDI (TCN over band-power trajectory) ===")
        print(f"CDI = {neu.cdi_bits:+.4f} bits | 95% LCB = {neu.lcb95_bits:+.4f} | "
              f"{'CERTIFIED' if neu.is_certified_positive else 'not certified'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
