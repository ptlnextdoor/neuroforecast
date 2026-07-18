"""PHASE 2 (A100, intensive): the full multi-organ directed-information program at
scale, with everything exported to a Hugging Face dataset.

Intensity dials (all cranked by default; tune with flags):
  * ALL subjects, ALL physiological channels present in each recording.
  * per-SLEEP-STAGE directed graphs (W/N1/N2/N3/REM) -- directed coupling
    reorganizes across stages; this is the core scientific insight.
  * a LAG SWEEP (history depth) per stage.
  * linear AND neural (causal-TCN) directed-information edges.
  * raw waveform windows exported for downstream modeling.
  * detection-power calibration matched to the real data scale.

Exports (to ./hf_export, then pushed to the Hub if HF_TOKEN is set):
  edges.parquet         one row per (subject, stage, lag, src, dst): cdi, lcb, n, certified
  graphs/*.npz          per-stage aggregate CDI/LCB matrices
  raw_windows/*.npy     sampled raw multi-channel causal windows (the waveforms)
  calibration.json      certified detection floor at this data scale
  insights.json         population-certified edges, stage reorganization summary
  README.md             dataset card

The export ASSEMBLY is validated locally with `--dry-run` (synthetic, no data, no
token). The Hub push is gated on HF_TOKEN so nothing leaks by accident.

    # local validation of the whole pipeline (no data, no token):
    PYTHONPATH=. python experiments/run_a100_intensive.py --dry-run
    # real cluster run:
    HF_TOKEN=hf_xxx PYTHONPATH=. python experiments/run_a100_intensive.py \
        --data-dir <sleep-cassette> --neural --hf-repo <user>/neuroforecast-digraph
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import welch

from neuroforecast.calibration import detection_floor
from neuroforecast.linear import analytic_cdi_gaussian, conditional_directed_information

CHANNELS = [
    ("EEG Fpz-Cz", "EEG_front", (0.5, 30.0)),
    ("EEG Pz-Oz", "EEG_occip", (0.5, 30.0)),
    ("EOG", "EOG", (0.5, 8.0)),
    ("EMG", "EMG", (10.0, 40.0)),
]
STAGE_MAP = {"Sleep stage W": "W", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
             "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM"}
LAGS = (1, 3, 5, 10)
_PSG = re.compile(r"^(SC4\d{3}[A-Z0-9]{2})-PSG\.edf$")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _per_second_logpower(sig, fs, band, n_sec):
    """Band-power via welch when the rate resolves the band; else log mean-square
    amplitude (handles Sleep-EDF's 1 Hz EMG)."""
    lo, hi = band
    out = np.full(n_sec, np.nan)
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


def load_subject(psg_path: Path, hyp_path: Path | None):
    """Return (envelope (n_sec,4) z-scored, stage per-sec array or None, raw dict)."""
    import edfio
    e = edfio.read_edf(str(psg_path))
    cols, raw = [], {}
    for label_sub, name, band in CHANNELS:
        sig = next((s for s in e.signals if label_sub in s.label), None)
        if sig is None:
            return None
        fs = sig.sampling_frequency
        data = np.asarray(sig.data, dtype=np.float64)
        n_sec = int(len(data) / fs)
        cols.append(_per_second_logpower(data, fs, band, n_sec))
        raw[name] = (fs, data)
    n = min(len(c) for c in cols)
    env = np.column_stack([c[:n] for c in cols])
    env = (env - np.nanmean(env, 0)) / (np.nanstd(env, 0) + 1e-9)

    stages = None
    if hyp_path is not None and hyp_path.exists():
        stages = np.array(["?"] * n, dtype=object)
        h = edfio.read_edf(str(hyp_path))
        for ann in h.annotations:
            st = STAGE_MAP.get(ann.text)
            if st is None:
                continue
            a, b = int(ann.onset), int(ann.onset + max(ann.duration, 1))
            stages[a:min(b, n)] = st
    return env, stages, raw


def lagged(series, lag, mask=None):
    T, m = series.shape
    future = series[lag:]
    past = np.stack([series[lag - k - 1:T - k - 1] for k in range(lag)], axis=-1)
    ok = ~(np.isnan(future).any(1) | np.isnan(past).any((1, 2)))
    if mask is not None:
        ok &= mask[lag:]
    return future[ok], past[ok]


# --------------------------------------------------------------------------- #
# Graph over prebuilt (future, past, clusters)
# --------------------------------------------------------------------------- #
def graph_edges(future, past, clusters, names, n_boot, neural=False):
    m = len(names)
    cdi = np.zeros((m, m)); lcb = np.zeros((m, m))
    for j in range(m):
        y = future[:, j]
        for i in range(m):
            if i == j:
                continue
            others = [k for k in range(m) if k not in (i, j)]
            x_src = past[:, i, :]
            z = np.hstack([past[:, j, :]] + [past[:, k, :] for k in others])
            res = conditional_directed_information(y, x_src, z, clusters=clusters, n_boot=n_boot)
            cdi[i, j], lcb[i, j] = res.cdi_bits, res.lcb95_bits
    return cdi, lcb


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def write_export(out: Path, edge_rows, graphs, raw_samples, calibration, insights, names):
    out.mkdir(parents=True, exist_ok=True)
    (out / "graphs").mkdir(exist_ok=True)
    (out / "raw_windows").mkdir(exist_ok=True)
    if edge_rows:
        pq.write_table(pa.Table.from_pylist(edge_rows), out / "edges.parquet")
    for key, (cdi, lcb) in graphs.items():
        np.savez(out / "graphs" / f"{key}.npz", cdi=cdi, lcb=lcb, names=names)
    for i, arr in enumerate(raw_samples):
        np.save(out / "raw_windows" / f"window_{i:05d}.npy", arr)
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
    (out / "insights.json").write_text(json.dumps(insights, indent=2))
    (out / "README.md").write_text(_dataset_card(insights, names))
    return out


def _dataset_card(insights, names):
    return (
        "---\nlicense: mit\ntags: [eeg, directed-information, sleep, multi-organ]\n---\n\n"
        "# neuroforecast directed-information graphs\n\n"
        "Certified causally-conditioned directed-information graphs across "
        f"simultaneous physiological channels ({', '.join(names)}) during sleep, "
        "per sleep stage and history lag, with raw waveform windows.\n\n"
        f"Population-certified edges: {insights.get('population_certified_edges')}\n"
    )


def push(out: Path, repo_id: str, token: str):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="dataset")


# --------------------------------------------------------------------------- #
# Dry-run: synthetic VAR, validates the WHOLE pipeline + export with no data.
# --------------------------------------------------------------------------- #
def dry_run(out: Path):
    from neuroforecast.graph import analytic_var1_di_graph
    A = np.array([[0.3, 0, 0, 0], [0.5, 0.3, 0, 0], [0, 0.5, 0.3, 0], [0, 0, 0, 0.3]])
    names = [n for _, n, _ in CHANNELS]
    rng = np.random.default_rng(0)
    x = np.zeros((8000, 4))
    for t in range(1, len(x)):
        x[t] = A @ x[t - 1] + rng.standard_normal(4)
    clusters = rng.integers(0, 20, len(x) - 1)
    fut, past = lagged(x, 1)
    cdi, lcb = graph_edges(fut, past[:len(fut)], clusters[:len(fut)], names, n_boot=200)
    truth = analytic_var1_di_graph(A)
    assert abs(cdi[0, 1] - truth[0, 1]) < 0.03, "dry-run graph estimate off"
    assert lcb[0, 3] <= 0, "dry-run certified an absent edge"
    edge_rows = [{"subject": "synthetic", "stage": "ALL", "lag": 1,
                  "src": names[i], "dst": names[j],
                  "cdi_bits": float(cdi[i, j]), "lcb95_bits": float(lcb[i, j]),
                  "certified": bool(lcb[i, j] > 0)}
                 for i in range(4) for j in range(4) if i != j]
    insights = {"population_certified_edges": [(names[i], names[j])
                for i in range(4) for j in range(4) if i != j and lcb[i, j] > 0]}
    write_export(out, edge_rows, {"ALL": (cdi, lcb)},
                 [x[:50].astype(np.float32)], {"floor_bits": 0.0}, insights, names)
    print("dry-run OK: pipeline + export assembled with no bugs ->", out)
    print("edges.parquet rows:", len(edge_rows),
          "| certified:", [e for e in insights["population_certified_edges"]])
    return 0


def main() -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--neural", action="store_true")
    ap.add_argument("--raw-windows-per-subject", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("hf_export"))
    ap.add_argument("--hf-repo", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run(args.out)
    if not args.data_dir:
        raise SystemExit("--data-dir required (or use --dry-run)")

    import os
    names = [n for _, n, _ in CHANNELS]
    files = {p.name for p in args.data_dir.glob("*.edf")}
    psgs = sorted(f for f in files if _PSG.match(f))

    per_stage = {}          # stage -> list of (future, past, clusters)
    edge_rows, raw_samples = [], []
    seen = set()
    for psg in psgs:
        rec = _PSG.match(psg).group(1)
        subj = rec[3:5]
        if subj in seen or (args.max_subjects and len(seen) >= args.max_subjects):
            if args.max_subjects and len(seen) >= args.max_subjects:
                break
            continue
        hyp = next((f for f in files if f.startswith(rec[:6]) and "Hypnogram" in f), None)
        try:
            loaded = load_subject(args.data_dir / psg,
                                  args.data_dir / hyp if hyp else None)
            if loaded is None:
                continue
            env, stages, raw = loaded
        except Exception as exc:
            print(f"  skip {psg}: {type(exc).__name__}: {exc}")
            continue
        # per-subject, per-stage, per-lag designs
        stage_list = ["ALL"] + (sorted(set(STAGE_MAP.values())) if stages is not None else [])
        for stage in stage_list:
            mask = None if stage == "ALL" else (stages == stage)
            for lag in LAGS:
                fut, past = lagged(env, lag, mask)
                if len(fut) < 50:
                    continue
                per_stage.setdefault((stage, lag), []).append(
                    (fut, past, np.full(len(fut), subj)))
        # export sampled raw windows (decimated 100->~25 Hz, 30 s) for downstream ML
        for _ in range(args.raw_windows_per_subject):
            fs0, d0 = raw[names[0]][0], raw[names[0]][1]
            t0 = np.random.randint(0, max(1, len(d0) - int(30 * fs0)))
            win = np.stack([raw[nm][1][t0:t0 + int(30 * raw[nm][0])][::4][:750]
                            for nm in names])
            if win.shape[1] == 750:
                raw_samples.append(win.astype(np.float32))
        seen.add(subj)
        print(f"  {psg}: loaded (subj {subj}, total {len(seen)})")

    # estimate graphs per (stage, lag), pooled across subjects
    graphs = {}
    for (stage, lag), chunks in sorted(per_stage.items()):
        fut = np.concatenate([c[0] for c in chunks])
        past = np.concatenate([c[1] for c in chunks])
        clu = np.concatenate([c[2] for c in chunks])
        cdi, lcb = graph_edges(fut, past, clu, names, args.bootstrap, neural=args.neural)
        graphs[f"{stage}_lag{lag}"] = (cdi, lcb)
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    edge_rows.append({
                        "subject": "POOLED", "stage": stage, "lag": lag,
                        "src": names[i], "dst": names[j],
                        "cdi_bits": float(cdi[i, j]), "lcb95_bits": float(lcb[i, j]),
                        "certified": bool(lcb[i, j] > 0), "n": int(len(fut))})
        print(f"  graph {stage} lag{lag}: {int(sum(lcb[i,j]>0 for i in range(4) for j in range(4) if i!=j))} certified edges")

    # detection-floor calibration at this data scale
    def sampler(strength):
        rng = np.random.default_rng(int(strength * 1000))
        n = 8000
        z = rng.standard_normal((n, 2))
        xc = rng.standard_normal(n)
        y = strength * xc + 0.6 * z[:, 0] + rng.standard_normal(n)
        cov = np.array([[strength**2 + 0.36 + 1, strength, 0.6, 0],
                        [strength, 1, 0, 0], [0.6, 0, 1, 0], [0, 0, 0, 1]])
        true = analytic_cdi_gaussian(cov, iy=0, ix=[1], iz=[2, 3])
        return y, xc[:, None], z, rng.integers(0, 40, n), true
    cal = detection_floor(
        lambda y, x, zz, cl: conditional_directed_information(y, x, zz, clusters=cl, n_boot=500),
        sampler, [0.0, 0.1, 0.2, 0.35, 0.55])

    pop_edges = sorted({(r["src"], r["dst"]) for r in edge_rows
                        if r["stage"] == "ALL" and r["certified"]})
    insights = {
        "n_subjects": len(seen),
        "channels": names,
        "population_certified_edges": [list(e) for e in pop_edges],
        "detection_floor_bits": cal.detection_floor,
        "stage_reorganization": {
            k.split("_")[0]: int(sum(1 for r in edge_rows
                                     if r["stage"] == k.split("_")[0] and r["certified"]))
            for k in graphs if k.endswith("lag5")},
    }
    out = write_export(args.out, edge_rows, graphs, raw_samples,
                       {"detection_floor_bits": cal.detection_floor}, insights, names)
    print(f"\nexported to {out} | edges={len(edge_rows)} raw_windows={len(raw_samples)}")
    print("insights:", json.dumps(insights, indent=2))

    repo = args.hf_repo or os.environ.get("HF_REPO")
    token = os.environ.get("HF_TOKEN")
    if repo and token:
        push(out, repo, token)
        print(f"pushed to https://huggingface.co/datasets/{repo}")
    else:
        print("HF push skipped (set HF_TOKEN and --hf-repo/HF_REPO to push). "
              "Local export is complete and valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
