# neuroforecast — cluster run guide

Self-contained. No private GitHub, no secrets. Verify, smoke, then run.

## 0. Verify the package
```bash
shasum -a 256 -c SHA256SUMS      # inside the extracted runner folder
cat COMMIT_HASH.txt
```

## 1. Environment
Preferred (conda):
```bash
conda create -n neuroforecast python=3.11 -y && conda activate neuroforecast
pip install -r requirements.txt
export PYTHONPATH=.
```

Docker fallback (no conda needed) — image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`:
```bash
PERSISTENT_ROOT="/raid/scratch/$USER/neuroforecast"
docker run --rm --gpus '"device=0"' --ipc=host --shm-size=32g \
  -v "$PWD":/workspace/repo -v "$PERSISTENT_ROOT":"$PERSISTENT_ROOT" \
  -w /workspace/repo pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc 'pip install -r requirements.txt && PYTHONPATH=. python -c "import neuroforecast; print(\"ok\")"'
```

## 2. Smoke (must pass, no data / no GPU needed)
Validates the full estimator + export pipeline on synthetic ground truth:
```bash
PYTHONPATH=. python experiments/run_a100_intensive.py --dry-run --out /tmp/nf_dry
# expect: "dry-run OK: pipeline + export assembled with no bugs"
PYTHONPATH=. python tests/test_linear.py     # analytic CDI recovery
PYTHONPATH=. python tests/test_graph.py      # directed-graph, direct-vs-mediated
```

## 3. Full run (the intensive job)
Needs a Sleep-EDF cassette dir on the cluster (or any PSG with EEG/EOG/EMG).
```bash
export HF_TOKEN=<your token, set by you in the env — never in this package>
PYTHONPATH=. python experiments/run_a100_intensive.py \
  --data-dir /path/to/sleep-edfx/sleep-cassette \
  --neural \
  --hf-repo <your-hf-user>/neuroforecast-digraph \
  --out "$PERSISTENT_ROOT/hf_export"
```
Outputs (also auto-pushed to the HF dataset if HF_TOKEN + --hf-repo set):
`edges.parquet`, `graphs/*.npz`, `raw_windows/*.npy`, `insights.json`, `calibration.json`, dataset card.

## 4. GPU usage — honest
- The **linear** CDI / directed-graph estimators are CPU (fast); a single GPU is
  plenty for the **neural** TCN estimator.
- This job is **not DDP/lockstep**. Multi-GPU here means **data parallelism across
  independent work** (subjects × stages × edges), not a synchronized all-reduce
  training run. Free GPUs help by running more shards concurrently, not by
  needing a 6-way collective.
- To use several GPUs, shard subjects across processes, one GPU each:
```bash
# 6 shards, one GPU per shard (example; --shard/--n-shards is the parallelism knob)
for g in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=. python experiments/run_a100_intensive.py \
    --data-dir <cassette> --neural --shard $g --n-shards 6 \
    --out "$PERSISTENT_ROOT/hf_export_shard$g" &
done; wait
```
(Single-GPU is fully sufficient; the shard flags are an optional speedup.)

## 5. Do NOT include secrets
`HF_TOKEN` is read from the environment at run time. It is never in this package.
No `.env`, `*.key`, `*.pem`, tokens.
