#!/usr/bin/env bash
# A100 cluster entry point for neuroforecast.
# Validates the estimators, then runs the neural CDI certificate on real data.
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] linear CDI validation (analytic ground truth)"
python tests/test_linear.py

echo "[2/4] neural CDI validation (must match closed form on linear data)"
python tests/test_neural.py

echo "[3/4] detection-power calibration figure"
python experiments/demo_gut_brain.py
python figures/plot_detection_power.py || true

echo "[4/4] real-data certificate (set DATA_DIR to a Sleep-EDF cassette dir)"
DATA_DIR="${DATA_DIR:-$HOME/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette}"
if [ -d "$DATA_DIR" ]; then
  python experiments/run_sleep_edf.py --data-dir "$DATA_DIR" --neural
else
  echo "  (DATA_DIR not found: $DATA_DIR -- skipping real-data run)"
fi
echo "done."
