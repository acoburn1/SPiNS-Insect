#!/usr/bin/env bash
set -euo pipefail

DATA_CFG=${1:?usage: run_sweep.sh <data_cfg_yaml> <model_cfg_yaml>}
MODEL_CFG=${2:?usage: run_sweep.sh <data_cfg_yaml> <model_cfg_yaml>}

REPO="/home/acoburn1/projects/SPiNS/SPiNS-Insect/lab_tests"
PY="$REPO/program.py"

# Get grid size (no SLURM yet)
N=$(cd "$REPO" && conda run -n labexp python "$PY" \
  --data-cfg "$DATA_CFG" \
  --model-cfg "$MODEL_CFG" \
  --print-grid-size)

ARRAY_MAX=$((N - 1))
echo "Submitting sweep: N=$N (array 0-$ARRAY_MAX)"

sbatch \
  --export=ALL,DATA_CFG="$DATA_CFG",MODEL_CFG="$MODEL_CFG" \
  --array=0-"$ARRAY_MAX" \
  "$REPO/sweep.sbatch"
