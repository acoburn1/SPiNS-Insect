#!/bin/bash
#SBATCH --job-name=spins_sweep
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm_results/%x_%A_%a.out
#SBATCH --error=slurm_results/%x_%A_%a.err

set -euo pipefail

cd /home/acoburn1/projects/SPiNS/SPiNS-Insect/lab_tests

source ~/.bashrc
conda activate labexp

python program.py \
  --data-cfg "$DATA_CFG" \
  --model-cfg "$MODEL_CFG" \
  --task-id "$SLURM_ARRAY_TASK_ID"
