#!/bin/bash
#SBATCH --job-name=labtest_gpu
#SBATCH --output=slurm_results/%x_%j.out
#SBATCH --error=slurm_results/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-t4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labexp

cd ~/projects/SPiNS/SPiNS-Insect/lab_tests

echo "HOST: $(hostname)"
nvidia-smi

python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

python program.py
