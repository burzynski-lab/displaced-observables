#!/bin/bash
# VAE/AE anomaly-detection study on a GPU.
#
#   sbatch slurm/vae_gpu.sbatch
#   SCENARIO=truth sbatch slurm/vae_gpu.sbatch
#
# ouheptmp is a single node (c1044) with 4x L40S, 64 CPUs, 515 GB and no
# time limit. It needs a CUDA-enabled torch: the default `ml` environment
# carries the conda-forge cpu_mkl build, which reports
# torch.cuda.is_available() == False even on a GPU node. Use `mlgpu`.
#
#SBATCH --job-name=vae-gpu
#SBATCH --partition=ouheptmp
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jburzyns/displaced-observables/logs/%x_%j.out
#SBATCH --error=/scratch/jburzyns/displaced-observables/logs/%x_%j.err

set -euo pipefail
cd /home/jburzyns/displaced-observables
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache
# Without this, python block-buffers stdout when redirected and the log
# stays empty for the whole run.
export PYTHONUNBUFFERED=1

SCENARIO=${SCENARIO:-truth}
ENVNAME=${ENVNAME:-mlgpu}

echo "host=$(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date
srun pixi run -e "$ENVNAME" python scripts/vae_study.py \
     --scenario "$SCENARIO" --recompute | tee "plots/vae_study.log"
date
