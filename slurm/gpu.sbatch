#!/bin/bash
# Run one `displaced-observables` command in the CUDA environment on ouheptmp.
#   sbatch slurm/gpu.sbatch train --all
#   sbatch slurm/gpu.sbatch evaluate --only anomaly
#SBATCH --job-name=do-gpu
#SBATCH --partition=ouheptmp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/displaced-observables/logs/%x_%j.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/displaced-observables/logs/%x_%j.err
set -euo pipefail
REPO=/home/jburzyns/displaced-observables
cd "$REPO"
export WORKDIR="$REPO"
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache
export CONDA_OVERRIDE_CUDA=12.4
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "host=$(hostname) job=$SLURM_JOB_ID task=$*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date
srun --chdir="$REPO" pixi run --manifest-path "$REPO/pixi.toml" -e gpu displaced-observables "$@"
date; echo done
