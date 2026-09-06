#!/bin/bash
# Run one `displaced-observables` command on the CPU partition.
#   sbatch slurm/cpu.sbatch analyze --all
#   sbatch slurm/cpu.sbatch evaluate --only rejections
#SBATCH --job-name=do-cpu
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/displaced-observables/logs/%x_%j.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/displaced-observables/logs/%x_%j.err
set -euo pipefail
REPO=/home/jburzyns/displaced-observables
cd "$REPO"
export WORKDIR="$REPO"
export PIXI_CACHE_DIR=/ourdisk/hpc/ouhep/jburzyns/dont_archive/pixi-cache
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
ENVF=${ENVF:-default}
echo "host=$(hostname) job=$SLURM_JOB_ID env=$ENVF task=$*"; date
srun --chdir="$REPO" pixi run --manifest-path "$REPO/pixi.toml" -e "$ENVF" displaced-observables "$@"
date; echo done
