#!/bin/bash
# Sharding equivalence check on a compute node (it is a single-core,
# ~30 minute job and does not belong on the login node).
#
#   sbatch slurm/validate.sbatch
#
#SBATCH --job-name=validate-shard
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jburzyns/displaced-observables/logs/%x_%j.out
#SBATCH --error=/scratch/jburzyns/displaced-observables/logs/%x_%j.err

set -euo pipefail
cd /home/jburzyns/displaced-observables
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache
export PYTHONUNBUFFERED=1

echo "host=$(hostname)"; date
srun pixi run python scripts/validate_sharding.py --nshards "${NSHARDS:-4}"
date
