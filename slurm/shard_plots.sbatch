#!/bin/bash
# One array task = one shard of the make_plots observable pass.
# Submitted by ./slurm/analysis.sh, not directly.
#
#SBATCH --job-name=shard-plots
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/jburzyns/displaced-observables/logs/%x_%A_%a.out
#SBATCH --error=/scratch/jburzyns/displaced-observables/logs/%x_%A_%a.err

set -euo pipefail
cd /home/jburzyns/displaced-observables
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache

SCENARIO=${SCENARIO:-truth}
NSHARDS=${NSHARDS:?set NSHARDS}

echo "host=$(hostname) shard=$SLURM_ARRAY_TASK_ID/$NSHARDS scenario=$SCENARIO"
date
srun pixi run python scripts/make_plots.py \
     --scenario "$SCENARIO" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSHARDS"
date
