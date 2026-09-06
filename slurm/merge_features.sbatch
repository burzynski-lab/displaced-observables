#!/bin/bash
# Merge feature shards into features_<scenario>.h5 + norm YAML.
#
#SBATCH --job-name=merge-feat
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jburzyns/displaced-observables/logs/%x_%j.out
#SBATCH --error=/scratch/jburzyns/displaced-observables/logs/%x_%j.err

set -euo pipefail
cd /home/jburzyns/displaced-observables
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache

SCENARIO=${SCENARIO:-truth}
NSHARDS=${NSHARDS:?set NSHARDS}

mkdir -p plots
# --merge itself refuses a partial set when --nshards is given.
srun pixi run -e ml python scripts/export_features.py \
     --scenario "$SCENARIO" --merge --nshards "$NSHARDS" \
     | tee "plots/export_features.log"
