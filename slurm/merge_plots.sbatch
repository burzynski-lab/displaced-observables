#!/bin/bash
# Merge the shard pickles into the plot cache and produce every figure.
# Submitted by ./slurm/analysis.sh with a dependency on the shard array.
#
#SBATCH --job-name=merge-plots
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/jburzyns/displaced-observables/logs/%x_%j.out
#SBATCH --error=/scratch/jburzyns/displaced-observables/logs/%x_%j.err

set -euo pipefail
cd /home/jburzyns/displaced-observables
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache

SCENARIO=${SCENARIO:-truth}
NSHARDS=${NSHARDS:?set NSHARDS}

# Refuse to merge a partial set: a missing shard would silently drop events.
n=$(ls -1 "data/cache/shards_${SCENARIO}"/shard_*.pkl 2>/dev/null | wc -l)
if [[ $n -ne $NSHARDS ]]; then
    echo "ERROR: found $n shard pickles, expected $NSHARDS. Refusing to merge."
    exit 1
fi
echo "merging $n shards"
mkdir -p plots
srun pixi run python scripts/make_plots.py --scenario "$SCENARIO" --merge \
     | tee "plots/rejection_${SCENARIO}.log"
