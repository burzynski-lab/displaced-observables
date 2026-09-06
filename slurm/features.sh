#!/bin/bash
# Sharded feature export: one task per input file, then merge to HDF5.
#
#   ./slurm/features.sh
set -euo pipefail
REPO=/home/jburzyns/displaced-observables
cd "$REPO"
mkdir -p /scratch/jburzyns/displaced-observables/logs

SCENARIO=${SCENARIO:-truth}
MAXRUN=${MAXRUN:-100}
NSHARDS=${NSHARDS:-$(find -L data -maxdepth 1 -name '*.parquet' -size +0 | wc -l)}

echo "scenario=$SCENARIO nshards=$NSHARDS"
rm -rf "data/cache/featshards_${SCENARIO}"

jid=$(sbatch --parsable --array="0-$((NSHARDS - 1))%${MAXRUN}" \
      --export=ALL,SCENARIO="$SCENARIO",NSHARDS="$NSHARDS" \
      "$REPO/slurm/shard_features.sbatch")
echo "feature shard array: $jid"

mid=$(sbatch --parsable --dependency=afterok:"$jid" \
      --export=ALL,SCENARIO="$SCENARIO",NSHARDS="$NSHARDS" \
      "$REPO/slurm/merge_features.sbatch")
echo "feature merge job:   $mid"
