#!/bin/bash
# Sharded make_plots: fan out one task per input file, then merge and plot.
#
#   ./slurm/analysis.sh                # truth scenario, one shard per file
#   SCENARIO=truth NSHARDS=90 ./slurm/analysis.sh
#
# The merge job runs under --dependency=afterok, so it starts only if every
# shard succeeded. If any shard fails the merge never runs, which is the
# intent: a partial merge would silently drop events.
set -euo pipefail

REPO=/home/jburzyns/displaced-observables
cd "$REPO"
mkdir -p /scratch/jburzyns/displaced-observables/logs

SCENARIO=${SCENARIO:-truth}
MAXRUN=${MAXRUN:-100}
# Default: one shard per input parquet file.
NSHARDS=${NSHARDS:-$(find -L data -maxdepth 1 -name '*.parquet' -size +0 | wc -l)}

echo "scenario=$SCENARIO nshards=$NSHARDS"
rm -rf "data/cache/shards_${SCENARIO}"

jid=$(sbatch --parsable \
      --array="0-$((NSHARDS - 1))%${MAXRUN}" \
      --export=ALL,SCENARIO="$SCENARIO",NSHARDS="$NSHARDS" \
      "$REPO/slurm/shard_plots.sbatch")
echo "shard array: $jid"

mid=$(sbatch --parsable \
      --dependency=afterok:"$jid" \
      --export=ALL,SCENARIO="$SCENARIO",NSHARDS="$NSHARDS" \
      "$REPO/slurm/merge_plots.sbatch")
echo "merge job:   $mid  (runs only if all shards succeed)"
