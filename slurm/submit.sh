#!/bin/bash
# Submit one array job per sample from the manifests.
#
# Usage:
#   ./slurm/submit.sh                     # all three samples
#   ./slurm/submit.sh signal              # just one
#   MAXRUN=200 ./slurm/submit.sh qcd      # throttle concurrent tasks
#
# The account allows 1500 submitted jobs at a time and the full production is
# 270 tasks, so everything fits in one pass. MAXRUN throttles how many run
# concurrently (%N in the array spec), which is polite on a shared partition.
set -euo pipefail

REPO=/home/jburzyns/displaced-observables
DIR=$REPO/slurm
mkdir -p /scratch/jburzyns/displaced-observables/logs

MAXRUN=${MAXRUN:-100}
NEVENTS=${NEVENTS:-10000}

# Walltime per sample. Calibrated on sooner_test at 500 events/task:
# signal 0.31 s/evt, qcd 0.17 s/evt, qcd_bb 0.36 s/evt, so a 10k task is
# 30-60 min. These limits carry ~4x headroom for slow nodes.
walltime() {
    case $1 in
        signal) echo 03:00:00 ;;
        qcd)    echo 02:00:00 ;;
        qcd_bb) echo 04:00:00 ;;
    esac
}

SAMPLES=${*:-signal qcd qcd_bb}

for s in $SAMPLES; do
    m=$DIR/manifest_$s.txt
    [[ -s $m ]] || { echo "manifest $m is empty or missing, run ./slurm/manifest.sh"; continue; }
    n=$(wc -l < "$m")
    echo -n "$s: $n tasks -> "
    sbatch --job-name="gen-$s" \
           --array="1-${n}%${MAXRUN}" \
           --time="$(walltime "$s")" \
           --export=ALL,MANIFEST="$m",NEVENTS="$NEVENTS" \
           "$DIR/generate.sbatch"
done
