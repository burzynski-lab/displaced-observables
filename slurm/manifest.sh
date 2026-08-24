#!/bin/bash
# Build (or rebuild) the per-sample job manifests.
#
# Each manifest has one line per 10k-event job: "<sample> <ctau|-> <seed>".
# The array index into the manifest is the SLURM_ARRAY_TASK_ID.
#
# Usage:
#   ./slurm/manifest.sh          # full manifests
#   ./slurm/manifest.sh --todo   # only jobs whose output parquet is missing
set -euo pipefail

REPO=/home/jburzyns/displaced-observables
DATA=$REPO/data
DIR=$REPO/slurm
TODO_ONLY=${1:-}

CTAUS="0 1 3 10 30 100 300"
SIGNAL_SEEDS=10    # 10 x 10k = 100k per ctau point
QCD_SEEDS=100      # 100 x 10k = 1M
QCDBB_SEEDS=100    # 100 x 10k = 1M

# outfile <sample> <ctau>  <seed>  ->  path generate_sample() will write
outfile() {
    case $1 in
        signal) printf '%s/signal_ctau%gmm_seed%d.parquet' "$DATA" "$2" "$3" ;;
        qcd)    printf '%s/qcd_seed%d.parquet'   "$DATA" "$3" ;;
        qcd_bb) printf '%s/qcdbb_seed%d.parquet' "$DATA" "$3" ;;
    esac
}

emit() {  # emit <sample> <ctau> <seed>
    if [[ $TODO_ONLY == --todo ]] && [[ -s $(outfile "$1" "$2" "$3") ]]; then
        return
    fi
    echo "$1 $2 $3"
}

for s in $(seq 1 $SIGNAL_SEEDS); do
    for c in $CTAUS; do emit signal "$c" "$s"; done
done > "$DIR/manifest_signal.txt"

for s in $(seq 1 $QCD_SEEDS);   do emit qcd    - "$s"; done > "$DIR/manifest_qcd.txt"
for s in $(seq 1 $QCDBB_SEEDS); do emit qcd_bb - "$s"; done > "$DIR/manifest_qcd_bb.txt"

for f in "$DIR"/manifest_*.txt; do
    printf '%-40s %5d jobs\n' "$f" "$(wc -l < "$f")"
done
