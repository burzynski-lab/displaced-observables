#!/bin/bash
# Build the generation manifests: one line per output file, "<sample> <ctau|-> <seed>".
#   ./slurm/manifest.sh          # full production
#   ./slurm/manifest.sh --todo   # only the points whose observable parquet is missing
set -euo pipefail
REPO=/home/jburzyns/displaced-observables
cd "$REPO"
TODO=${1:-}

# pTHatMin was lowered 450 -> 400 so the 500 GeV jet cut is not sitting on the
# generator turn-on. Measured selection efficiency drops 1.203 -> 0.726 selected
# jets per generated event, so the background seed counts rise by 1.66x to hold
# the jet statistics of the previous production.
CTAUS="0 1 3 10 30 100 300"
SIGNAL_SEEDS=${SIGNAL_SEEDS:-10}
QCD_SEEDS=${QCD_SEEDS:-165}
QCDBB_SEEDS=${QCDBB_SEEDS:-165}

stem() {   # stem <sample> <ctau> <seed>
    case $1 in
        signal) printf 'signal_ctau%gmm_seed%d' "$2" "$3" ;;
        qcd)    printf 'qcd_seed%d' "$3" ;;
        qcd_bb) printf 'qcdbb_seed%d' "$3" ;;
    esac
}
emit() {   # skip points that already have BOTH stages done
    if [[ $TODO == --todo ]] && [[ -s "data/observables/$(stem "$1" "$2" "$3").parquet" ]]; then
        return
    fi
    echo "$1 $2 $3"
}

for s in $(seq 1 "$SIGNAL_SEEDS"); do
    for c in $CTAUS; do emit signal "$c" "$s"; done
done > slurm/manifest_signal.txt
for s in $(seq 1 "$QCD_SEEDS");   do emit qcd    - "$s"; done > slurm/manifest_qcd.txt
for s in $(seq 1 "$QCDBB_SEEDS"); do emit qcd_bb - "$s"; done > slurm/manifest_qcd_bb.txt

for f in slurm/manifest_*.txt; do
    printf '%-34s %5d tasks\n' "$f" "$(wc -l < "$f")"
done
