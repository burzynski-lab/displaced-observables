#!/bin/bash
# Submit one array per manifest. Each task generates a file and analyzes it.
#   ./slurm/submit.sh                  # all manifests
#   ./slurm/submit.sh signal           # one
#   NEVENTS=10000 MAXRUN=150 ./slurm/submit.sh
set -euo pipefail
REPO=/home/jburzyns/displaced-observables
cd "$REPO"
MAXRUN=${MAXRUN:-100}
NEVENTS=${NEVENTS:-10000}

for s in ${*:-signal qcd qcd_bb}; do
    m="slurm/manifest_$s.txt"
    [[ -s $m ]] || { echo "$m is empty or missing, run ./slurm/manifest.sh"; continue; }
    n=$(wc -l < "$m")
    printf '%-8s %4d tasks -> ' "$s" "$n"
    sbatch --job-name="gen-$s" --array="1-${n}%${MAXRUN}" \
           --export=ALL,MANIFEST="$REPO/$m",NEVENTS="$NEVENTS" \
           slurm/generate.sbatch
done
