#!/bin/bash
# The post-generation chain as a SLURM dependency chain:
#   features -> train -> evaluate(anomaly,supervised) -> plot
# Rejections/correlations/scan do not need the GPU and go out in parallel.
#   ./slurm/chain.sh [--after <jobid>]
set -euo pipefail
REPO=/home/jburzyns/displaced-observables
cd "$REPO"
AFTER=""
if [[ ${1:-} == --after ]]; then AFTER="--dependency=afterany:$2"; shift 2; fi

j1=$(sbatch --parsable $AFTER --job-name=do-features --export=ALL,ENVF=ml slurm/cpu.sbatch features)
j2=$(sbatch --parsable --dependency=afterok:$j1 --job-name=do-train  slurm/gpu.sbatch train --all)
j3=$(sbatch --parsable --dependency=afterok:$j2 --job-name=do-anom   slurm/gpu.sbatch evaluate --only anomaly)
j4=$(sbatch --parsable --dependency=afterok:$j2 --job-name=do-sup    slurm/gpu.sbatch evaluate --only supervised)
j5=$(sbatch --parsable $AFTER --job-name=do-rej --export=ALL,ENVF=default slurm/cpu.sbatch evaluate --only rejections)
j6=$(sbatch --parsable $AFTER --job-name=do-corr --export=ALL,ENVF=default slurm/cpu.sbatch evaluate --only correlations)
j7=$(sbatch --parsable --dependency=afterok:$j3:$j4:$j5:$j6 --job-name=do-plot \
     --export=ALL,ENVF=ml slurm/cpu.sbatch plot)
echo "features $j1 -> train $j2 -> anomaly $j3 / supervised $j4"
echo "rejections $j5, correlations $j6 (parallel) -> plot $j7"
