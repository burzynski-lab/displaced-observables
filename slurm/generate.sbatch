#!/bin/bash
# One array task = one 10k-event parquet file.
#
# Do not sbatch this directly, use ./slurm/submit.sh, which sets the array
# range from the manifest length and picks the walltime for the sample.
#
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=/scratch/jburzyns/displaced-observables/logs/%x_%A_%a.out
# stderr goes to its own file: the tqdm progress bar is written there one line
# per event, which would otherwise bury the useful output in ~300 kB of noise.
# Tracebacks land here too, so this is the file to read when a task fails.
#SBATCH --error=/scratch/jburzyns/displaced-observables/logs/%x_%A_%a.err

set -euo pipefail

REPO=/home/jburzyns/displaced-observables
cd "$REPO"
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache

NEVENTS=${NEVENTS:-10000}
MANIFEST=${MANIFEST:?set MANIFEST}

read -r SAMPLE CTAU SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[[ -n ${SAMPLE:-} ]] || { echo "no manifest line $SLURM_ARRAY_TASK_ID in $MANIFEST"; exit 1; }

ARGS=(--sample "$SAMPLE" --nevents "$NEVENTS" --seed "$SEED" --out "$REPO/data")
[[ $CTAU != - ]] && ARGS+=(--ctau "$CTAU")

echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
echo "sample=$SAMPLE ctau=$CTAU seed=$SEED nevents=$NEVENTS"
date

srun pixi run generate "${ARGS[@]}"

date
echo "done"
