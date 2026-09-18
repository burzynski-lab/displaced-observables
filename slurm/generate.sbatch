#!/bin/bash
# One array task = generate one truth-record file and analyze it.
# Submitted by ./slurm/submit.sh, which sets MANIFEST and the array range.
#SBATCH --partition=sooner_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=/ourdisk/hpc/ouhep/jburzyns/dont_archive/displaced-observables/logs/%x_%A_%a.out
#SBATCH --error=/ourdisk/hpc/ouhep/jburzyns/dont_archive/displaced-observables/logs/%x_%A_%a.err
set -euo pipefail
REPO=/home/jburzyns/displaced-observables
cd "$REPO"
export WORKDIR="$REPO"
export PIXI_CACHE_DIR=/ourdisk/hpc/ouhep/jburzyns/dont_archive/pixi-cache
export PYTHONUNBUFFERED=1

NEVENTS=${NEVENTS:-10000}
MANIFEST=${MANIFEST:?set MANIFEST}
read -r SAMPLE CTAU SEED < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
[[ -n ${SAMPLE:-} ]] || { echo "no manifest line $SLURM_ARRAY_TASK_ID"; exit 1; }

ARGS=(--sample "$SAMPLE" --nevents "$NEVENTS" --seed "$SEED" --quiet)
[[ $CTAU != - ]] && ARGS+=(--ctau "$CTAU")
echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID sample=$SAMPLE ctau=$CTAU seed=$SEED"; date

GENLOG=$(mktemp)
srun --chdir="$REPO" pixi run --manifest-path "$REPO/pixi.toml" \
     displaced-observables generate "${ARGS[@]}" | tee "$GENLOG"
# chain on the filename the generate step printed, so the two stages cannot
# disagree about where the file went
EVENTS=$(sed -n 's/^wrote \(.*\.parquet\).*/\1/p' "$GENLOG" | tail -1); rm -f "$GENLOG"
[[ -n $EVENTS ]] || { echo "generate did not report an output file"; exit 1; }
srun --chdir="$REPO" pixi run --manifest-path "$REPO/pixi.toml" \
     displaced-observables analyze "$EVENTS"
date; echo done
