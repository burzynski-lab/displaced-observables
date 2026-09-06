#!/bin/bash
#SBATCH --job-name=gen-calib
#SBATCH --partition=sooner_test
#SBATCH --array=1-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/jburzyns/displaced-observables/logs/calib_%A_%a.out
#SBATCH --error=/scratch/jburzyns/displaced-observables/logs/calib_%A_%a.out

set -euo pipefail
cd /home/jburzyns/displaced-observables
export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache

N=500
OUT=/scratch/jburzyns/displaced-observables/calib
mkdir -p "$OUT"

case $SLURM_ARRAY_TASK_ID in
  1) ARGS="--sample signal --ctau 10" ;;
  2) ARGS="--sample qcd" ;;
  3) ARGS="--sample qcd_bb" ;;
esac

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID args=$ARGS nevents=$N"
/usr/bin/time -v pixi run generate $ARGS --nevents $N --seed 9001 --out "$OUT"
