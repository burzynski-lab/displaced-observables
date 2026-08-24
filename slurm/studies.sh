#!/bin/bash
# Phase 3: everything downstream of features_truth.h5, plus the (kappa,beta)
# scan. All are independent, so they go out as separate jobs.
#
#   ./slurm/studies.sh
set -euo pipefail
REPO=/home/jburzyns/displaced-observables
cd "$REPO"
LOGS=/scratch/jburzyns/displaced-observables/logs
mkdir -p "$LOGS" plots
SC=${SCENARIO:-truth}
PIX="export PIXI_CACHE_DIR=/scratch/jburzyns/.pixi-cache && cd $REPO &&"

sub () {  # sub <name> <cpus> <mem> <time> <partition> <command>
    local name=$1 cpus=$2 mem=$3 tl=$4 part=$5; shift 5
    sbatch --parsable --job-name="$name" --partition="$part" \
        --cpus-per-task="$cpus" --mem="$mem" --time="$tl" \
        --output="$LOGS/${name}_%j.out" --error="$LOGS/${name}_%j.err" \
        --wrap="$PIX $*"
}

echo -n "appendix_inputs:      "
sub appendix 2 24G 04:00:00 sooner_test \
    "pixi run -e ml python scripts/appendix_inputs.py --scenario $SC > plots/appendix_inputs.log 2>&1"

echo -n "correlation_matrix:   "
sub correl 2 24G 04:00:00 sooner_test \
    "pixi run -e ml python scripts/correlation_matrix.py --scenario $SC > plots/correlations.log 2>&1"

echo -n "interpretability:     "
sub interp 16 64G 12:00:00 sooner_test \
    "OMP_NUM_THREADS=16 pixi run -e ml python scripts/interpretability_study.py --scenario $SC --recompute > plots/interpretability_study.log 2>&1"

echo -n "vae_study:            "
sub vae 16 64G 2-00:00:00 sooner_test \
    "OMP_NUM_THREADS=16 pixi run -e ml python scripts/vae_study.py --scenario $SC --recompute > plots/vae_study.log 2>&1"

# Goes through load_raw_tables, so it needs the whole jet-table set in memory
# (~90 GB). The signal files are NOT restricted to the scanned ctau points,
# because pt_weights builds its reference from every signal sample and
# trimming them would silently change the weights relative to make_plots.
echo -n "kappa_beta_scan:      "
sub kbscan 4 200G 12:00:00 sooner_test_256gb_64core \
    "pixi run python scripts/kappa_beta_scan.py --scenario $SC --ctau 3 30 --eff 0.7 > plots/kb_scan.log 2>&1"
