#!/bin/bash
# Full analysis chain: all plots, exports, and ML trainings.
# Run from the repo root:  bash scripts/run_full_chain.sh
# Logs land in plots/*.log; exits non-zero on the first failure.
set -e
cd "$(dirname "$0")/.."

echo "=== plot suites (3 scenarios) ==="
for s in truth standard standard_lrt; do
    echo "--- make_plots $s"
    pixi run python scripts/make_plots.py --scenario $s > plots/rejection_$s.log 2>&1
done

echo "=== scenario comparison ==="
pixi run python scripts/scenario_comparison.py > plots/scenario_comparison.log 2>&1

echo "=== kappa-beta scan ==="
pixi run python scripts/kappa_beta_scan.py --ctau 3 30 > plots/kb_scan.log 2>&1

echo "=== track IP distributions ==="
pixi run python scripts/track_ip_plots.py --scenario truth > plots/track_ip.log 2>&1

echo "=== pileup study ==="
pixi run python scripts/pileup_study.py > plots/pileup_study.log 2>&1

echo "=== feature + track exports ==="
pixi run -e ml python scripts/export_features.py --scenario truth > plots/export_features.log 2>&1
pixi run -e ml python scripts/export_tracks.py --scenario truth > plots/export_tracks.log 2>&1

echo "=== AE/VAE study ==="
pixi run -e ml python scripts/vae_study.py --scenario truth > plots/vae_study.log 2>&1

echo "=== transformer ceiling study ==="
pixi run -e ml python scripts/transformer_study.py --scenario truth > plots/transformer_study.log 2>&1

echo "=== paper ==="
pixi run paper > /dev/null 2>&1

echo "FULL CHAIN COMPLETE"
