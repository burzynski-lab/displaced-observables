#!/bin/bash
# Files on disk per stage, queue state, and any failed tasks.
set -uo pipefail
set +o pipefail          # several stages pipe into head; SIGPIPE must not abort
REPO=/home/jburzyns/displaced-observables
cd "$REPO"

echo "=== queue ==="
squeue -u "$USER" -o '%.18i %.14j %.2t %.10M %R' | head -15
printf 'pending/running tasks: %s\n\n' "$(squeue -u "$USER" -h -r | wc -l)"

echo "=== files per stage ==="
for stage in events observables; do
    for pat in 'signal_ctau*mm_seed*' 'qcd_seed*' 'qcdbb_seed*'; do
        printf '  %-13s %-24s %4d\n' "$stage" "$pat" \
            "$(find -L "data/$stage" -maxdepth 1 -name "$pat.parquet" -size +0 2>/dev/null | wc -l)"
    done
done
printf '  %-13s %s\n' "features" "$(ls -1 data/features/*.h5 2>/dev/null | wc -l) file(s)"
printf '  %-13s %s\n' "models"   "$(ls -1d models/*/ 2>/dev/null | wc -l) dir(s)"
printf '  %-13s %s\n' "results"  "$(ls -1 results/*.{json,parquet,h5,npz} 2>/dev/null | wc -l) file(s)"
printf '  %-13s %s\n' "figures"  "$(ls -1 results/figures/*.pdf 2>/dev/null | wc -l) pdf"

echo
echo "=== non-COMPLETED tasks since yesterday ==="
sacct -u "$USER" -S "$(date -d yesterday +%Y-%m-%d)" -n -X \
      --format=JobID%16,JobName%14,State%12,Elapsed 2>/dev/null \
    | grep -Ev 'COMPLETED|RUNNING|PENDING' | head -10 || echo "  none"
echo
echo "refill gaps with: ./slurm/manifest.sh --todo && ./slurm/submit.sh"
