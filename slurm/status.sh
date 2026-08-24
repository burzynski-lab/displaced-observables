#!/bin/bash
# Production progress: queue state, files on disk, and any failed tasks.
# No pipefail here: several stages pipe into `head`, and the resulting SIGPIPE
# would abort the script before it ever reported the file counts.
set -uo pipefail
set +o pipefail

REPO=/home/jburzyns/displaced-observables
DATA=$REPO/data

echo "=== queue ==="
squeue -u "$USER" -o '%.18i %.12j %.2t %.10M %.6D %R' | head -20
printf 'pending/running tasks: %s\n' "$(squeue -u "$USER" -h -r | wc -l)"

echo
echo "=== files on disk (expect 70 signal / 100 qcd / 100 qcdbb) ==="
for pat in 'signal_ctau*mm_seed*.parquet' 'qcd_seed*.parquet' 'qcdbb_seed*.parquet'; do
    # shellcheck disable=SC2086
    printf '%-34s %4d files %8s\n' "$pat" \
        "$(find -L "$DATA" -maxdepth 1 -name "$pat" -size +0 2>/dev/null | wc -l)" \
        "$(find -L "$DATA" -maxdepth 1 -name "$pat" -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {printf "%.1fG", s/1073741824}')"
done

echo
echo "=== signal breakdown by ctau ==="
for c in 0 1 3 10 30 100 300; do
    printf '  ctau=%-5s %3d/10\n' "$c" "$(find -L "$DATA" -maxdepth 1 -name "signal_ctau${c}mm_seed*.parquet" -size +0 | wc -l)"
done

echo
echo "=== non-COMPLETED tasks in the last day ==="
sacct -u "$USER" -S "$(date -d yesterday +%Y-%m-%d)" -n -X \
      --format=JobID%18,JobName%12,State%12,Elapsed,ExitCode 2>/dev/null \
    | grep -Ev 'COMPLETED|RUNNING|PENDING' || echo "  none"

echo
echo "If any are missing: ./slurm/manifest.sh --todo && ./slurm/submit.sh"
