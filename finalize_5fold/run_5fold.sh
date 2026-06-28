#!/usr/bin/env bash
# Official 5-fold finalization: champion train.py verbatim (SCHED_TMAX mult 34) at 9000s/fold.
# Usage: ./run_5fold.sh 0          (fold-0-first) ; later: ./run_5fold.sh 1 2 3 4
set -u
cd /work/autohive_runs/2026-06-19_mpgap_finalize
source .venv/bin/activate
SUMMARY=summary_5fold.tsv
[ -f "$SUMMARY" ] || printf "fold\tmae\tsteps\ttrain_s\tfeat_s\n" > "$SUMMARY"
for f in "$@"; do
  echo "=== fold $f START $(date) ==="
  MPGAP_FOLD="$f" MPGAP_TIME_BUDGET=9000 python train_5fold.py > "fold${f}.log" 2>&1
  mae=$(grep "^mae:" "fold${f}.log" | awk "{print \$2}")
  steps=$(grep "^steps:" "fold${f}.log" | awk "{print \$2}")
  ts=$(grep "^training_seconds:" "fold${f}.log" | awk "{print \$2}")
  fs=$(grep "^feat_seconds:" "fold${f}.log" | awk "{print \$2}")
  printf "%s\t%s\t%s\t%s\t%s\n" "$f" "$mae" "$steps" "$ts" "$fs" >> "$SUMMARY"
  echo "=== fold $f DONE mae=$mae $(date) ==="
done
echo "=== ALL REQUESTED FOLDS DONE $(date) ==="
