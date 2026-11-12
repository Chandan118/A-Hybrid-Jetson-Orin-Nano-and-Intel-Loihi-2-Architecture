#!/usr/bin/env bash
# Re-run ONLY the parts that did not complete in the first pipeline pass:
#   Run 1  (power and energy, full reviewer protocol)  -- was interrupted
#   Run 1c (pose RMSE output-equivalence)                -- failed on GPU OOM
#   Run 1d (measurement-boundary table)                  -- re-run for a fresh
#                                                           instrument inventory
# Run 1b (stages), Run 2 (memory), Run 3 (endurance) and the lookups already
# completed and are NOT repeated.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results
LOGS=logs
mkdir -p "$RESULTS/run1" "$LOGS/run1"

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] RUN 1: power and energy (full protocol: 5 repeats, 10 min warm-up + 10 min log)"
python3 -u run1_power.py \
    --configs C1,C2,C3,C4,C5 \
    --repeats 5 \
    --warmup-s 600 \
    --log-s 600 \
    --rate 50 \
    --outdir "$RESULTS/run1" \
    --logdir "$LOGS/run1" \
    > "$LOGS/run1_full.log" 2>&1
echo "[$(stamp)] RUN 1 exit=$?"

python3 -u analyse_run1.py --rundir "$RESULTS/run1" > "$LOGS/run1_analysis.log" 2>&1
echo "[$(stamp)] RUN 1 analysis exit=$?"

echo "[$(stamp)] RUN 1c: pose RMSE output-equivalence check"
python3 -u run1c_pose_check.py \
    --configs C1,C2,C3,C4,C5 \
    --out "$RESULTS/run1/pose_equivalence.json" \
    > "$LOGS/run1c_pose_check.log" 2>&1
echo "[$(stamp)] RUN 1c exit=$?"

echo "[$(stamp)] RUN 1d: measurement-boundary table"
python3 -u run1d_boundary_table.py > "$LOGS/run1d_boundary.log" 2>&1
echo "[$(stamp)] RUN 1d exit=$?"

echo "[$(stamp)] ALL DONE"
