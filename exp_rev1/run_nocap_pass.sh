#!/usr/bin/env bash
# Run 2 WITHOUT the artificial address-space cap.
#
# Why this script exists
# ----------------------
# The clean-pass Run 2 applied the harness's own limit
# (resource.setrlimit(RLIMIT_AS, 3000 MB), --map-mem-limit-mb 3000).
# The failures that produced were therefore *address-space-limit failures
# raised by the harness*, not out-of-memory failures of the real 8 GB
# Jetson module.  That shows the map code needs more than 3000 MB without
# Algorithm 1; it does NOT show that Algorithm 1 prevents real Jetson
# out-of-memory crashes.
#
# This script re-runs Run 2 with --map-mem-limit-mb 0, i.e. with NO
# artificial limit, so that the only remaining limit is the physical
# 8 GB module.  The failure that then occurs is a real allocation failure
# against the real memory of the machine, and /proc/vmstat oom_kill plus
# dmesg tell us whether the kernel OOM killer was the one that fired.
#
# The same workload, the same recorded stream and the same four
# configurations are used, so the result is directly comparable with the
# capped run.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results/run2_nocap
LOGS=logs
mkdir -p "$RESULTS" "$LOGS"

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] quiescence check before the uncapped run"
python3 quiescence.py --out "$RESULTS/quiescence_before_nocap.json" \
    > "$LOGS/quiescence_before_nocap.log" 2>&1
echo "[$(stamp)] quiescence exit=$?"

echo "[$(stamp)] RUN 2 (NO ARTIFICIAL CAP): 67 recorded replays per configuration"
python3 -u run2_memory.py \
    --runs 67 --rate 50 \
    --map-mem-limit-mb 0 --compress-threshold-mb 900 \
    --outdir "$RESULTS" \
    > "$LOGS/run2_nocap.log" 2>&1
echo "[$(stamp)] RUN 2 (no cap) exit=$?"

echo "[$(stamp)] DONE"
