#!/usr/bin/env bash
# Run 1 at a MATCHED rate of 30 Hz.
#
# Why this script exists
# ----------------------
# The 50 Hz Run 1 measurement is not a like-for-like energy comparison.
# At 50 Hz:
#
#   C1  44.59 FPS  -- drops frames
#   C2  31.24 FPS  -- drops frames
#   C3  50.00 FPS  -- holds the rate
#   C4  44.08 FPS  -- drops frames
#   C5  49.98 FPS  -- holds the rate
#
# so the joules-per-frame figures in the 50 Hz tables are energy per frame
# ACTUALLY DELIVERED, at each configuration's own delivered rate, and they
# are not directly comparable to each other.  C3 and C5 look best because
# they are the only two that finished the frame budget; C2 looks worst
# partly because it was offered frames it could not take.
#
# 30 Hz is below the slowest configuration's own maximum sustainable rate
# (C2's worst measured maximum is 31.24 FPS), so at 30 Hz EVERY
# configuration should sustain the full rate.  The joules-per-frame figures
# from this pass are then directly comparable, because all five
# configurations processed the same frames at the same rate.
#
# This pass does NOT replace the 50 Hz pass.  The 50 Hz measurement is the
# one that answers "can this configuration keep up with the real-time
# budget?", and that result -- that C1, C2 and C4 cannot sustain 50 Hz
# while C3 and C5 can -- is a finding in its own right.  This pass answers
# the separate question "at a rate all five can hold, which configuration
# costs less energy per frame?".
#
# The same workload, the same recorded stream, the same five
# configurations and the same measurement protocol (10 min warm-up + 10 min
# log per repeat, 5 repeats, power from the on-module INA3221) are used,
# so the result is directly comparable with the 50 Hz run.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results/run1_30hz
LOGS=logs/run1_30hz
mkdir -p "$RESULTS" "$LOGS"

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] quiescence check before the matched-rate pass"
python3 quiescence.py --out "$RESULTS/quiescence_before.json" \
    > "$LOGS/quiescence_before.log" 2>&1
echo "[$(stamp)] quiescence exit=$?"

echo "[$(stamp)] RUN 1 AT 30 Hz: 5 configurations x 5 repeats x 20 min"
python3 -u run1_power.py \
    --configs C1,C2,C3,C4,C5 \
    --repeats 5 --warmup-s 600 --log-s 600 --rate 30 \
    --resume \
    --outdir "$RESULTS" --logdir "$LOGS" \
    > "$LOGS/run1_30hz.log" 2>&1
echo "[$(stamp)] RUN 1 (30 Hz) exit=$?"

echo "[$(stamp)] DONE"
