#!/usr/bin/env bash
# Re-run Run 1 at 50 Hz with the FRESH-PROCESS protocol.
#
# Why this script exists
# ----------------------
# The 50 Hz figures that are currently published in the README, the review
# response, the appeal letter and the manuscript tables were measured by the
# PRE-FIX harness: every configuration and every repeat ran inside ONE
# process, in a fixed config-major order, with no quiescence check and with
# no per-repeat teardown.
#
# The contamination is visible in the recorded power CSVs (results/run1 and
# logs/run1).  The RAM trace at the end of each repeat:
#
#     C1  3885 -> 4202 MB     (climbing)
#     C2  3984 -> 4155 MB     (climbing)
#     C3  5289 -> 5309 MB     (starts high: inherits C1's and C2's state)
#     C4  5322 -> 5311 MB     (starts high)
#     C5  5303 -> 5060 MB
#
# C3, C4 and C5 all start near 5300 MB because the configurations that ran
# before them in the same process had already grown the heap, so the
# absolute RAM figures for those three are inflated by roughly 1.4 GB and
# are not comparable with C1's and C2's.  The power figures are less badly
# affected -- the leak is RAM, not a power leak -- but a number measured in
# a process whose state has already grown is not the same number as one
# measured in a fresh process, and the power trace cannot rule that out.
#
# The 30 Hz pass (results/run1_30hz) was re-measured with the corrected
# protocol -- one fresh process per (config, repeat), a shuffled schedule
# so thermal drift cannot favour whichever configuration runs last, and a
# quiescence check before every window -- and it passed its gate.  This
# script applies the SAME protocol to the 50 Hz pass.
#
# This pass does NOT replace the 30 Hz pass.  The two answer different
# questions:
#
#   50 Hz: "can this configuration keep up with the real-time budget?"
#          -> C1, C2 and C4 cannot; C3 and C5 can.
#   30 Hz: "at a rate all five can hold, which configuration costs less
#           energy per frame?"  -> the J/frame figures are directly
#           comparable across all five.
#
# Both are needed, and the 30 Hz numbers stay published either way.
#
# The results go to a NEW directory (results/run1_50hz_fresh) so that the
# old contaminated results stay on the record until the new ones have been
# through the gate.  Nothing is overwritten.
#
# The pass takes about 8.5 hours.  --resume means an interruption costs
# one repeat, not the whole pass.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results/run1_50hz_fresh
LOGS=logs/run1_50hz_fresh
mkdir -p "$RESULTS" "$LOGS"

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] quiescence check before the 50 Hz fresh-process pass"
python3 quiescence.py --out "$RESULTS/quiescence_before.json" \
    > "$LOGS/quiescence_before.log" 2>&1
echo "[$(stamp)] quiescence exit=$?"

echo "[$(stamp)] RUN 1 AT 50 Hz: 5 configurations x 5 repeats x 20 min"
echo "[$(stamp)] one fresh process per (config, repeat), shuffled schedule"
python3 -u run1_power.py \
    --configs C1,C2,C3,C4,C5 \
    --repeats 5 --warmup-s 600 --log-s 600 --rate 50 \
    --resume \
    --outdir "$RESULTS" --logdir "$LOGS" \
    > "$LOGS/run1_50hz_fresh.log" 2>&1
echo "[$(stamp)] RUN 1 (50 Hz, fresh process) exit=$?"

echo "[$(stamp)] DONE"
