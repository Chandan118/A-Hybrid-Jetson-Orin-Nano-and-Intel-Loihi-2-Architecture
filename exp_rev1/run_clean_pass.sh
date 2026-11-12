#!/usr/bin/env bash
# Re-run every experiment that was measured while a stray CUDA process was
# still alive on the machine.
#
# Why this script exists
# ----------------------
# A leftover debug process (a multiprocessing worker holding a CUDA context)
# was running from 13:23 until it was killed at 21:55 on 2026-11-03.  It
# covered the whole of Run 1b (stage decomposition), Run 2 (memory) and Run 3
# (endurance).  The evidence is unambiguous: the "system idle" stage of the
# stage decomposition recorded GR3D_FREQ = 29.5 %, i.e. the GPU was busy
# during a measurement that is supposed to have nothing running on it.
#
# All three experiments are therefore re-run here, in isolation, with the
# quiescence guard (quiescence.py) refusing to start a measurement unless the
# machine is genuinely idle.
#
# Run 1 is NOT repeated: the re-run that started at 21:58 was after the stray
# process had been killed, and its own quiescence record shows the GPU at 0 %
# before every window.
#
# Order matters: everything runs sequentially, because two measurements
# running at once would contaminate each other exactly as the stray process
# did.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results
LOGS=logs
mkdir -p "$RESULTS/run1" "$RESULTS/run2" "$RESULTS/run3" "$LOGS"

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

# ---------------------------------------------------------------- wait for Run 1
echo "[$(stamp)] waiting for the Run 1 re-run to finish before starting the clean pass"
while pgrep -f "run1_power.py" > /dev/null; do sleep 60; done
while pgrep -f "run_remaining.sh" > /dev/null; do sleep 30; done
echo "[$(stamp)] Run 1 finished; the machine is now free"

# ---------------------------------------------------------------- quiescence
python3 quiescence.py --out "$RESULTS/quiescence_before_clean_pass.json" \
    > "$LOGS/quiescence_before_clean_pass.log" 2>&1
echo "[$(stamp)] quiescence check exit=$?"

# ---------------------------------------------------------------- Run 1b
echo "[$(stamp)] RUN 1b: stage decomposition (clean, 3 repeats x 300 s per stage)"
python3 -u run1b_stages.py \
    --duration 300 --repeats 3 --rate 50 \
    --outdir "$RESULTS/run1" --logdir "$LOGS/run1" \
    > "$LOGS/run1b_stages.log" 2>&1
echo "[$(stamp)] RUN 1b exit=$?"

# ---------------------------------------------------------------- Run 2
echo "[$(stamp)] RUN 2: memory and crashes (clean, 67 recorded replays per configuration)"
python3 -u run2_memory.py \
    --runs 67 --rate 50 \
    --map-mem-limit-mb 3000 --compress-threshold-mb 900 \
    --outdir "$RESULTS/run2" \
    > "$LOGS/run2_full.log" 2>&1
echo "[$(stamp)] RUN 2 exit=$?"

# ---------------------------------------------------------------- Run 3
echo "[$(stamp)] RUN 3: endurance (clean, 900 s x 3 repeats per configuration)"
python3 -u run3_endurance.py \
    --duration 900 --repeats 3 --rate 50 \
    --outdir "$RESULTS/run3" --logdir "$LOGS/run3" \
    > "$LOGS/run3_full.log" 2>&1
echo "[$(stamp)] RUN 3 exit=$?"

# ---------------------------------------------------------------- report
echo "[$(stamp)] regenerating the report and the figures"
python3 analyse_run1.py > "$LOGS/run1_analysis.log" 2>&1
python3 make_review_response.py > "$LOGS/review_response_build.log" 2>&1
python3 make_figures.py >> "$LOGS/review_response_build.log" 2>&1

echo "[$(stamp)] ALL DONE"
