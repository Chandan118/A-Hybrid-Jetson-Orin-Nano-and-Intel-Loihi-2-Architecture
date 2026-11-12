#!/usr/bin/env bash
# The locked-GPU-clock pass (C1, C2, C4).
#
# WHY A SEPARATE PASS, IN A SEPARATE TABLE
# ----------------------------------------
# The main 30 Hz pass runs with the GPU on its normal 'nvhost_podgov'
# governor, which is the state a deployed system is in and the state the
# main table should describe.  This pass pins the GPU to a single
# frequency for the whole run, so that the three configurations it
# covers are compared at an identical clock rather than at whatever the
# governor happened to choose for each of them.
#
# The results are reported in their OWN table and are NOT merged with the
# main table.  The two tables answer different questions:
#
#   main table      : what does each configuration draw in normal
#                     operation?            (governor active)
#   locked-clock    : what does each configuration draw when the clock
#     table          : is held constant?     (clock pinned)
#
# Merging them would answer neither question, because a pinned clock is
# not what the system does in practice, and a governor-driven clock is
# not a controlled comparison.
#
# WHY C1, C2 AND C4
# -----------------
# Those are the three configurations that hold the GPU: C1 (PyTorch on
# CUDA), C2 (TensorRT on CUDA) and C4 (GPU front end + CPU fusion core).
# C3 and C5 are CPU-only, so pinning the GPU clock cannot change them --
# they are already covered by the main table, and re-running them here
# would spend 3.5 h of clock time to reproduce numbers that cannot move.
#
# WHY THIS NEEDS ROOT
# -------------------
# The devfreq node is root-owned:
#
#     $ ls -l /sys/class/devfreq/17000000.gpu/min_freq
#     -r--r--r-- 1 root root 4096 ... /sys/class/devfreq/17000000.gpu/min_freq
#
# so the pin cannot be applied from the experiment account.  The board
# owner applies it (see exp_rev1/ROOT_REQUEST_NOTE.md), or applies it
# here with a passwordless sudo if the board has been configured for it.
#
# THE PIN
# -------
#     echo userspace > /sys/class/devfreq/17000000.gpu/governor
#     echo 306000000     > /sys/class/devfreq/17000000.gpu/min_freq
#     echo 306000000     > /sys/class/devfreq/17000000.gpu/max_freq
#
# 306 MHz is the board's own min_freq, i.e. the floor the main pass
# observed anyway.  Pinning it makes that floor a GUARANTEE rather than
# an observation, which is what makes the three configurations
# comparable to each other.
#
# THE RELEASE
# -----------
# The governor is restored afterwards, in a trap, so the board is left in
# its normal state even if the pass is interrupted:
#
#     echo nvhost_podgov > /sys/class/devfreq/17000000.gpu/governor
#
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results/run1_lockedclock
LOGS=logs/run1_lockedclock
mkdir -p "$RESULTS" "$LOGS"

GPU_DEV=/sys/class/devfreq/17000000.gpu
LOCK_HZ=${LOCK_HZ:-306000000}
CONFIGS=${CONFIGS:-C1,C2,C4}
REPEATS=${REPEATS:-5}

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

restore_clock () {
    echo "[$(stamp)] restoring the GPU governor and the original clock range"
    if [ -w "$GPU_DEV/governor" ]; then
        echo nvhost_podgov > "$GPU_DEV/governor" 2>/dev/null || true
        # the original min/max are read from the driver defaults, which
        # are the values the board ships with
        echo 306000000 > "$GPU_DEV/min_freq" 2>/dev/null || true
        echo 1020000000 > "$GPU_DEV/max_freq" 2>/dev/null || true
    else
        echo "  the devfreq node is not writable from this account --"
        echo "  the board owner must restore it (see ROOT_REQUEST_NOTE.md)"
    fi
}
trap restore_clock EXIT INT TERM

# ------------------------------------------------- pin the clock
echo "[$(stamp)] pinning the GPU clock to ${LOCK_HZ} Hz"
if ! echo userspace > "$GPU_DEV/governor" 2>/dev/null; then
    echo "  FAILED to set the governor -- this pass needs root."
    echo "  Ask the board owner to run the commands in ROOT_REQUEST_NOTE.md."
    exit 1
fi
echo "${LOCK_HZ}" > "$GPU_DEV/min_freq" || exit 1
echo "${LOCK_HZ}" > "$GPU_DEV/max_freq" || exit 1
echo "  governor=$(cat $GPU_DEV/governor) min=$(cat $GPU_DEV/min_freq) max=$(cat $GPU_DEV/max_freq)"
if [ "$(cat $GPU_DEV/min_freq)" != "${LOCK_HZ}" ]; then
    echo "  FAILED: the clock did not take the pin"
    exit 1
fi

# ------------------------------------------------- quiescence before
echo "[$(stamp)] quiescence check before the locked-clock pass"
python3 quiescence.py --out "$RESULTS/quiescence_before.json" \
    > "$LOGS/quiescence_before.log" 2>&1
echo "[$(stamp)] quiescence exit=$?"

# ------------------------------------------------- the pass
echo "[$(stamp)] LOCKED-CLOCK PASS: $CONFIGS x $REPEATS repeats"
python3 -u run1_power.py \
    --configs "$CONFIGS" \
    --repeats "$REPEATS" --warmup-s 600 --log-s 600 --rate 30 \
    --resume \
    --outdir "$RESULTS" --logdir "$LOGS" \
    > "$LOGS/run1_lockedclock.log" 2>&1
RC=$?
echo "[$(stamp)] the pass exited with $RC"

# ------------------------------------------------- record the clock state
python3 - "$RESULTS" <<'PY'
import json, os, sys
gpu = "/sys/class/devfreq/17000000.gpu"
def rd(n):
    try:
        return open(os.path.join(gpu, n)).read().strip()
    except Exception:
        return None
out = {
    "governor": rd("governor"),
    "min_freq_hz": rd("min_freq"),
    "max_freq_hz": rd("max_freq"),
    "cur_freq_hz": rd("cur_freq"),
    "available_frequencies": (rd("available_frequencies") or "").split(),
    "note": ("The GPU clock was PINNED for this whole pass, so every "
             "configuration in this table was measured at the same clock. "
             "This table is reported separately from the main pass and "
             "must not be merged with it: the main pass measures normal "
             "governor-driven operation, this one measures a controlled "
             "constant clock."),
}
dst = os.path.join(sys.argv[1], "clock_lock_record.json")
with open(dst, "w") as f:
    json.dump(out, f, indent=2)
print("wrote", dst, "->", out["governor"], out["min_freq_hz"], out["max_freq_hz"])
PY

echo "[$(stamp)] DONE"
