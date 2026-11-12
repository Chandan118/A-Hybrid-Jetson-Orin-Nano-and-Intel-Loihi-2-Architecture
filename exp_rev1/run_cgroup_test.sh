#!/usr/bin/env bash
# The cgroup memory test: J1 vs J2 under a real 6 GB kernel-enforced cap.
#
# WHAT IS DIFFERENT FROM THE EARLIER RUN 2
# -----------------------------------------
# The earlier run imposed the limit with `resource.setrlimit(RLIMIT_AS,
# 3000 MB)` inside the map process.  That is a per-process VIRTUAL
# ADDRESS SPACE cap, not a vehicle memory budget:
#
#   * it fails on address-space RESERVATIONS, not on real residency, so
#     it can kill a process that is not actually using the memory;
#   * it applies to one process, so the group of processes together can
#     exceed the budget;
#   * the harness imposed it, so the failures were the harness's
#     failures, not the system's.
#
# This test uses the KERNEL's cgroup v2 memory controller instead:
#
#   memory.max        = 6G      the vehicle's memory budget
#   memory.swap.max   = 0        no swap: a footprint that does not fit
#                                is an OOM KILL, not a slow slide into
#                                swap, so a kill means what it says
#
# `memory.max` caps the GROUP's real memory (page cache included) and is
# enforced by the kernel.  The failures it produces are therefore real
# kernel OOM kills, and they are counted from the kernel's own
# `oom_kill` counter in /proc/vmstat, which needs no root and counts
# every kill rather than only the ones still in the ring buffer.
#
# WHY root IS NEEDED
# ------------------
# /sys/fs/cgroup is root-owned on this board, so the cgroup has to be
# CREATED by the board owner (see ROOT_REQUEST_NOTE.md).  This script
# only needs write access to the cgroup's own `cgroup.procs`, which the
# board owner can delegate with a chown.
#
# WHAT IS LOGGED IN THE SAME RUN
# -----------------------------
# Latency and FPS are recorded by the SLAM child itself and land in the
# same time series as the memory trace, so a memory figure can never be
# quoted without the throughput it was achieved at.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

CG=${CG:-/sys/fs/cgroup/run2_cgroup_test}
RESULTS=results/run2_cgroup
LOGS=logs/run2_cgroup
RUNS=${RUNS:-30}
CONFIGS=${CONFIGS:-J1,J2}

stamp () { date '+%Y-%m-%d %H:%M:%S'; }
mkdir -p "$RESULTS" "$LOGS"

# ------------------------------------------------- check the cgroup
echo "[$(stamp)] checking the measurement cgroup at $CG"
if [ ! -d "$CG" ]; then
    echo "  the cgroup does not exist."
    echo "  The board owner has to create it (see ROOT_REQUEST_NOTE.md):"
    echo "      mkdir $CG"
    echo "      echo 6G > $CG/memory.max"
    echo "      echo 0  > $CG/memory.swap.max"
    echo "      chown -R jetson:jetson $CG"
    exit 1
fi
echo "  memory.max      = $(cat $CG/memory.max 2>&1)"
echo "  memory.swap.max = $(cat $CG/memory.swap.max 2>&1)"
if [ ! -w "$CG/cgroup.procs" ]; then
    echo "  $CG/cgroup.procs is not writable by this account."
    echo "  The board owner has to delegate it (chown jetson:jetson $CG)."
    exit 1
fi

# The oom counter BEFORE, so the kills caused by this run can be stated
# exactly rather than inferred.
OOM_BEFORE=$(awk '/^oom_kill /{print $2}' /proc/vmstat)
echo "[$(stamp)] kernel oom_kill counter before: $OOM_BEFORE"

# ------------------------------------------------- quiescence
echo "[$(stamp)] quiescence check before the cgroup test"
python3 quiescence.py --out "$RESULTS/quiescence_before.json" \
    > "$LOGS/quiescence_before.log" 2>&1
echo "[$(stamp)] quiescence exit=$?"

# ------------------------------------------------- the test
echo "[$(stamp)] CGROUP TEST: $CONFIGS x $RUNS replays, memory.max=6G, swap=0"
python3 -u run2_memory.py \
    --configs "$CONFIGS" \
    --runs "$RUNS" \
    --cgroup "$CG" \
    --outdir "$RESULTS" \
    > "$LOGS/run2_cgroup.log" 2>&1
RC=$?
echo "[$(stamp)] the test exited with $RC"

# ------------------------------------------------- the oom evidence
OOM_AFTER=$(awk '/^oom_kill /{print $2}' /proc/vmstat)
echo "[$(stamp)] kernel oom_kill counter after: $OOM_AFTER"
python3 - "$RESULTS" "$OOM_BEFORE" "$OOM_AFTER" "$CG" <<'PY'
import json, os, sys
outdir, before, after, cg = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
def rd(fn):
    try:
        return open(os.path.join(cg, fn)).read().strip()
    except Exception as e:
        return f"unreadable ({e})"
rec = {
    "cgroup": cg,
    "memory_max": rd("memory.max"),
    "memory_swap_max": rd("memory.swap.max"),
    "memory_peak": rd("memory.peak"),
    "memory_events": rd("memory.events"),
    "oom_kill_counter_before": before,
    "oom_kill_counter_after": after,
    "oom_kills_during_this_run": after - before,
    "note": ("The kernel's own oom_kill counter is the authoritative count "
             "of OOM kills: it counts every kill since boot, and it does "
             "not need root (dmesg does, and is not readable on this "
             "board).  memory.swap.max = 0 means a footprint that does not "
             "fit is an OOM kill rather than a slide into swap, so every "
             "kill counted here is a real failure to fit the budget."),
}
dst = os.path.join(outdir, "cgroup_oom_evidence.json")
with open(dst, "w") as f:
    json.dump(rec, f, indent=2)
print(json.dumps(rec, indent=2))
print("wrote", dst)
PY

echo "[$(stamp)] DONE"
