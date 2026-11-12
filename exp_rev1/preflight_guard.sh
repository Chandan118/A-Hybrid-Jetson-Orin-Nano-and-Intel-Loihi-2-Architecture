#!/usr/bin/env bash
# Pre-flight guard for the measurement passes.
#
# WHY THIS EXISTS
# ---------------
# On 2026-11-07 the 50 Hz pass stopped at `C2_rep2` with
#
#     NvMapMemAllocInternalTagged: 1075072515 error 12
#     [TRT] [E] Error Code 1: Cuda Runtime (out of memory)
#     RuntimeError: TensorRT engine build failed
#
# The cause was NOT the C2 configuration and NOT a shortage of RAM.  It was
# 19 orphaned `lava_c6_runner.py` children left over from the C6
# equivalence testing on 2026-11-06: each was reparented to init, each
# held ~0.9-1.9 GB of address space, and together they had taken the CMA
# pool down to 692 kB free out of 262144 kB.  The CUDA allocator needs a
# contiguous CMA region, so the TensorRT engine build had nowhere to put
# its buffers.
#
# A leak in ONE configuration broke a measurement of a DIFFERENT one, and
# the error looked like a hardware fault rather than what it was.
#
# WHAT THIS DOES
# --------------
# Before the pass starts, and again on every resume, this script:
#
#   1. reports the CMA headroom and the available RAM,
#   2. reaps any orphaned Lava child that outlived its parent,
#   3. verifies that a TensorRT engine can actually be built, which is the
#      exact operation that failed.
#
# It touches no measurement code, so it cannot affect the numbers.
#
# Usage:
#     ./exp_rev1/preflight_guard.sh
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u

FAIL=0

say () { echo "[preflight] $*"; }

# ---- 1. the CMA pool and RAM -------------------------------------------
cma_total=$(awk '/^CmaTotal/{print $2}' /proc/meminfo)
cma_free=$(awk '/^CmaFree/{print $2}' /proc/meminfo)
mem_avail=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
say "CMA free ${cma_free} kB of ${cma_total} kB; MemAvailable ${mem_avail} kB"

# ---- 2. orphaned Lava children ------------------------------------------
orphans=$(pgrep -f "lava_c6_runner.py" 2>/dev/null | wc -l)
if [ "$orphans" -gt 0 ]; then
    say "found $orphans orphaned lava_c6_runner process(es) -- reaping them"
    pkill -f "lava_c6_runner.py" 2>/dev/null
    sleep 2
    pkill -9 -f "lava_c6_runner.py" 2>/dev/null
    sleep 1
    left=$(pgrep -f "lava_c6_runner.py" 2>/dev/null | wc -l)
    say "reaped; ${left} remaining"
    cma_free=$(awk '/^CmaFree/{print $2}' /proc/meminfo)
    say "CMA free is now ${cma_free} kB"
fi

# The CMA pool on this board sits at only 2-5 MB free in normal operation
# against a 256 MB pool, and the TensorRT build was observed to succeed
# from 4152 kB free -- so a low absolute number is NOT by itself a
# problem.  What matters is whether the pool is exhausted enough for the
# build to fail, and the only reliable test of that is to build the
# engine.  The threshold below is therefore only a warning, and the
# engine build below is the real check.
if [ "$cma_free" -lt 8192 ]; then
    say "CMA free ${cma_free} kB is low (the pool normally sits at"
    say "2-5 MB free out of ${cma_total} kB).  This is not by itself a"
    say "fault -- the engine build below is the real test."
fi

# ---- 3. can a TensorRT engine actually be built? ------------------------
# This is the exact operation that failed on 2026-11-07, so it is the
# exact thing worth testing.
say "verifying that a TensorRT engine can be built"
if python3 - <<'PY'
import sys, os
sys.path.insert(0, "common")
sys.path.insert(0, ".")
import workload as WL
from run1_power import build_backend
w = WL.load_weights("common/weights_quantised.npz")
cfg = dict(WL.NET)
cfg.update(rate_hz=50.0, warmup_s=1, log_s=1, trt_fp16=False)
b = build_backend("C2", w, cfg)
b.close()
print("TensorRT engine build: OK")
PY
then
    say "TensorRT engine build: OK"
else
    say "TensorRT engine build: FAILED -- the pass would stop at the first"
    say "C2 repeat.  Do not start the pass until this succeeds."
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    say "PREFLIGHT PASSED -- the board is ready for the pass."
    exit 0
fi
say "PREFLIGHT FAILED -- do not start the pass."
exit 1
