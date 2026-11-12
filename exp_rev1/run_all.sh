#!/usr/bin/env bash
# Sequential execution of the three reviewer-requested experiments.
#
# The runs are executed SEQUENTIALLY and EXCLUSIVELY: no other workload runs
# while a power measurement is in progress, otherwise the power numbers
# would be contaminated.  Every step writes its own log and its own result
# files, so a partial completion still leaves usable data.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results
LOGS=logs
mkdir -p "$RESULTS" "$LOGS"

# ---- tunables (overridable from the command line) ---------------------
RUN1_WARMUP=${RUN1_WARMUP:-600}     # 10 min warm-up  (reviewer protocol)
RUN1_LOG=${RUN1_LOG:-600}           # 10 min log      (reviewer protocol)
RUN1_REPEATS=${RUN1_REPEATS:-5}     # 5 repeats       (reviewer protocol)
RUN1_CONFIGS=${RUN1_CONFIGS:-C1,C2,C3,C4,C5}

STAGE_DURATION=${STAGE_DURATION:-300}
STAGE_REPEATS=${STAGE_REPEATS:-3}

RUN2_RUNS=${RUN2_RUNS:-67}
RUN2_FRAMES=${RUN2_FRAMES:-2746}
# The address-space cap applied to the map process.  It is the physical
# stand-in for the vehicle's memory ceiling: when the UNCOMPRESSED map
# outgrows it the allocation fails and the map process dies with a
# MemoryError, which is exactly the OOM crash under study.  3000 MB leaves
# the other three components of the stack their share of the 8 GB board.
RUN2_MAP_MEM_LIMIT_MB=${RUN2_MAP_MEM_LIMIT_MB:-3000}
RUN2_COMPRESS_THRESHOLD_MB=${RUN2_COMPRESS_THRESHOLD_MB:-900}

RUN3_DURATION=${RUN3_DURATION:-900}
RUN3_REPEATS=${RUN3_REPEATS:-3}
RUN3_CONFIGS=${RUN3_CONFIGS:-C1,C4}

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

# ---------------------------------------------------------------- Run 1
echo "[$(stamp)] RUN 1: power and energy"
python3 -u run1_power.py \
    --configs "$RUN1_CONFIGS" \
    --repeats "$RUN1_REPEATS" \
    --warmup-s "$RUN1_WARMUP" \
    --log-s "$RUN1_LOG" \
    --rate 50 \
    --outdir "$RESULTS/run1" \
    --logdir "$LOGS/run1" \
    > "$LOGS/run1_full.log" 2>&1
echo "[$(stamp)] RUN 1 exit=$?"

python3 -u analyse_run1.py --rundir "$RESULTS/run1" > "$LOGS/run1_analysis.log" 2>&1
echo "[$(stamp)] RUN 1 analysis exit=$?"

# ------------------------------------------------------- Run 1c: output equivalence
echo "[$(stamp)] RUN 1c: pose RMSE output-equivalence check"
python3 -u run1c_pose_check.py \
    --configs "$RUN1_CONFIGS" \
    --out "$RESULTS/run1/pose_equivalence.json" \
    > "$LOGS/run1c_pose_check.log" 2>&1
echo "[$(stamp)] RUN 1c exit=$?"

# ------------------------------------------------------- Run 1d: boundary table
echo "[$(stamp)] RUN 1d: measurement-boundary table"
python3 -u run1d_boundary_table.py > "$LOGS/run1d_boundary.log" 2>&1
echo "[$(stamp)] RUN 1d exit=$?"

# ------------------------------------------------------- Run 1b: stages
echo "[$(stamp)] RUN 1b: stage decomposition"
python3 -u run1b_stages.py \
    --duration "$STAGE_DURATION" \
    --repeats "$STAGE_REPEATS" \
    --rate 50 \
    --outdir "$RESULTS/run1" \
    --logdir "$LOGS/run1" \
    > "$LOGS/run1b_stages.log" 2>&1
echo "[$(stamp)] RUN 1b exit=$?"

# ---------------------------------------------------------------- Run 2
echo "[$(stamp)] RUN 2: memory and crashes"
python3 -u run2_memory.py \
    --runs "$RUN2_RUNS" \
    --frames "$RUN2_FRAMES" \
    --rate 50 \
    --map-mem-limit-mb "$RUN2_MAP_MEM_LIMIT_MB" \
    --compress-threshold-mb "$RUN2_COMPRESS_THRESHOLD_MB" \
    --outdir "$RESULTS/run2" \
    > "$LOGS/run2_full.log" 2>&1
echo "[$(stamp)] RUN 2 exit=$?"

# ---------------------------------------------------------------- Run 3
echo "[$(stamp)] RUN 3: endurance"
python3 -u run3_endurance.py \
    --duration "$RUN3_DURATION" \
    --repeats "$RUN3_REPEATS" \
    --configs "$RUN3_CONFIGS" \
    --rate 50 \
    --outdir "$RESULTS/run3" \
    --logdir "$LOGS/run3" \
    > "$LOGS/run3_full.log" 2>&1
echo "[$(stamp)] RUN 3 exit=$?"

# ---------------------------------------------------------------- Lookups
echo "[$(stamp)] LOOKUPS (no experiment needed)"
python3 -u lookups/lava_footprint.py > "$LOGS/lookup_lava_footprint.log" 2>&1
g++ -O2 -o /tmp/c_equiv lookups/c_equivalent_size.cpp \
    && /tmp/c_equiv > "$RESULTS/c_equivalent_size.txt" 2>&1
python3 -u lookups/interconnect_and_provenance.py \
    > "$LOGS/lookup_interconnect.log" 2>&1
echo "[$(stamp)] LOOKUPS exit=$?"

echo "[$(stamp)] ALL DONE"
