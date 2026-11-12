#!/usr/bin/env bash
# Final pass.  Runs after run_clean_pass.sh has finished.
#
# Two defects were found in the first Run 1 measurement and are fixed here:
#
#  1. The TensorRT backend (C2) did not execute the 25-step LIF recurrence:
#     it ran the three dense layers ONCE per frame instead of 25 times, so it
#     did 1/25 of the arithmetic of every other configuration and its power
#     and pose numbers were meaningless.  run1_power.py now loops the three
#     engines T times per frame, as the reference recurrence and the other
#     backends do.  Only C2 is affected, so only C2 is re-measured; the
#     other four configurations' numbers are unaffected and are kept.
#
#  2. The stage decomposition (Run 1b) aborted at the A1 ablation because
#     the quiescence guard produced a FALSE POSITIVE on update-manager: the
#     guard used psutil's cpu_percent(), whose baseline is cached inside a
#     Process object, so a process that was already being tracked could be
#     reported as 99 % busy while it was demonstrably idle.  The guard now
#     computes the share from the process's own CPU-time counter over a known
#     wall-clock window, which has no such state.  Run 1b is re-run in full.
#
#  3. Run 1c (pose RMSE) is re-run as well, so that the C2 fix is verified
#     against the float64 reference.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results
LOGS=logs
mkdir -p "$RESULTS/run1" "$LOGS"

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

# ------------------------------------------------- wait for the clean pass
echo "[$(stamp)] waiting for run_clean_pass.sh to finish"
while pgrep -f "run_clean_pass.sh" > /dev/null; do sleep 60; done
echo "[$(stamp)] clean pass finished"

python3 quiescence.py --out "$RESULTS/quiescence_before_final_pass.json" \
    > "$LOGS/quiescence_before_final_pass.log" 2>&1
echo "[$(stamp)] quiescence check exit=$?"

# ------------------------------------------------- Run 1b, full, clean
echo "[$(stamp)] RUN 1b: stage decomposition (re-run, 3 repeats x 300 s per stage)"
python3 -u run1b_stages.py \
    --duration 300 --repeats 3 --rate 50 \
    --outdir "$RESULTS/run1" --logdir "$LOGS/run1" \
    > "$LOGS/run1b_stages.log" 2>&1
echo "[$(stamp)] RUN 1b exit=$?"

# ------------------------------------------------- Run 1, C2 only
echo "[$(stamp)] RUN 1: re-measuring C2 only (the other four configurations are unaffected)"
python3 -u run1_power.py \
    --configs C2 --repeats 5 --warmup-s 600 --log-s 600 --rate 50 \
    --outdir "$RESULTS/run1" --logdir "$LOGS/run1" \
    > "$LOGS/run1_c2_rerun.log" 2>&1
echo "[$(stamp)] RUN 1 (C2) exit=$?"

# ------------------------------------------------- merge C2 into the summary
python3 - <<'PY'
import json, os
HERE = os.path.dirname(os.path.abspath("."))
R = os.path.join("results", "run1")
allr = {}
p = os.path.join(R, "all_runs.json")
if os.path.exists(p):
    with open(p) as f:
        allr = json.load(f)
for fn in sorted(os.listdir(R)):
    if fn.endswith("_run.json"):
        with open(os.path.join(R, fn)) as f:
            d = json.load(f)
        if d.get("runs"):
            allr[d["runs"][0]["config"]] = d["runs"]
with open(p, "w") as f:
    json.dump(allr, f, indent=2)
print("merged configurations:", sorted(allr))
PY

# ------------------------------------------------- Run 1c
echo "[$(stamp)] RUN 1c: pose RMSE output-equivalence check (all five configurations)"
python3 -u run1c_pose_check.py \
    --configs C1,C2,C3,C4,C5 \
    --out "$RESULTS/run1/pose_equivalence.json" \
    > "$LOGS/run1c_pose_check.log" 2>&1
echo "[$(stamp)] RUN 1c exit=$?"

# ------------------------------------------------- Run 1d
echo "[$(stamp)] RUN 1d: measurement-boundary table"
python3 -u run1d_boundary_table.py > "$LOGS/run1d_boundary.log" 2>&1
echo "[$(stamp)] RUN 1d exit=$?"

# ------------------------------------------------- report
echo "[$(stamp)] regenerating the report and the figures"
python3 analyse_run1.py > "$LOGS/run1_analysis.log" 2>&1
python3 make_review_response.py > "$LOGS/review_response_build.log" 2>&1
python3 make_figures.py >> "$LOGS/review_response_build.log" 2>&1

# ------------------------------------------------- published numbers
# The README key-results table, the appeal letter and results/metrics/*.json
# currently carry random draws from src/.  Replace them with the measured
# values so that nothing outside exp_rev1/ claims to be a measurement.
python3 sync_published_numbers.py >> "$LOGS/review_response_build.log" 2>&1

# ------------------------------------------------- the paper's own figures
# S1, S2 and S3 were drawn from hard-coded / np.random values.  Redraw them
# from the measured results; the synthetic originals are kept as
# figure/S*_synthetic.* for the audit trail.
python3 make_paper_figures.py >> "$LOGS/review_response_build.log" 2>&1
echo "[$(stamp)] make_paper_figures exit=$?"

# ------------------------------------------------- manuscript tables
# The LaTeX/markdown tables the paper can be written around, generated from
# the measured data so that the prose and the data cannot disagree.
python3 build_manuscript_tables.py >> "$LOGS/review_response_build.log" 2>&1
echo "[$(stamp)] build_manuscript_tables exit=$?"

# ------------------------------------------------- final verification
# Check that every deliverable exists and is newer than the clean pass, and
# that no result artefact still predates the measurement it claims to come
# from.  A silent failure here would leave stale numbers in the paper.
echo "[$(stamp)] FINAL VERIFICATION"
python3 verify_final.py
echo "[$(stamp)] verify_final exit=$?"

echo "[$(stamp)] ALL DONE"
