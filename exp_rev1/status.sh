#!/usr/bin/env bash
# One-screen status of the whole measurement pipeline.
#
#   ./status.sh
#
# Shows which pass is running, how far each measurement has got, and
# whether the results on disk are current or stale.
cd "$(dirname "$0")"

echo "==============================================================="
echo " MEASUREMENT PIPELINE STATUS    $(date '+%Y-%m-%d %H:%M:%S')"
echo "==============================================================="

echo
echo "--- running passes ---"
if pgrep -af "run1_power.py" | grep -v grep > /dev/null; then
    echo "  Run 1  (power)          : RUNNING"
elif pgrep -af "run1b_stages.py" | grep -v grep > /dev/null; then
    echo "  Run 1b (stages)         : RUNNING"
elif pgrep -af "run2_memory.py" | grep -v grep > /dev/null; then
    echo "  Run 2  (memory/crashes) : RUNNING"
elif pgrep -af "run3_endurance.py" | grep -v grep > /dev/null; then
    echo "  Run 3  (endurance)      : RUNNING"
elif pgrep -af "run1c_pose_check.py" | grep -v grep > /dev/null; then
    echo "  Run 1c (pose RMSE)      : RUNNING"
else
    echo "  (no measurement pass running)"
fi
for s in run_clean_pass run_final_pass; do
    if pgrep -af "${s}.sh" | grep -v grep > /dev/null; then
        echo "  ${s}.sh          : QUEUED/RUNNING"
    fi
done

echo
echo "--- Run 1 (50 Hz) progress (5 configs x 5 repeats) ---"
python3 - <<'PY'
import json, os, glob
R = "results"
for tag, sub in (("50 Hz", "run1"), ("30 Hz matched", "run1_30hz")):
    d = os.path.join(R, sub)
    if not os.path.isdir(d):
        continue
    n_done = 0
    for f in sorted(glob.glob(os.path.join(d, "*_run.json"))):
        try:
            n_done += len(json.load(open(f)).get("runs", []))
        except Exception:
            pass
    csvs = len(glob.glob(os.path.join(R, "..", "logs", sub, "*.csv")))
    print(f"  {tag:<14}: {n_done}/25 repeats recorded, {csvs} power CSVs on disk")
PY

echo
echo "--- Run 2 progress (67 replays per configuration) ---"
for j in J1 J2 J3 J4; do
    f="results/run2/${j}_timeseries.csv"
    if [ -f "$f" ]; then
        n=$(python3 -c "
import csv,sys
with open('$f') as fh: rows=list(csv.DictReader(fh))
print(len(set(r['run'] for r in rows)))
" 2>/dev/null)
        printf "  %s : %s/67\n" "$j" "$n"
    fi
done

echo
echo "--- result freshness (vs the quiescence record that opened the clean pass) ---"
python3 - <<'PY'
import json, os
from datetime import datetime
R = "results"
marker = os.path.join(R, "quiescence_before_clean_pass.json")
if not os.path.exists(marker):
    print("  (no clean-pass marker yet)")
else:
    ts = datetime.strptime(json.load(open(marker))["timestamp"],
                            "%Y-%m-%d %H:%M:%S").timestamp()
    stale = []
    for rel in ("run1/all_runs.json", "run1/summary.json",
                "run1/stage_decomposition.json", "run1/pose_equivalence.json",
                "run2/summary.json", "run3/summary.json"):
        p = os.path.join(R, rel)
        if os.path.exists(p):
            if os.path.getmtime(p) < ts:
                stale.append(rel)
    if stale:
        print("  STALE (predate the clean pass, must be regenerated):")
        for s in stale:
            print(f"    - {s}")
    else:
        print("  all current")
PY

echo
echo "--- deliverable ---"
for f in results/REVIEW_RESPONSE.md results/manuscript/summary.txt \
         results/manuscript/tables.tex results/provenance_audit.json; do
    if [ -f "$f" ]; then
        printf "  %-42s %s\n" "$f" "$(stat -c '%y' "$f" | cut -c1-16)"
    else
        printf "  %-42s MISSING\n" "$f"
    fi
done

echo
echo "--- last pipeline log lines ---"
for l in logs/run_clean_pass.log logs/run_final_pass.log; do
    if [ -f "$l" ]; then
        echo "  [$l]"
        tail -2 "$l" | sed 's/^/    /'
    fi
done
echo "==============================================================="
