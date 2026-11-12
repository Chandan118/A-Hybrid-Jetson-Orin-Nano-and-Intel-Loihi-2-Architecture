#!/usr/bin/env bash
# Finalise the 30 Hz matched-rate pass.
#
# Runs AFTER run_matched_rate.sh has finished.  It does four things:
#
#  1. analyses the 30 Hz per-repeat data into results/run1_30hz/summary.json
#  2. regenerates the review response, the figures, the manuscript tables
#     and the appeal letters, so that the new matched-rate section appears
#     in every document
#  3. re-runs the verification suite
#  4. reports what changed
#
# The 50 Hz pass is NOT re-run and NOT overwritten: it answers "can this
# configuration keep up with the real-time budget?", and the 30 Hz pass
# answers the separate question "at a rate all five can hold, which
# configuration costs less energy per frame?".  Both are reported.
set -u
cd "$(dirname "$0")"
set +u
source /opt/ros/humble/setup.bash
set -u
export PYTHONUNBUFFERED=1

RESULTS=results
LOGS=logs

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

# ------------------------------------------------- wait for the pass
echo "[$(stamp)] waiting for run_matched_rate.sh to finish"
while pgrep -f "run_matched_rate.sh" > /dev/null; do
    sleep 60
done
echo "[$(stamp)] the 30 Hz pass has finished"

# ------------------------------------------------- quiescence before the
# ------------------------------------------------- document generation
# The document generators are CPU-heavy.  They are not measurements, so
# they do not need the quiescent machine -- but the quiescence record is
# taken anyway, so that the state of the machine at the moment the
# reports were written is on the record.
python3 quiescence.py --out "$RESULTS/run1_30hz/quiescence_before_report.json" \
    > "$LOGS/quiescence_before_report.log" 2>&1
echo "[$(stamp)] quiescence check exit=$?"

# ------------------------------------------------- analyse the 30 Hz data
echo "[$(stamp)] analysing the 30 Hz matched-rate results"
python3 -u analyse_run1.py --rundir "$RESULTS/run1_30hz" \
    > "$LOGS/run1_30hz_analysis.log" 2>&1
echo "[$(stamp)] analysis exit=$?"

# ------------------------------------------------- regenerate everything
echo "[$(stamp)] regenerating the report, the figures, the tables and the letters"
python3 make_review_response.py > "$LOGS/review_response_build.log" 2>&1
echo "[$(stamp)] make_review_response exit=$?"
python3 make_figures.py >> "$LOGS/review_response_build.log" 2>&1
echo "[$(stamp)] make_figures exit=$?"
python3 sync_published_numbers.py >> "$LOGS/review_response_build.log" 2>&1
echo "[$(stamp)] sync_published_numbers exit=$?"
python3 make_paper_figures.py >> "$LOGS/review_response_build.log" 2>&1
echo "[$(stamp)] make_paper_figures exit=$?"
python3 build_manuscript_tables.py >> "$LOGS/review_response_build.log" 2>&1
echo "[$(stamp)] build_manuscript_tables exit=$?"
python3 make_appeal_letters.py >> "$LOGS/review_response_build.log" 2>&1
echo "[$(stamp)] make_appeal_letters exit=$?"

# ------------------------------------------------- final verification
echo "[$(stamp)] FINAL VERIFICATION"
python3 verify_final.py
echo "[$(stamp)] verify_final exit=$?"

echo "[$(stamp)] ALL DONE"
