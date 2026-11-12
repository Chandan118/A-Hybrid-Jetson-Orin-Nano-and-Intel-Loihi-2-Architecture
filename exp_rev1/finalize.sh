#!/usr/bin/env bash
# Wait for the Run 1 re-run to finish, then regenerate the report and figures.
cd "$(dirname "$0")"
while pgrep -f "run_remaining.sh" > /dev/null; do sleep 60; done
echo "run_remaining.sh finished at $(date '+%Y-%m-%d %H:%M:%S')"
python3 analyse_run1.py > logs/run1_analysis.log 2>&1
python3 make_review_response.py > logs/review_response_build.log 2>&1
python3 make_figures.py >> logs/review_response_build.log 2>&1
echo "report + figures regenerated at $(date '+%Y-%m-%d %H:%M:%S')"
