#!/bin/bash
# Master script to run all experiments

echo "=============================================="
echo "JETSON ORIN NANO EXPERIMENT SUITE"
echo "=============================================="

cd "$(dirname "$0")"

echo ""
echo "Running all experiments..."
echo ""

if [ -f exp_rev1/run_all.sh ]; then
  bash exp_rev1/run_all.sh
else
  python3 src/run_all_experiments.py
fi

echo ""
echo "=============================================="
echo "EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: results/"
echo "Appeal letter: appeal_letter/appeal_letter.txt"
