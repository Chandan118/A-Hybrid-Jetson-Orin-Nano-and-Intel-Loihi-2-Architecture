#!/bin/bash
# Master script to run all experiments

echo "=============================================="
echo "JETSON ORIN NANO EXPERIMENT SUITE"
echo "=============================================="

cd /home/jetson/jetson_slam_experiments

echo ""
echo "Running all experiments..."
echo ""

python3 src/run_all_experiments.py

echo ""
echo "=============================================="
echo "EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to: results/"
echo "Appeal letter: appeal_letter/appeal_letter.txt"
