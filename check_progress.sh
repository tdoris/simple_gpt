#!/bin/bash

# Simple script to check training progress
echo "SimpleGPT Training Progress Check"
echo "================================="

# Path to latest training metrics
METRICS_FILE="/home/jim/repos/simple_gpt/output/slimpajama_all_20250324_084832/metrics.json"
echo "Checking metrics file: $METRICS_FILE"
echo

# Latest step and loss
echo "Latest training information:"
LATEST_STEP=$(grep '"step":' "$METRICS_FILE" | tail -n 1 | cut -d':' -f2 | cut -d',' -f1)
LATEST_LOSS=$(grep '"loss":' "$METRICS_FILE" | tail -n 1 | cut -d':' -f2 | cut -d',' -f1)
echo "Current step: $LATEST_STEP"
echo "Current loss: $LATEST_LOSS"
echo

# Latest evaluation metrics
echo "Latest evaluation metrics:"
grep -B3 "eval_perplexity" "$METRICS_FILE" | tail -n 4

# Training time and remaining time
echo
echo "Time information:"
ELAPSED=$(grep '"elapsed":' "$METRICS_FILE" | tail -n 1 | cut -d':' -f2 | cut -d',' -f1)
ELAPSED_HOURS=$(echo "$ELAPSED / 3600" | bc -l | xargs printf "%.2f")
REMAINING=$(grep "estimated_remaining" "$METRICS_FILE" | tail -n 1)
echo "Elapsed training time: $ELAPSED_HOURS hours"
echo $REMAINING

# Training speed
echo
echo "Training speed:"
SPEED=$(grep '"samples_per_second":' "$METRICS_FILE" | tail -n 1 | cut -d':' -f2 | cut -d',' -f1)
echo "Current speed: $SPEED samples/second"

echo
echo "================================="