#!/bin/bash

# Script to check training progress

# Default metrics file location
METRICS_FILE="./output/latest_training/metrics.json"

# Allow overriding the metrics file path
if [ $# -gt 0 ]; then
  METRICS_FILE="$1"
fi

# Check if the metrics file exists
if [ ! -f "$METRICS_FILE" ]; then
  echo "Metrics file not found: $METRICS_FILE"
  echo "No training in progress or the training hasn't started logging metrics yet."
  exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
  echo "jq is not installed. Using basic metrics display."
  
  # Show last few lines of metrics file
  echo "=== Latest metrics ==="
  tail -n 20 "$METRICS_FILE"
  
  # Check if training process is running
  TRAINING_DIR=$(dirname "$METRICS_FILE")
  if [ -f "$TRAINING_DIR/training.log" ]; then
    echo ""
    echo "=== Latest log entries ==="
    tail -n 10 "$TRAINING_DIR/training.log"
  fi
  
  exit 0
fi

# Get latest metrics
echo "=== Training Progress Report ==="
echo "Metrics file: $METRICS_FILE"
echo ""

# Get the latest entry
LATEST=$(jq -r 'if type=="array" then .[-1] else . end' "$METRICS_FILE")

# Extract information
STEP=$(echo "$LATEST" | jq -r '.step // .global_step // "N/A"')
LOSS=$(echo "$LATEST" | jq -r '.loss // "N/A"')
EVAL_LOSS=$(echo "$LATEST" | jq -r '.eval_loss // "N/A"')
PERPLEXITY=$(echo "$LATEST" | jq -r '.eval_perplexity // "N/A"')
LEARNING_RATE=$(echo "$LATEST" | jq -r '.learning_rate // "N/A"')
ELAPSED=$(echo "$LATEST" | jq -r '.elapsed // "N/A"')
REMAINING=$(echo "$LATEST" | jq -r '.estimated_remaining // "N/A"')
EPOCH=$(echo "$LATEST" | jq -r '.epoch // "N/A"')

# Format time if it's a number
if [[ "$ELAPSED" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  HOURS=$((ELAPSED / 3600))
  MINUTES=$(((ELAPSED % 3600) / 60))
  SECONDS=$((ELAPSED % 60))
  ELAPSED="${HOURS}h ${MINUTES}m ${SECONDS}s"
fi

# Display information
echo "Current Status:"
echo "  Step: $STEP"
echo "  Epoch: $EPOCH"
echo "  Training Loss: $LOSS"
echo "  Evaluation Loss: $EVAL_LOSS"
echo "  Perplexity: $PERPLEXITY"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Elapsed Time: $ELAPSED"
echo "  Estimated Remaining: $REMAINING"
echo ""

# Check GPU usage
echo "=== GPU Status ==="
nvidia-smi

# Get training directory
TRAINING_DIR=$(dirname "$METRICS_FILE")

# Check if there are any checkpoint directories
if [ -d "$TRAINING_DIR" ]; then
  # Count checkpoints
  CHECKPOINT_COUNT=$(find "$TRAINING_DIR" -type d -name "checkpoint-*" | wc -l)
  
  echo ""
  echo "=== Training Stats ==="
  echo "Checkpoints saved: $CHECKPOINT_COUNT"
  
  # Check how long ago the metrics file was modified
  LAST_MODIFIED=$(stat -c %Y "$METRICS_FILE" 2>/dev/null || stat -f %m "$METRICS_FILE" 2>/dev/null)
  CURRENT_TIME=$(date +%s)
  
  if [ ! -z "$LAST_MODIFIED" ]; then
    TIME_DIFF=$((CURRENT_TIME - LAST_MODIFIED))
    
    if [ $TIME_DIFF -lt 300 ]; then
      echo "Status: ACTIVE (last update: $((TIME_DIFF / 60))m ${TIME_DIFF % 60}s ago)"
    else
      echo "Status: INACTIVE (last update: $((TIME_DIFF / 60))m ago)"
      echo "Training may have completed or stalled."
    fi
  fi
fi

# Check log file
LOG_FILE="$TRAINING_DIR/training.log"
if [ -f "$LOG_FILE" ]; then
  echo ""
  echo "=== Recent Log Entries ==="
  tail -n 10 "$LOG_FILE"
fi

echo ""
echo "To see the full metrics, run:"
echo "python scripts/plot_metrics.py --metrics_file=\"$METRICS_FILE\""