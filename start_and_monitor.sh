#!/bin/bash

# Script to start training and monitor it for hangs

# Default values
MAX_HANG_TIME=600  # 10 minutes in seconds
CHECK_INTERVAL=60  # Check every minute
MAX_RETRIES=3      # Maximum number of restart attempts

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-hang-time)
            MAX_HANG_TIME=$2
            shift 2
            ;;
        --check-interval)
            CHECK_INTERVAL=$2
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting training with hang monitoring..."
echo "Max hang time: $MAX_HANG_TIME seconds"
echo "Check interval: $CHECK_INTERVAL seconds"
echo "Max retries: $MAX_RETRIES"

# Create output directory with timestamp
OUTPUT_DIR="./output/extended_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
METRICS_FILE="$OUTPUT_DIR/metrics.json"
LOG_FILE="$OUTPUT_DIR/training.log"

# Set environment variables for cache directories
export HF_HOME="./cache/huggingface"
export HF_DATASETS_CACHE="./cache/datasets"

# Create symlink to latest training
rm -f ./output/latest_extended_training 2>/dev/null
ln -sf "$OUTPUT_DIR" ./output/latest_extended_training

# Start the training process in the background
echo "Starting training process..."
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Metrics file: $METRICS_FILE"

# Start the training script
python scripts/long_train.py \
  --model_type=gpt \
  --d_model=768 \
  --num_heads=12 \
  --num_layers=12 \
  --d_ff=3072 \
  --max_seq_length=512 \
  --dropout=0.1 \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=4 \
  --gradient_accumulation_steps=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=10 \
  --max_train_time=12 \
  --warmup_ratio=0.1 \
  --fp16 \
  --logging_steps=50 \
  --eval_steps=500 \
  --save_steps=1000 \
  --dataset_name=wikitext \
  --dataset_config_name=wikitext-103-raw-v1 \
  --max_train_samples=50000 \
  --max_val_samples=5000 \
  --metrics_file="metrics.json" \
  > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo "Training process started with PID $TRAINING_PID"

# Function to check if metrics file has been updated
function check_metrics_file() {
    if [ ! -f "$METRICS_FILE" ]; then
        # File doesn't exist yet, this is normal at the start
        echo "Waiting for metrics file to be created..."
        return 0
    fi
    
    # Get last modification time of metrics file
    local last_mod=$(stat -c %Y "$METRICS_FILE")
    local current_time=$(date +%s)
    local time_diff=$((current_time - last_mod))
    
    if [ $time_diff -gt $MAX_HANG_TIME ]; then
        echo "WARNING: Metrics file not updated for $time_diff seconds (threshold: $MAX_HANG_TIME seconds)"
        return 1
    else
        echo "Metrics file last updated $time_diff seconds ago"
        return 0
    fi
}

# Function to check if training process is still running
function is_training_running() {
    if kill -0 $TRAINING_PID 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to restart training
function restart_training() {
    local retry_count=$1
    echo "Restarting training (attempt $retry_count of $MAX_RETRIES)..."
    
    # Kill previous process if still running
    if is_training_running; then
        echo "Killing previous training process (PID $TRAINING_PID)..."
        kill -9 $TRAINING_PID
        sleep 5
    fi
    
    # Create a new directory for restarted run
    local new_output_dir="${OUTPUT_DIR}_restart_$retry_count"
    mkdir -p "$new_output_dir"
    local new_metrics_file="$new_output_dir/metrics.json"
    local new_log_file="$new_output_dir/training.log"
    
    # Update symlink
    rm -f ./output/latest_extended_training 2>/dev/null
    ln -sf "$new_output_dir" ./output/latest_extended_training
    
    # Copy previous model files if they exist
    if [ -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
        echo "Copying previous model checkpoint..."
        cp -r "$OUTPUT_DIR"/* "$new_output_dir/"
    fi
    
    echo "Starting new training process in $new_output_dir..."
    
    # Start new training process
    python scripts/long_train.py \
      --model_type=gpt \
      --d_model=768 \
      --num_heads=12 \
      --num_layers=12 \
      --d_ff=3072 \
      --max_seq_length=512 \
      --dropout=0.1 \
      --output_dir="$new_output_dir" \
      --batch_size=4 \
      --gradient_accumulation_steps=8 \
      --learning_rate=3e-5 \
      --num_train_epochs=10 \
      --max_train_time=12 \
      --warmup_ratio=0.1 \
      --fp16 \
      --logging_steps=50 \
      --eval_steps=500 \
      --save_steps=1000 \
      --dataset_name=wikitext \
      --dataset_config_name=wikitext-103-raw-v1 \
      --max_train_samples=50000 \
      --max_val_samples=5000 \
      --metrics_file="metrics.json" \
      --debug_mode \
      > "$new_log_file" 2>&1 &
    
    TRAINING_PID=$!
    echo "New training process started with PID $TRAINING_PID"
    OUTPUT_DIR=$new_output_dir
    METRICS_FILE=$new_metrics_file
    LOG_FILE=$new_log_file
    
    # Give the new process some time to start up
    sleep 30
}

# Monitor the training process
retry_count=0

while true; do
    # Check if the training process is still running
    if ! is_training_running; then
        echo "Training process ($TRAINING_PID) has exited."
        
        # Check if it completed successfully (need to check log for successful completion message)
        if grep -q "Training complete" "$LOG_FILE"; then
            echo "Training completed successfully!"
            break
        else
            echo "Training process ended unexpectedly."
            
            # Try to restart if we haven't exceeded max retries
            if [ $retry_count -lt $MAX_RETRIES ]; then
                retry_count=$((retry_count + 1))
                restart_training $retry_count
            else
                echo "Maximum retry attempts ($MAX_RETRIES) reached. Giving up."
                break
            fi
        fi
    fi
    
    # Check if the metrics file is being updated
    if ! check_metrics_file; then
        echo "Training appears to be hanging..."
        
        # Try to restart if we haven't exceeded max retries
        if [ $retry_count -lt $MAX_RETRIES ]; then
            retry_count=$((retry_count + 1))
            restart_training $retry_count
        else
            echo "Maximum retry attempts ($MAX_RETRIES) reached. Giving up."
            break
        fi
    fi
    
    # Wait for next check
    echo "Next check in $CHECK_INTERVAL seconds..."
    sleep $CHECK_INTERVAL
done

echo "Monitoring finished."

# Check the final training status
if grep -q "Training complete" "$LOG_FILE"; then
    echo "Final status: Training completed successfully"
    exit 0
else
    echo "Final status: Training did not complete successfully"
    exit 1
fi