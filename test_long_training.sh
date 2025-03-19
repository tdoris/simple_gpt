#!/bin/bash

# Script to test the long training script with minimal parameters for quick verification

# Create output directory
OUTPUT_DIR="./output/test_long_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Set environment variables for cache directories
export HF_HOME="./cache/huggingface"
export HF_DATASETS_CACHE="./cache/datasets"

echo "Starting test training with minimal parameters..."
echo "Output directory: $OUTPUT_DIR"

python scripts/long_train.py \
  --model_type=gpt \
  --d_model=256 \
  --num_heads=4 \
  --num_layers=4 \
  --d_ff=512 \
  --max_seq_length=128 \
  --dropout=0.1 \
  --output_dir="$OUTPUT_DIR" \
  --batch_size=2 \
  --gradient_accumulation_steps=2 \
  --learning_rate=5e-5 \
  --num_train_epochs=1 \
  --max_train_time=1 \
  --warmup_ratio=0.1 \
  --fp16 \
  --logging_steps=10 \
  --eval_steps=50 \
  --save_steps=50 \
  --dataset_name=wikitext \
  --dataset_config_name=wikitext-103-raw-v1 \
  --max_train_samples=500 \
  --max_val_samples=100 \
  --metrics_file=metrics.json \
  --debug_mode

# Create symlink to latest test training
ln -sf "$OUTPUT_DIR" ./output/latest_test_training

echo "Training started. Waiting for completion..."

# Function to check if model_config.json exists
function wait_for_completion {
    local timeout=600  # 10 minutes timeout
    local interval=10  # Check every 10 seconds
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if [ -f "$OUTPUT_DIR/model_config.json" ]; then
            echo "Training complete! Model config file found."
            return 0
        fi
        
        # Print status from log if available
        if [ -f "$OUTPUT_DIR/training.log" ]; then
            tail -n 1 "$OUTPUT_DIR/training.log"
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        echo "Waiting for training to complete... ($elapsed seconds elapsed)"
    done
    
    echo "Timeout waiting for training to complete."
    return 1
}

wait_for_completion

echo "Now testing text generation..."

# Test text generation with the trained model
./run_text_generation.sh "$OUTPUT_DIR" "The quick brown fox" --max_length 30 --temperature 0.8

echo "Test complete!"