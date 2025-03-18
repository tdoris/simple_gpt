#!/bin/bash

# Script to run training directly (without screen)

# Create output directory with timestamp
OUTPUT_DIR="./output/extended_training_$(date +%Y%m%d_%H%M%S)"
METRICS_FILE="$OUTPUT_DIR/metrics.json"
LOG_FILE="$OUTPUT_DIR/training.log"

echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Create a symlink to the latest training
rm -f ./output/latest_training 2>/dev/null
ln -sf "$OUTPUT_DIR" ./output/latest_training

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi

# Test GPU with a small model
echo "Running GPU test with a small model..."
python test_gpu.py

# Run the training script
echo "Starting training at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Metrics file: $METRICS_FILE"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Start training with output redirected to log file
echo "Starting training, logs will be written to $LOG_FILE"
(
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
    --metrics_file=metrics.json
) > "$LOG_FILE" 2>&1 &

TRAINING_PID=$!
echo "Training started with PID: $TRAINING_PID"
echo "To monitor training progress, run: ./check_progress.sh"
echo "To see the logs, run: tail -f $LOG_FILE"

# Create a README in the output directory
cat > "$OUTPUT_DIR/README.txt" << EOF
Training Information
===================

This directory contains the results of a long training run started on $(date).

Key files:
- pytorch_model.bin: The trained model weights
- model_config.json: Model architecture configuration
- training_config.json: Training parameters
- metrics.json: Training metrics logged during training
- training.log: Training log output

To monitor training progress:
  ./check_progress.sh

To generate text with this model:
  python -m simple_gpt.scripts.generate --model_path="$OUTPUT_DIR" --prompt="Your prompt here" --max_length=200 --do_sample --temperature=0.8
EOF

echo "Created README in $OUTPUT_DIR"
echo "Training process is running in the background (PID: $TRAINING_PID)"
echo "Use 'ps -p $TRAINING_PID' to check if it's still running"