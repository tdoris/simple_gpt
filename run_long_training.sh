#!/bin/bash

# Script to run a long training session

# Create output directory
OUTPUT_DIR="./output/extended_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Set environment variables for cache directories
export HF_HOME="./cache/huggingface"
export HF_DATASETS_CACHE="./cache/datasets"

# Run the training script
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

# Create a symlink to the latest training
ln -sf "$OUTPUT_DIR" ./output/latest_extended_training

echo "Training complete. Results saved to $OUTPUT_DIR"
