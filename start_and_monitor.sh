#!/bin/bash

# Script to start the long training and monitor progress

# Define output directory
OUTPUT_DIR="./output/extended_training_$(date +%Y%m%d_%H%M%S)"
METRICS_FILE="$OUTPUT_DIR/metrics.json"
SCREEN_NAME="simplegpt_training"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start training in a screen session
screen -dmS $SCREEN_NAME bash -c "
    # Print useful information
    echo \"Starting training at $(date)\"
    echo \"Output directory: $OUTPUT_DIR\"
    echo \"Log file: $OUTPUT_DIR/training.log\"
    echo \"Metrics file: $METRICS_FILE\"
    
    # Set environment variables
    export HF_HOME=\"./cache/huggingface\"
    export HF_DATASETS_CACHE=\"./cache/datasets\"
    
    # Run the training script
    python scripts/long_train.py \
      --model_type=gpt \
      --d_model=768 \
      --num_heads=12 \
      --num_layers=12 \
      --d_ff=3072 \
      --max_seq_length=512 \
      --dropout=0.1 \
      --output_dir=\"$OUTPUT_DIR\" \
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
    ln -sf \"$OUTPUT_DIR\" ./output/latest_extended_training
    
    echo \"Training complete at $(date). Results saved to $OUTPUT_DIR\"
"

echo "Training started in screen session '$SCREEN_NAME'"
echo "To attach to the training session, run: screen -r $SCREEN_NAME"
echo "To detach from the session without stopping it, press Ctrl+A followed by D"

# Wait a moment for the metrics file to be created
sleep 5

# Start the monitor in the foreground
echo "Starting training monitor..."
echo "Press Ctrl+C to exit monitor (training will continue in the background)"
python scripts/monitor_training.py --metrics_file="$METRICS_FILE" --refresh_rate=5

echo "Monitor stopped, but training continues in the background"
echo "To check status, run: screen -r $SCREEN_NAME"
echo "To generate plots after training completes, run:"
echo "  python scripts/plot_metrics.py --metrics_file=\"$METRICS_FILE\""

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
  python scripts/monitor_training.py --metrics_file="$METRICS_FILE"

To generate plots from metrics:
  python scripts/plot_metrics.py --metrics_file="$METRICS_FILE"

To generate text with this model:
  python -m simple_gpt.scripts.generate --model_path="$OUTPUT_DIR" --prompt="Your prompt here" --max_length=200 --do_sample --temperature=0.8
EOF

echo "Created README in $OUTPUT_DIR"