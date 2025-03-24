#!/bin/bash

# Script to run training on SlimPajama-30B dataset

# Default values
SLIMPAJAMA_SUBSET="all"
# Default to using entire dataset (empty means use all)
MAX_TRAIN_SAMPLES=""
MAX_VAL_SAMPLES=""
NUM_EPOCHS=30
MAX_TRAIN_TIME=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --subset)
            SLIMPAJAMA_SUBSET="$2"
            shift 2
            ;;
        --train-samples)
            MAX_TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --val-samples)
            MAX_VAL_SAMPLES="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --max-time)
            MAX_TRAIN_TIME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --subset <name>         SlimPajama subset (all, arxiv, github, etc.)"
            echo "  --train-samples <num>   Limit training samples (default: all)"
            echo "  --val-samples <num>     Limit validation samples (default: all)"
            echo "  --epochs <num>          Number of training epochs"
            echo "  --max-time <hours>      Maximum training time in hours"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Create output directory with timestamp and subset name
OUTPUT_DIR="./output/slimpajama_${SLIMPAJAMA_SUBSET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Set environment variables for cache directories
export HF_HOME="./cache/huggingface"
export HF_DATASETS_CACHE="./cache/datasets"

echo "Starting SlimPajama-30B training with the following configuration:"
echo "- Subset: $SLIMPAJAMA_SUBSET"
if [ ! -z "$MAX_TRAIN_SAMPLES" ]; then
    echo "- Training samples: $MAX_TRAIN_SAMPLES (limited)"
else
    echo "- Training samples: ALL (using entire dataset)"
fi
if [ ! -z "$MAX_VAL_SAMPLES" ]; then
    echo "- Validation samples: $MAX_VAL_SAMPLES (limited)"
else
    echo "- Validation samples: ALL (using entire validation set)"
fi
echo "- Epochs: $NUM_EPOCHS"
echo "- Max training time: $MAX_TRAIN_TIME hours (0 = no limit)"
echo "- Output directory: $OUTPUT_DIR"

# Build the command
TRAIN_CMD="python scripts/long_train.py \
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
  --num_train_epochs=$NUM_EPOCHS \
  --max_train_time=$MAX_TRAIN_TIME \
  --warmup_ratio=0.1 \
  --fp16 \
  --logging_steps=50 \
  --eval_steps=500 \
  --save_steps=1000 \
  --use_slimpajama \
  --slimpajama_subset=$SLIMPAJAMA_SUBSET \
  --metrics_file=\"metrics.json\""

# Add sample limits only if specified
if [ ! -z "$MAX_TRAIN_SAMPLES" ]; then
  TRAIN_CMD="$TRAIN_CMD --max_train_samples=$MAX_TRAIN_SAMPLES"
fi

if [ ! -z "$MAX_VAL_SAMPLES" ]; then
  TRAIN_CMD="$TRAIN_CMD --max_val_samples=$MAX_VAL_SAMPLES"
fi

# Run the command
echo "Running command: $TRAIN_CMD"
eval $TRAIN_CMD

# Create a symlink to the latest training
ln -sf "$OUTPUT_DIR" ./output/latest_slimpajama_training

echo "Training complete. Results saved to $OUTPUT_DIR"