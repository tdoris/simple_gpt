#!/bin/bash

# Script to continue training from a checkpoint using the same long_train.py script

# Default values
MODEL_PATH=""
OUTPUT_DIR=""
NUM_EPOCHS=5
# Default to using entire dataset (empty means use all)
TRAIN_SAMPLES=""
VAL_SAMPLES=""
BATCH_SIZE=""
LEARNING_RATE=""
SLIMPAJAMA_SUBSET="all"
USE_SLIMPAJAMA=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --train-samples)
            TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --val-samples)
            VAL_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --use-slimpajama)
            USE_SLIMPAJAMA=true
            shift
            ;;
        --slimpajama-subset)
            SLIMPAJAMA_SUBSET="$2"
            USE_SLIMPAJAMA=true
            shift 2
            ;;
        --help)
            echo "Usage: $0 --model-path <path> [options]"
            echo ""
            echo "Options:"
            echo "  --model-path <path>         Path to the model checkpoint (required)"
            echo "  --output-dir <path>         Directory to save the continued model"
            echo "  --epochs <num>              Number of training epochs (default: 5)"
            echo "  --train-samples <num>       Limit training samples (default: all)"
            echo "  --val-samples <num>         Limit validation samples (default: all)"
            echo "  --batch-size <num>          Batch size (default: use value from checkpoint)"
            echo "  --learning-rate <num>       Learning rate (default: use value from checkpoint)"
            echo "  --use-slimpajama            Use SlimPajama dataset instead of WikiText"
            echo "  --slimpajama-subset <name>  SlimPajama subset (default: all)"
            echo ""
            echo "Example:"
            echo "  $0 --model-path ./output/my_model --epochs 3"
            echo "  $0 --model-path ./output/my_model --epochs 3 --train-samples 5000"
            echo "  $0 --model-path ./output/my_model --use-slimpajama --slimpajama-subset arxiv"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Check for required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model-path is required"
    echo "Run '$0 --help' for usage information."
    exit 1
fi

# Verify the model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist"
    exit 1
fi

# Set default output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${MODEL_PATH}_continued_$(date +%Y%m%d_%H%M%S)"
fi

# Set environment variables for cache directories
export HF_HOME="./cache/huggingface"
export HF_DATASETS_CACHE="./cache/datasets"

# Extract model configuration from model_config.json
MODEL_CONFIG_PATH="$MODEL_PATH/model_config.json"
if [ ! -f "$MODEL_CONFIG_PATH" ]; then
    echo "Error: Model configuration file not found at $MODEL_CONFIG_PATH"
    exit 1
fi

# Parse model configuration using Python
MODEL_PARAMS=$(python -c "
import json
with open('$MODEL_CONFIG_PATH', 'r') as f:
    config = json.load(f)
print(f\"--model_type={config.get('model_type', 'gpt')} --d_model={config.get('d_model', 768)} --num_heads={config.get('num_heads', 12)} --num_layers={config.get('num_layers', 12)} --d_ff={config.get('d_ff', 3072)} --max_seq_length={config.get('max_seq_length', 1024)} --dropout={config.get('dropout', 0.1)}\")
")

# Build the command using long_train.py with the checkpoint path
CMD="python scripts/long_train.py \
  $MODEL_PARAMS \
  --output_dir=\"$OUTPUT_DIR\" \
  --num_train_epochs=$NUM_EPOCHS \
  --checkpoint_path=\"$MODEL_PATH\" \
  --fp16 \
  --logging_steps=50 \
  --eval_steps=500 \
  --save_steps=1000 \
  --metrics_file=\"metrics.json\""

# Add sample limits only if specified
if [ ! -z "$TRAIN_SAMPLES" ]; then
    CMD="$CMD --max_train_samples=$TRAIN_SAMPLES"
fi

if [ ! -z "$VAL_SAMPLES" ]; then
    CMD="$CMD --max_val_samples=$VAL_SAMPLES"
fi

# Add optional arguments if provided
if [ ! -z "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size=$BATCH_SIZE"
fi

if [ ! -z "$LEARNING_RATE" ]; then
    CMD="$CMD --learning_rate=$LEARNING_RATE"
fi

# Add SlimPajama options if selected
if [ "$USE_SLIMPAJAMA" = true ]; then
    CMD="$CMD --use_slimpajama --slimpajama_subset=$SLIMPAJAMA_SUBSET"
fi

# Display information
echo "Continuing training with the following configuration:"
echo "- Model path: $MODEL_PATH"
echo "- Output directory: $OUTPUT_DIR"
echo "- Training epochs: $NUM_EPOCHS"

if [ ! -z "$TRAIN_SAMPLES" ]; then
    echo "- Training samples: $TRAIN_SAMPLES (limited)"
else
    echo "- Training samples: ALL (using entire dataset)"
fi

if [ ! -z "$VAL_SAMPLES" ]; then
    echo "- Validation samples: $VAL_SAMPLES (limited)"
else
    echo "- Validation samples: ALL (using entire validation set)"
fi

if [ ! -z "$BATCH_SIZE" ]; then
    echo "- Batch size: $BATCH_SIZE"
fi

if [ ! -z "$LEARNING_RATE" ]; then
    echo "- Learning rate: $LEARNING_RATE"
fi

if [ "$USE_SLIMPAJAMA" = true ]; then
    echo "- Dataset: SlimPajama ($SLIMPAJAMA_SUBSET)"
else
    echo "- Dataset: WikiText (default)"
fi

# Run the command
echo "Running command: $CMD"
eval $CMD

echo "Training complete!"